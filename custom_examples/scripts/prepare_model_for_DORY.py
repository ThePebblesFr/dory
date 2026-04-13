#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from helpers.model_prep import (
    add_quantisation_transform,
    all_pact_create_configs_int8,
    all_pact_f2f_recipe,
    build_plain_mlp_from_converted_model,
    export_reference_outputs,
    build_fused_plan_from_onnx,
    f2t_convert,
    generate_dummy_vector_dataset,
    get_input_range,
)
from helpers.onnx_utils import (
    eliminate_identity_nodes,
    fix_linear_metadata,
    get_onnx_input_shape,
    load_net_from_onnx,
    normalize_quantlab_onnx_for_dory,
    rename_all_tensors_to_numeric,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a DORY-ready ONNX and reference txt files from an original ONNX model."
    )
    parser.add_argument("--model", required=True, help="Path to original ONNX model, usually under custom_examples/models/.")
    parser.add_argument("--name", required=True, help="Logical model name. Outputs are written to outputs/<name>/.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for deterministic dummy input generation.")
    parser.add_argument("--samples", type=int, default=1, help="Number of dummy samples used for calibration. Default: 1.")
    parser.add_argument("--device", default="cpu", help="Torch device for conversion, e.g. cpu or cuda.")
    parser.add_argument("--keep-softmax", action="store_true", help="Keep softmax when rebuilding the plain MLP. Disabled by default because DORY references usually use argmax on logits.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    root = Path(__file__).resolve().parents[1]
    models_dir = root / "models"
    outputs_dir = root / "outputs" / args.name
    outputs_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = (root / model_path).resolve() if (root / model_path).exists() else (models_dir / model_path.name).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Reading ONNX from: {model_path}")

    # 1) Load and normalize converted ONNX model.
    converted = load_net_from_onnx(str(model_path), device)
    fix_linear_metadata(converted)

    # 2) Rebuild a plain MLP structure from the converted model.
    plain = build_plain_mlp_from_converted_model(converted, device=device, with_softmax=args.keep_softmax)
    print("Rebuilt plain sequential MLP from converted ONNX.")

    # 3) Prepare a deterministic dummy dataset from the ONNX input shape.
    input_shape = get_onnx_input_shape(str(model_path))
    if len(input_shape) != 2 or input_shape[0] != 1:
        raise ValueError(
            f"This first version expects a single-sample vector input of shape (1, N). Got {input_shape}. "
            "Add a dedicated helper for your input format before using this script on that model."
        )
    input_dim = input_shape[1]
    dataset = generate_dummy_vector_dataset(input_dim=input_dim, n_samples=max(1, args.samples), seed=args.seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # 4) Fake-to-fake PACT setup.
    min_, max_ = get_input_range(loader)
    input_eps = add_quantisation_transform(loader, n_levels=256, min_=min_, max_=max_)
    print(f"Input epsilon: {float(input_eps)}")

    name2config = all_pact_create_configs_int8(plain, patches={})
    pact_model = all_pact_f2f_recipe(plain, name2config)
    pact_model = pact_model.to(device)
    pact_model.eval()

    # 5) Fake-to-true integerization.
    tq_model = f2t_convert(loader, input_eps, pact_model)
    tq_model.eval()
    tq_model = tq_model.to(torch.device("cpu"))
    print("Integerized model with QuantLib fake-to-true pass.")

    # Use the exact same sample for ONNX export and reference generation.
    x0, _ = loader.dataset[0]
    sample = x0.unsqueeze(0).to(torch.device("cpu"))

        # 6) Export integerized ONNX.
    export_input = sample
    onnx_tmp = outputs_dir / f"{args.name}__tmp_int.onnx"
    onnx_no_identity = outputs_dir / f"{args.name}__tmp_no_identity.onnx"
    onnx_debug = outputs_dir / f"{args.name}__debug.onnx"
    onnx_final = outputs_dir / f"{args.name}.onnx"   # DORY-facing ONNX

    torch.onnx.export(
        tq_model,
        export_input,
        str(onnx_tmp),
        input_names=["input"],
        output_names=["output"],
        opset_version=9,
        do_constant_folding=True,
    )

    eliminate_identity_nodes(str(onnx_tmp), str(onnx_no_identity))

    # Keep one ONNX that mirrors the raw QuantLib export structure for debugging/reference parsing.
    debug_name_map = rename_all_tensors_to_numeric(str(onnx_no_identity), str(onnx_debug))

    # Build a DORY-facing normalized ONNX by removing the extra Add(+0.5) nodes.
    rewritten = normalize_quantlab_onnx_for_dory(str(onnx_debug), str(onnx_final))
    print(f"Normalized {rewritten} requant block(s) for DORY compatibility.")

    # Optional: rename tensors in the normalized ONNX too, for consistency.
    # If you want numeric names there as well, run a second rename pass:
    onnx_final_renamed = outputs_dir / f"{args.name}__dory_tmp.onnx"
    final_name_map = rename_all_tensors_to_numeric(str(onnx_final), str(onnx_final_renamed))
    onnx_final.unlink()
    onnx_final_renamed.rename(onnx_final)

    # Remove intermediate ONNX files.
    for tmp in (onnx_tmp, onnx_no_identity):
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass

    # 7) Build the fused execution plan from the FINAL exported ONNX.
    # This makes the txt references follow exactly what DORY will parse.
    fused_plan = build_fused_plan_from_onnx(onnx_debug)
    print("Fused execution plan parsed from final ONNX:")
    for i, layer in enumerate(fused_plan):
        print(f"  {i}: {layer.kind} :: {layer.name} :: W{layer.weights.shape} B{layer.bias.shape}")

    # 8) Export reference txt using exact PULP math on the ONNX-derived plan.
    x_u8 = np.clip(np.rint(x0.cpu().numpy()), 0, 255).astype(np.uint8).reshape(-1)
    checksums = export_reference_outputs(fused_plan, input_u8=x_u8, output_dir=outputs_dir)

    # 9) Generate a DORY config JSON next to the outputs.
    dory_config = {
        "BNRelu_bits": 64,
        "onnx_file": f"./{args.name}.onnx",
        "code reserved space": 92000,
        "input_bits": 8,
        "input_signed": False,
    }
    config_path = outputs_dir / f"config_Quantlab_{args.name}.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(dory_config, f, indent=2)

    # 10) Save metadata for debugging / later automation.
    metadata = {
        "source_model": str(model_path),
        "generated_onnx_debug": onnx_debug.name,
        "generated_onnx_dory": onnx_final.name,
        "generated_config": config_path.name,
        "dummy_seed": args.seed,
        "input_shape": list(input_shape),
        "checksums": checksums,
        "debug_name_map_preview": dict(list(debug_name_map.items())[:20]),
        "dory_name_map_preview": dict(list(final_name_map.items())[:20]),
        "fused_plan": [
            {
                "index": i,
                "name": layer.name,
                "kind": layer.kind,
                "weights_shape": list(layer.weights.shape),
                "bias_shape": list(layer.bias.shape),
                "out_mult": layer.out_mult,
                "out_shift": layer.out_shift,
            }
            for i, layer in enumerate(fused_plan)
        ],
    }
    with open(outputs_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nDone.")
    print(f"Artifacts written to: {outputs_dir}")
    print(f"  - {onnx_final.name}")
    print("  - input.txt")
    for i in range(len(fused_plan)):
        print(f"  - out_layer{i}.txt")
    print(f"  - {config_path.name}")
    print("  - metadata.json")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except NotImplementedError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
