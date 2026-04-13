from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import onnx
from onnx import numpy_helper
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import quantlib.editing.lightweight as qlw
import quantlib.algorithms as qa
import quantlib.editing.fx as qfx

from helpers.pulp_nn_math import (
    fc_u8_i32_i8,
    fc_u8_u8_i8,
    pulp_nn_quant_u8,
    clip8,
)


# ---------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------

class VectorDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


def generate_dummy_vector_dataset(
    input_dim: int,
    n_samples: int = 1,
    seed: int = 0,
) -> Dataset:
    g = torch.Generator().manual_seed(seed)
    X_dummy = torch.randn(n_samples, input_dim, generator=g)
    y_dummy = torch.zeros(n_samples, dtype=torch.long)
    return VectorDataset(X_dummy, y_dummy, transform=None)


# ---------------------------------------------------------------------
# Build plain MLP from converted ONNX model
# ---------------------------------------------------------------------

def _ordered_linears(src_model: nn.Module) -> List[nn.Linear]:
    return [m for m in src_model.modules() if isinstance(m, nn.Linear)]


def build_plain_mlp_from_converted_model(
    src_model: nn.Module,
    device: torch.device,
    with_softmax: bool = False,
) -> nn.Module:
    """
    Generalises the notebook's hardcoded 4-layer PlainMLP:
    rebuild a plain Sequential as
        Linear -> ReLU -> Linear -> ReLU -> ... -> Linear [-> Softmax]
    using the exact Linear shapes from the converted model.
    """
    src_linears = _ordered_linears(src_model)
    if not src_linears:
        raise ValueError("No Linear layers found in converted ONNX model.")

    layers: List[nn.Module] = []
    for i, src in enumerate(src_linears):
        dst = nn.Linear(src.in_features, src.out_features, bias=(src.bias is not None))
        dst.weight.data.copy_(src.weight.data)
        if src.bias is not None:
            dst.bias.data.copy_(src.bias.data)
        layers.append(dst)

        is_last = i == len(src_linears) - 1
        if not is_last:
            layers.append(nn.ReLU())

    if with_softmax:
        layers.append(nn.Softmax(dim=1))

    model = nn.Sequential(*layers).to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------
# PACT fake-to-fake helpers
# ---------------------------------------------------------------------

def all_pact_f2f_recipe(network: nn.Module, name2config: Dict[str, Dict]) -> nn.Module:
    lwg = qlw.LightweightGraph(network)
    name2type = {n.name: n.module.__class__.__name__ for n in lwg.nodes_list}

    assert set(name2config.keys()).issubset(set(name2type.keys()))

    type2rule = {
        "Linear": qlw.rules.pact.ReplaceConvLinearPACTRule,
        "ReLU": qlw.rules.pact.ReplaceActPACTRule,
    }

    rhos = [
        type2rule[name2type[n]](qlw.rules.NameFilter(n), **name2config[n])
        for n in name2config.keys()
    ]

    lwe = qlw.LightweightEditor(lwg)
    lwe.startup()
    for rho in rhos:
        lwe.set_lwr(rho)
        lwe.apply()
    lwe.shutdown()

    return lwe.graph.net


def all_pact_create_configs_int8(
    network: nn.Module,
    patches: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Dict]:
    if patches is None:
        patches = {}

    lwg = qlw.LightweightGraph(network)

    linear_nodes = {n.name for n in lwg.nodes_list if n.module.__class__.__name__ == "Linear"}
    relu_nodes = {n.name for n in lwg.nodes_list if n.module.__class__.__name__ == "ReLU"}

    assert set(patches.keys()).issubset(linear_nodes | relu_nodes)

    linear_default = {
        "quantize": "per_layer",
        "init_clip": "sawb_asymm",
        "learn_clip": False,
        "symm_wts": True,
        "tqt": False,
        "n_levels": 256,
    }

    relu_default = {
        "init_clip": "std",
        "learn_clip": True,
        "nb_std": 3,
        "rounding": False,
        "tqt": False,
        "n_levels": 256,
    }

    out: Dict[str, Dict] = {}
    for n in linear_nodes:
        cfg = linear_default.copy()
        cfg.update(patches.get(n, {}))
        out[n] = cfg

    for n in relu_nodes:
        cfg = relu_default.copy()
        cfg.update(patches.get(n, {}))
        out[n] = cfg

    return out


# ---------------------------------------------------------------------
# Input quantisation helpers
# ---------------------------------------------------------------------

def get_input_range(data_loader: DataLoader) -> Tuple[float, float]:
    min_ = 0.0
    max_ = 0.0

    for x, _ in data_loader:
        min_ = min(min_, float(x.min().item()))
        max_ = max(max_, float(x.max().item()))

    return min_, max_


def add_quantisation_transform(
    data_loader: DataLoader,
    n_levels: int,
    min_: float,
    max_: float,
) -> float:
    quantiser = qa.pact.PACTAsymmetricAct(
        n_levels=n_levels,
        symm=True,
        learn_clip=False,
        init_clip="max",
        act_kind="identity",
    )

    clip_lo, clip_hi = qa.pact.util.almost_symm_quant(
        torch.tensor([max(abs(min_), abs(max_))]),
        n_levels,
    )

    quantiser.clip_lo.data = clip_lo
    quantiser.clip_hi.data = clip_hi
    quantiser.started |= True

    transform_list = []
    if getattr(data_loader.dataset, "transform", None) is not None:
        transform_list.append(data_loader.dataset.transform)

    transform_list.append(quantiser)
    transform_list.append(transforms.Lambda(lambda x: x / quantiser.get_eps()))

    data_loader.dataset.transform = transforms.Compose(transform_list)
    return float(quantiser.get_eps())


def f2t_convert(
    dataloader: DataLoader,
    input_eps: float,
    network: nn.Module,
) -> nn.Module:
    network.eval()
    network = network.to(torch.device("cpu"))

    x, _ = dataloader.dataset.__getitem__(0)
    x = x.unsqueeze(0).to(torch.device("cpu"))

    fake2true_converter = qfx.passes.pact.IntegerizePACTNetPass(
        shape_in=x.shape,
        eps_in=input_eps,
        D=2**19,
    )

    return fake2true_converter(network)


# ---------------------------------------------------------------------
# Exact integer reference export
# ---------------------------------------------------------------------

@dataclass
class FusedLayer:
    name: str
    kind: str
    weights: np.ndarray       # int8, shape [out, in]
    bias: np.ndarray          # int32, shape [out]
    out_mult: Optional[int]   # None for final non-requantized output layer
    out_shift: Optional[int]  # None for final non-requantized output layer


def _tensor_map(model: onnx.ModelProto) -> Dict[str, np.ndarray]:
    return {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}


def _producer_map(model: onnx.ModelProto) -> Dict[str, onnx.NodeProto]:
    out = {}
    for node in model.graph.node:
        for name in node.output:
            out[name] = node
    return out


def _consumer_map(model: onnx.ModelProto) -> Dict[str, List[onnx.NodeProto]]:
    out: Dict[str, List[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for name in node.input:
            out.setdefault(name, []).append(node)
    return out


def _const_scalar(initializers: Dict[str, np.ndarray], name: str) -> Optional[float]:
    arr = initializers.get(name)
    if arr is None:
        return None
    flat = np.asarray(arr).reshape(-1)
    if flat.size != 1:
        return None
    return float(flat[0])


def _extract_gemm_or_matmul_bias(
    node: onnx.NodeProto,
    initializers: Dict[str, np.ndarray],
    consumers: Dict[str, List[onnx.NodeProto]],
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Returns (weights, bias, effective_output_name).
    Supports:
      - Gemm(X, W, B)
      - MatMul(X, W) -> Add(..., B)
    """
    if node.op_type == "Gemm":
        if len(node.input) < 2:
            raise ValueError(f"Gemm node {node.name or '<unnamed>'} missing weight input")

        x_name, w_name = node.input[0], node.input[1]
        b_name = node.input[2] if len(node.input) >= 3 else None

        if w_name not in initializers:
            raise ValueError(f"Gemm node {node.name or '<unnamed>'} has non-constant weights")
        w = np.asarray(initializers[w_name])

        # ONNX Gemm commonly stores W as [in, out] with transB=0.
        # We want [out, in].
        transB = 0
        for attr in node.attribute:
            if attr.name == "transB":
                transB = int(attr.i)
        w2 = w if transB == 1 else w.T

        if b_name is None:
            b = np.zeros(w2.shape[0], dtype=np.int32)
        else:
            if b_name not in initializers:
                raise ValueError(f"Gemm node {node.name or '<unnamed>'} has non-constant bias")
            b = np.asarray(initializers[b_name]).reshape(-1)

        return np.rint(w2).astype(np.int8), np.rint(b).astype(np.int32), node.output[0]

    if node.op_type == "MatMul":
        if len(node.input) != 2:
            raise ValueError(f"MatMul node {node.name or '<unnamed>'} malformed")

        w_name = node.input[1]
        if w_name not in initializers:
            raise ValueError(f"MatMul node {node.name or '<unnamed>'} has non-constant weights")

        w = np.asarray(initializers[w_name])

        # ONNX MatMul with row-vector input: [1,in] @ [in,out] -> [1,out]
        # Convert to torch Linear layout [out,in].
        if w.ndim != 2:
            raise ValueError(f"MatMul weights for {node.name or '<unnamed>'} are not rank-2")
        w2 = w.T

        b = np.zeros(w2.shape[0], dtype=np.int32)
        effective_out = node.output[0]

        node_consumers = consumers.get(node.output[0], [])
        if len(node_consumers) == 1 and node_consumers[0].op_type == "Add":
            add_node = node_consumers[0]
            other = add_node.input[0] if add_node.input[1] == node.output[0] else add_node.input[1]
            if other in initializers:
                b = np.asarray(initializers[other]).reshape(-1)
                effective_out = add_node.output[0]

        return np.rint(w2).astype(np.int8), np.rint(b).astype(np.int32), effective_out

    raise ValueError(f"Unsupported FC op: {node.op_type}")


def _follow_passthrough_nodes(
    tensor: str,
    consumers: Dict[str, List[onnx.NodeProto]],
) -> str:
    """
    Skip harmless structural / activation nodes that do not change the integer scale.
    """
    while True:
        next_nodes = consumers.get(tensor, [])
        if len(next_nodes) != 1:
            return tensor

        node = next_nodes[0]
        if node.op_type in {"Relu", "Clip", "Cast", "Identity", "Flatten", "Reshape"}:
            tensor = node.output[0]
            continue

        return tensor


def _follow_requant_params(
    start_tensor: str,
    initializers: Dict[str, np.ndarray],
    consumers: Dict[str, List[onnx.NodeProto]],
    is_last_fc: bool,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract a *local* requant pattern only.

    Expected hidden-layer pattern:
        FC_out
          -> [optional Relu/Clip/Cast/...]
          -> Mul(const_m) or Div(const_d)
          -> [optional one more Mul/Div(const)]
          -> [optional Relu/Clip/Cast/...]
          -> next FC

    We do NOT walk arbitrarily far through the graph anymore.
    """
    if is_last_fc:
        return None, None

    tensor = _follow_passthrough_nodes(start_tensor, consumers)

    mult = 1.0
    div = 1.0
    n_scale_ops = 0

    for _ in range(2):  # at most two scale ops: one Mul and/or one Div
        next_nodes = consumers.get(tensor, [])
        if len(next_nodes) != 1:
            break

        node = next_nodes[0]
        if node.op_type not in {"Mul", "Div"}:
            break

        other = node.input[0] if node.input[1] == tensor else node.input[1]
        c = _const_scalar(initializers, other)
        if c is None:
            break

        if node.op_type == "Mul":
            mult *= c
        else:
            div *= c

        n_scale_ops += 1
        tensor = node.output[0]
        tensor = _follow_passthrough_nodes(tensor, consumers)

    if n_scale_ops == 0:
        raise NotImplementedError(
            "No local requant pattern found after hidden FC layer."
        )

    eff = mult / div
    if eff <= 0:
        raise ValueError(f"Invalid effective requant scale: {eff}")

    # Prefer exact power-of-two divisor if possible.
    # Search a small family of shifts and keep the first int16-fit solution
    # with smallest reconstruction error.
    best = None
    for d in range(0, 31):
        m = int(round(eff * (2 ** d)))
        if -32768 <= m <= 32767:
            approx = m / float(2 ** d)
            err = abs(approx - eff)
            if best is None or err < best[2]:
                best = (m, d, err)

    if best is None:
        raise ValueError(
            f"Inferred effective scale {eff} cannot be represented as int16 / 2**d"
        )

    m, d, _ = best
    return m, d


def build_fused_plan_from_onnx(onnx_path: Path | str) -> List[FusedLayer]:
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    initializers = _tensor_map(model)
    consumers = _consumer_map(model)

    fc_nodes = [n for n in model.graph.node if n.op_type in {"Gemm", "MatMul"}]
    if not fc_nodes:
        raise ValueError(f"No Gemm/MatMul nodes found in {onnx_path}")

    fused: List[FusedLayer] = []
    for i, node in enumerate(fc_nodes):
        w_i8, b_i32, eff_out = _extract_gemm_or_matmul_bias(node, initializers, consumers)
        is_last = i == len(fc_nodes) - 1
        out_mult, out_shift = _follow_requant_params(
            eff_out,
            initializers=initializers,
            consumers=consumers,
            is_last_fc=is_last,
        )

        layer_kind = "Linear" if (out_mult is None or out_shift is None) else "Linear+RequantShift"

        fused.append(
            FusedLayer(
                name=node.name or f"fc_{i}",
                kind=layer_kind,
                weights=w_i8,
                bias=b_i32,
                out_mult=out_mult,
                out_shift=out_shift,
            )
        )

    return fused

def export_reference_outputs(fused_plan, input_u8, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x = np.asarray(input_u8, dtype=np.uint8).reshape(-1)
    x_dbg = x.squeeze(0).cpu().numpy().reshape(-1)
    print("\nInput tensor stats before txt export:")
    print("  min:", x_dbg.min())
    print("  max:", x_dbg.max())
    print("  all_integer:", np.allclose(x_dbg, np.rint(x_dbg)))
    print("  n_negative:", int((x_dbg < 0).sum()))
    print("  first_16:", x_dbg[:16])
    np.savetxt(output_dir / "input.txt", x.reshape(-1, 1), fmt="%d", delimiter=",")

    checksums = {"input.txt": int(x.astype(np.int64).sum())}
    cur = x

    for i, layer in enumerate(fused_plan):
        if layer.kind == "Linear+RequantShift":
            y = fc_u8_u8_i8(cur, layer.weights, layer.bias, layer.out_mult, layer.out_shift)
        elif layer.kind == "Linear":
            y = fc_u8_i32_i8(cur, layer.weights, layer.bias)
        else:
            raise ValueError(f"Unsupported fused layer kind: {layer.kind}")

        filename = f"out_layer{i}.txt"
        np.savetxt(output_dir / filename, y.reshape(-1, 1), fmt="%d", delimiter=",")
        checksums[filename] = int(np.asarray(y, dtype=np.int64).sum())

        if y.dtype == np.uint8:
            cur = y
        else:
            cur = y  # final int32 logits typically end here

    return checksums