from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import onnx
import onnxoptimizer
import torch
import torch.nn as nn
from onnx import numpy_helper
from onnx2pytorch import ConvertModel
import numpy as np

from collections import Counter

def _init_map(model):
    return {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}

def _const_node_map(model):
    out = {}
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    out[node.output[0]] = numpy_helper.to_array(attr.t)
    return out

def dump_requant_constants(onnx_path: str) -> None:
    model = onnx.load(onnx_path)
    init_map = _init_map(model)
    const_map = _const_node_map(model)

    def get_scalar(name):
        if name in init_map:
            arr = np.asarray(init_map[name]).reshape(-1)
            if arr.size == 1:
                return float(arr[0])
        if name in const_map:
            arr = np.asarray(const_map[name]).reshape(-1)
            if arr.size == 1:
                return float(arr[0])
        return None

    for i, node in enumerate(model.graph.node):
        if node.op_type != "Mul":
            continue

        print(f"\nRequant block starting at Mul #{i}: {node.name}")
        for j in range(i, min(i + 7, len(model.graph.node))):
            n = model.graph.node[j]
            vals = []
            for x in n.input:
                v = get_scalar(x)
                if v is not None:
                    vals.append((x, v))
            print(f"  {j:02d}: {n.op_type} :: {n.name}")
            if vals:
                for name, v in vals:
                    print(f"      const {name} = {v}")


def load_net_from_onnx(model_path: str, device: torch.device) -> nn.Module:
    onnx_model = onnx.load(model_path)
    pytorch_model = ConvertModel(onnx_model)
    # Keep the ONNX around because the caller uses model.onnx_model in places.
    pytorch_model.onnx_model = onnx_model
    pytorch_model.to(device)
    pytorch_model.eval()
    return pytorch_model


def fix_linear_metadata(model: nn.Module) -> None:
    """
    onnx2pytorch sometimes gives Linear modules with stale in_features/out_features
    metadata even though weight tensors are correct.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            w_out, w_in = module.weight.shape
            if module.out_features != w_out or module.in_features != w_in:
                print(
                    f"Fixing {name}: "
                    f"(in={module.in_features}, out={module.out_features}) "
                    f"-> (in={w_in}, out={w_out})"
                )
                module.in_features = w_in
                module.out_features = w_out


def get_onnx_input_shape(model_path: str) -> List[int]:
    model = onnx.load(model_path)
    if not model.graph.input:
        raise ValueError(f"No graph input found in {model_path}")

    dims: List[int] = []
    for d in model.graph.input[0].type.tensor_type.shape.dim:
        if d.HasField("dim_value"):
            dims.append(int(d.dim_value))
        else:
            raise ValueError(
                f"Dynamic/unknown ONNX input dimension is not supported yet in {model_path}"
            )
    return dims


def eliminate_identity_nodes(src_path: str, dst_path: str) -> None:
    model = onnx.load(src_path)
    model_opt = onnxoptimizer.optimize(model, ["eliminate_identity"])
    onnx.checker.check_model(model_opt)
    onnx.save(model_opt, dst_path)


def _collect_tensor_names(model: onnx.ModelProto) -> List[str]:
    names: List[str] = []

    for vi in model.graph.input:
        if vi.name:
            names.append(vi.name)

    for vi in model.graph.output:
        if vi.name:
            names.append(vi.name)

    for vi in model.graph.value_info:
        if vi.name:
            names.append(vi.name)

    for init in model.graph.initializer:
        if init.name:
            names.append(init.name)

    for node in model.graph.node:
        for x in node.input:
            if x:
                names.append(x)
        for x in node.output:
            if x:
                names.append(x)

    seen = set()
    ordered: List[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)

    return ordered


def rename_all_tensors_to_numeric(src_path: str, dst_path: str) -> Dict[str, str]:
    model = onnx.load(src_path)

    tensor_names = _collect_tensor_names(model)
    name_map = {old: str(i) for i, old in enumerate(tensor_names)}
    dump_requant_constants(str(src_path))

    for vi in model.graph.input:
        if vi.name in name_map:
            vi.name = name_map[vi.name]

    for vi in model.graph.output:
        if vi.name in name_map:
            vi.name = name_map[vi.name]

    for vi in model.graph.value_info:
        if vi.name in name_map:
            vi.name = name_map[vi.name]

    for init in model.graph.initializer:
        if init.name in name_map:
            init.name = name_map[init.name]

    for node in model.graph.node:
        for i, x in enumerate(node.input):
            if x in name_map:
                node.input[i] = name_map[x]
        for i, x in enumerate(node.output):
            if x in name_map:
                node.output[i] = name_map[x]

    onnx.checker.check_model(model)
    onnx.save(model, dst_path)
    return name_map

def _initializer_map(model: onnx.ModelProto) -> Dict[str, onnx.TensorProto]:
    return {init.name: init for init in model.graph.initializer}


def _constant_output_map(model: onnx.ModelProto) -> Dict[str, onnx.NodeProto]:
    out = {}
    for node in model.graph.node:
        if node.op_type == "Constant" and len(node.output) == 1:
            out[node.output[0]] = node
    return out


def _consumer_map(model: onnx.ModelProto) -> Dict[str, List[onnx.NodeProto]]:
    consumers: Dict[str, List[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)
    return consumers


def _producer_map(model: onnx.ModelProto) -> Dict[str, onnx.NodeProto]:
    producers: Dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for out in node.output:
            producers[out] = node
    return producers


def _const_scalar_from_name(
    name: str,
    init_map: Dict[str, onnx.TensorProto],
    const_out_map: Dict[str, onnx.NodeProto],
) -> Optional[float]:
    if name in init_map:
        arr = numpy_helper.to_array(init_map[name]).reshape(-1)
        if arr.size == 1:
            return float(arr[0])

    if name in const_out_map:
        node = const_out_map[name]
        for attr in node.attribute:
            if attr.name == "value":
                arr = numpy_helper.to_array(attr.t).reshape(-1)
                if arr.size == 1:
                    return float(arr[0])

    return None


def normalize_quantlab_onnx_for_dory(src_path: str, dst_path: str, atol: float = 1e-9) -> int:
    """
    Rewrite QuantLib exported hidden-layer requant blocks from:

        Mul -> Add -> Div -> Add(+0.5) -> Floor -> Clip

    to:

        Mul -> Add -> Div -> Floor -> Clip

    so they match the DORY Quantlab frontend pattern:
        BNRelu = Mul-Add-Div-Floor-Clip

    Returns the number of rewritten requant blocks.
    """
    model = onnx.load(src_path)
    graph = model.graph

    init_map = _initializer_map(model)
    const_out_map = _constant_output_map(model)
    consumers = _consumer_map(model)
    producers = _producer_map(model)

    nodes = list(graph.node)
    nodes_to_remove = set()
    rewrites = 0

    def one_consumer(tensor_name: str) -> Optional[onnx.NodeProto]:
        nxt = consumers.get(tensor_name, [])
        return nxt[0] if len(nxt) == 1 else None

    for node in nodes:
        if node.op_type != "Div" or len(node.output) != 1:
            continue

        div_out = node.output[0]
        add1 = one_consumer(div_out)
        if add1 is None or add1.op_type != "Add" or len(add1.output) != 1:
            continue

        # Identify the constant side of Add(+0.5)
        if add1.input[0] == div_out:
            other = add1.input[1]
        elif add1.input[1] == div_out:
            other = add1.input[0]
        else:
            continue

        add1_const = _const_scalar_from_name(other, init_map, const_out_map)
        if add1_const is None or abs(add1_const - 0.5) > atol:
            continue

        floor = one_consumer(add1.output[0])
        if floor is None or floor.op_type != "Floor" or len(floor.input) != 1 or len(floor.output) != 1:
            continue

        if floor.input[0] != add1.output[0]:
            continue

        # Rewire Floor input directly to Div output.
        floor.input[0] = div_out

        nodes_to_remove.add(add1.name or id(add1))

        # If the +0.5 constant is produced by a Constant node and unused elsewhere, remove it too.
        const_prod = producers.get(other)
        if const_prod is not None and const_prod.op_type == "Constant":
            other_consumers = consumers.get(other, [])
            if len(other_consumers) == 1 and other_consumers[0] is add1:
                nodes_to_remove.add(const_prod.name or id(const_prod))

        rewrites += 1

    new_nodes = []
    for node in graph.node:
        key = node.name or id(node)
        if key not in nodes_to_remove:
            new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)

    onnx.checker.check_model(model)
    onnx.save(model, dst_path)
    return rewrites