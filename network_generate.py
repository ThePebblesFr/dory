#!/usr/bin/env python
# network_generate.py
# Alessio Burrello <alessio.burrello@unibo.it>
#
# Copyright (C) 2019-2020 University of Bologna
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#####################CONFIG PARAMETERS #########################
# BNRelu_bits. Number of bits for lambda and k parameters in BNRelu. 32 or 64
# onnx file.

# Libraries
import argparse
import os.path
from argparse import RawTextHelpFormatter
import json
from importlib import import_module

import os
import numpy as np

def clip8(x: int) -> int:
    return max(0, min(255, int(x)))

def pulp_nn_quant_u8(phi: int, m: int, d: int) -> int:
    return clip8((m * phi) >> d)

def decode_i8_weights(entry, out_ch, in_ch):
    arr = np.asarray(entry["value"])
    if arr.dtype == np.uint8:
        arr = arr.view(np.int8)
    else:
        arr = arr.astype(np.int8, copy=False)
    return arr.reshape(out_ch, in_ch)

def decode_i32_bias(entry, out_ch):
    arr = np.asarray(entry["value"])
    if arr.dtype == np.uint8 and arr.size == out_ch * 4:
        return arr.view(np.int32)
    if arr.dtype == np.int32 and arr.size == out_ch:
        return arr
    return arr.astype(np.int32, copy=False).reshape(out_ch)

def get_scalar(entry):
    arr = np.asarray(entry["value"]).reshape(-1)
    return int(arr[0])

def fc_u8_u8_i8(x_u8, w_i8, b_i32, out_mult, out_shift):
    x_u8 = np.asarray(x_u8, dtype=np.uint8).reshape(-1)
    y = np.zeros(w_i8.shape[0], dtype=np.uint8)
    for i in range(w_i8.shape[0]):
        phi = int(b_i32[i]) + int(np.dot(w_i8[i].astype(np.int64), x_u8.astype(np.int64)))
        y[i] = pulp_nn_quant_u8(phi, out_mult, out_shift)
    return y

def fc_u8_i32_i8(x_u8, w_i8, b_i32):
    x_u8 = np.asarray(x_u8, dtype=np.uint8).reshape(-1)
    y = np.zeros(w_i8.shape[0], dtype=np.int32)
    for i in range(w_i8.shape[0]):
        phi = int(b_i32[i]) + int(np.dot(w_i8[i].astype(np.int64), x_u8.astype(np.int64)))
        y[i] = np.int32(phi)
    return y

def export_reference_txt_from_dory_graph(dory_graph, input_txt_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    x0 = np.loadtxt(input_txt_path, delimiter=",", dtype=np.int64, usecols=[0]).astype(np.uint8)
    x0 = x0.reshape(-1)

    fc_nodes = [n for n in dory_graph if getattr(n, "name", None) == "FullyConnected"]
    assert len(fc_nodes) == 4, f"Expected 4 FullyConnected nodes, got {len(fc_nodes)}"

    fc0, fc1, fc2, fc3 = fc_nodes

    W0 = decode_i8_weights(fc0.weights, 128, 16)
    B0 = decode_i32_bias(fc0.bias, 128)
    M0 = get_scalar(fc0.outmul)
    S0 = get_scalar(fc0.outshift)

    W1 = decode_i8_weights(fc1.weights, 64, 128)
    B1 = decode_i32_bias(fc1.bias, 64)
    M1 = get_scalar(fc1.outmul)
    S1 = get_scalar(fc1.outshift)

    W2 = decode_i8_weights(fc2.weights, 32, 64)
    B2 = decode_i32_bias(fc2.bias, 32)
    M2 = get_scalar(fc2.outmul)
    S2 = get_scalar(fc2.outshift)

    W3 = decode_i8_weights(fc3.weights, 10, 32)
    B3 = decode_i32_bias(fc3.bias, 10)

    y0 = fc_u8_u8_i8(x0, W0, B0, M0, S0)
    y1 = fc_u8_u8_i8(y0, W1, B1, M1, S1)
    y2 = fc_u8_u8_i8(y1, W2, B2, M2, S2)
    y3 = fc_u8_i32_i8(y2, W3, B3)

    np.savetxt(os.path.join(out_dir, "input.txt"), x0.reshape(-1, 1), fmt="%d", delimiter=",")
    np.savetxt(os.path.join(out_dir, "out_layer0.txt"), y0.reshape(-1, 1), fmt="%d", delimiter=",")
    np.savetxt(os.path.join(out_dir, "out_layer1.txt"), y1.reshape(-1, 1), fmt="%d", delimiter=",")
    np.savetxt(os.path.join(out_dir, "out_layer2.txt"), y2.reshape(-1, 1), fmt="%d", delimiter=",")
    np.savetxt(os.path.join(out_dir, "out_layer3.txt"), y3.reshape(-1, 1), fmt="%d", delimiter=",")

    print("Checksums from lowered DORY graph:")
    print("input     ", int(x0.sum()))
    print("out_layer0", int(y0.sum()))
    print("out_layer1", int(y1.sum()))
    print("out_layer2", int(y2.sum()))
    print("out_layer3", int(y3.view(np.uint8).sum()))

def dory_to_c(graph, target, conf, confdir, verbose_level, perf_layer, optional, appdir, n_inputs):
    # Including and running the transformation from DORY IR to DORY HW IR
    onnx_manager = import_module(f'dory.Hardware_targets.{target}.HW_Parser')
    dory_to_dory_hw = onnx_manager.onnx_manager
    graph = dory_to_dory_hw(graph, conf, confdir, n_inputs).full_graph_parsing()

    # export_reference_txt_from_dory_graph(
    #     graph,
    #     input_txt_path="./dory/dory_examples/examples/Custom/input.txt",
    #     out_dir="./dory/dory_examples/examples/Custom"
    # )

    # Deployment of the model on the target architecture
    onnx_manager = import_module(f'dory.Hardware_targets.{target}.C_Parser')
    dory_hw_to_c = onnx_manager.C_Parser
    dory_hw_to_c(graph, conf, confdir, verbose_level, perf_layer, optional, appdir, n_inputs).full_graph_parsing()


def network_generate(frontend, target, conf_file, verbose_level='Check_all+Perf_final', perf_layer='No', optional='auto',
                     appdir='./application', prefix=""):
    print(f"Using {frontend} as frontend. Targeting {target} platform. ")

    if len(prefix) > 0 and prefix[-1] != "_":
        prefix += "_"
    # Reading the json configuration file
    with open(conf_file) as f:
        conf = json.load(f)

    try:
        n_inputs = conf["n_inputs"]
    except KeyError:
        n_inputs = 1
    if n_inputs != 1:
        assert n_inputs > 1, "n_inputs must be >= 1!"

    # Reading the onnx file
    confdir = os.path.dirname(conf_file)
    onnx_file = os.path.join(confdir, conf["onnx_file"])
    print(f"Using {onnx_file} target input onnx.\n")

    # Including and running the transformation from Onnx to a DORY compatible graph
    onnx_manager = import_module(f'dory.Frontend_frameworks.{frontend}.Parser')
    onnx_to_dory = onnx_manager.onnx_manager
    graph = onnx_to_dory(onnx_file, conf, prefix).full_graph_parsing()

    dory_to_c(graph, target, conf, confdir, verbose_level, perf_layer, optional, appdir, n_inputs)


if __name__ == '__main__':
    Frontends = ["NEMO", "Quantlab"]
    Hardware_targets = ["PULP.GAP8", "PULP.GAP8_L2", "PULP.PULP_gvsoc", "PULP.GAP9", "PULP.GAP9_NE16", "Occamy", "Diana.Diana_TVM", "Diana.Diana_SoC"]
    verbose_levels = ["None", "Perf_final", "Check_all+Perf_final", "Last+Perf_final"]
    optional_choices = ["auto", "8bit", "mixed-hw", "mixed-sw"]

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('frontend', type=str, choices=Frontends, help='Frontend from which the onnx is produced and from which the network has been trained')
    parser.add_argument('hardware_target', type=str, choices=Hardware_targets, help='Hardware platform for which the code is optimized')
    parser.add_argument('config_file', type=str, help='Path to the JSON file that specifies the ONNX file of the network and other information.')
    parser.add_argument('--verbose_level', choices=verbose_levels, default='Check_all+Perf_final',
                        help="None: No_printf.\n"
                             "Perf_final: only total performance\n"
                             "Check_all+Perf_final: all check + final performances \n"
                             "Last+Perf_final: all check + final performances \n"
                             "Extract the parameters from the onnx model")
    parser.add_argument('--perf_layer', action='store_true', help='Print the performance of each layer.')
    parser.add_argument('--optional', default='auto', choices=optional_choices,
                        help='auto (based on layer precision, 8bits or mixed-sw), 8bit, mixed-hw, mixed-sw')
    parser.add_argument('--app_dir', default='./application', help='Path to the generated application. Default: ./application')
    parser.add_argument('--prefix', default="", help='Prefix to prepend to network-specific generated functions', type=str)

    args = parser.parse_args()

    network_generate(args.frontend, args.hardware_target, args.config_file, args.verbose_level, 'Yes' if args.perf_layer else 'No',
                     args.optional, args.app_dir, args.prefix)
