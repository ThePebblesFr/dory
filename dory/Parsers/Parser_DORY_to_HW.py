     # should work even without -*-
# -*- coding: utf-8 -*-
#!/bin/bash
# ONNX_to_DORY_generic.py
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

# Libraries
import numpy as np
import sys
import copy

# DORY modules
from .HW_node import HW_node
from dory.Utils.DORY_utils import Printer


class Parser_DORY_to_HW:
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.
    def __init__(self, graph, rules, Pattern_rewriter, supported_nodes, HW_description, network_directory, config_file, Tiler, n_inputs=1):
        self.supported_nodes = supported_nodes
        self.DORY_Graph = graph
        self.Printer_Frontend = Printer("logs/HW_related")
        self.Pattern_rewriter = Pattern_rewriter
        self.rules = rules
        self.HW_description = HW_description
        self.network_directory = network_directory
        self.config_file = config_file
        self.n_inputs = n_inputs
        HW_node.Tiler = Tiler

    def mapping_to_HW_nodes(self):
        print("\nBackend: Matching patterns from generated DORY ONNX to HW Nodes.")
        for i, node in enumerate(self.DORY_Graph):
            string_matching, indexes = self.pattern_matching(node, i)
            if isinstance(string_matching, str):
                self.DORY_Graph = self.Pattern_rewriter(self.DORY_Graph).execute(string_matching, indexes)

    def check_graph(self):
        for node in self.DORY_Graph:
            if node.name not in self.supported_nodes:
                sys.exit("\nDORY Backend Check. Node {} is not accepted inside the HW Frontend IR.\n".format(node.name))
        print("\nDORY checking of the graph: OK\n")

    def check_parameters(self):
        print("\nTo be implemented in the target backend")

    def pattern_matching(self, input_node, input_index):
        number_of_nodes = 0
        rule_found = False
        DORY_node_indexes_to_export = []
        for key, rule in self.rules.items():
            DORY_node_indexes = []
            DORY_node_indexes.append(input_index)
            if rule["number_of_nodes"] == 1 and input_node.name in rule["nodes_name"]:
                rule_found = key
            elif input_node.name in rule["nodes_name"]:
                node = input_node
                match = 1
                nodes = copy.deepcopy(rule["nodes_name"])
                index = nodes.index(node.name)
                nodes[index] = "Match"
                while match == 1:
                    match = 0
                    inputs = rule["dependencies"][str(index)]["inputs"]
                    outputs = rule["dependencies"][str(index)]["outputs"]
                    for nodes_index in inputs:
                        int_index = node.input_indexes
                        node_to_search = nodes[int(nodes_index)]
                        for i,node_i in enumerate(self.DORY_Graph):
                            if node_i.output_index in int_index and node_i.name == node_to_search:
                                nodes[int(nodes_index)] = "Match"
                                match = 1
                                DORY_node_indexes.append(i)
                    for nodes_index in outputs:
                        out_index = node.output_index
                        node_to_search = nodes[int(nodes_index)]
                        for i,node_i in enumerate(self.DORY_Graph):
                            if out_index in node_i.input_indexes and node_i.name == node_to_search:
                                nodes[int(nodes_index)] = "Match"
                                match = 1
                                DORY_node_indexes.append(i)
                                node = node_i
                                index = int(nodes_index)
                if sum(x=="Match" for x in nodes) == len(nodes):
                    if number_of_nodes < rule["number_of_nodes"]:
                        rule_found = key
                        number_of_nodes = rule["number_of_nodes"]
                        DORY_node_indexes_to_export = DORY_node_indexes
        return rule_found, DORY_node_indexes_to_export

    def update_branches_graph(self):
        print("\nDORY generic Frontend. Updating branches pointers.")
        # updating branch in/out connections
        for i, node in enumerate(self.DORY_Graph):
            if len(node.input_indexes)>1:
                node.add_existing_parameter("branch_in", 1)
            else:
                node.add_existing_parameter("branch_in", 0)
            node_out = 0
            for nodes_scan in self.DORY_Graph:
                if node.output_index in nodes_scan.input_indexes:
                    node_out+=1
            if node_out > 1:
                node.add_existing_parameter("branch_out", 1)
            else:
                node.add_existing_parameter("branch_out", 0)
            node.add_existing_parameter("branch_change", 0)
            node.add_existing_parameter("branch_last", 0)
            for nodes_scan in self.DORY_Graph:
                if node.output_index in nodes_scan.input_indexes and len(nodes_scan.input_indexes)>1:
                    for j, nodes_scan_2 in enumerate(self.DORY_Graph):
                        if nodes_scan_2.output_index in nodes_scan.input_indexes and nodes_scan_2.output_index != node.output_index:
                            if nodes_scan_2.branch_out != 1 and node.branch_out != 1:
                                if(i < j):
                                    node.add_existing_parameter("branch_change", 1)
                                    nodes_scan_2.add_existing_parameter("branch_last", 0)
                                    break
                                else:
                                    nodes_scan_2.add_existing_parameter("branch_change", 1)  
                                    node.add_existing_parameter("branch_last", 1)   
                                    break
                            else:
                                if(i < j):
                                    node.add_existing_parameter("branch_last", 1)
                                else:
                                    nodes_scan_2.add_existing_parameter("branch_last", 1)  

    # def update_dimensions_graph(self):
    #     print("\nUpdating dimensions of vectors inside the graph, if they do not match among nodes")
    #     for i, node in enumerate(self.DORY_Graph):
    #         if i > 0:
    #             if isinstance(self.DORY_Graph[i].input_channels, type(None)):
    #                 if "FullyConnected" in self.DORY_Graph[i].name:
    #                     self.DORY_Graph[i].input_channels = int(self.DORY_Graph[i-1].output_channels*np.prod(self.DORY_Graph[i-1].output_dimensions))
    #                 else:
    #                     self.DORY_Graph[i].input_channels = self.DORY_Graph[i-1].output_channels
    #             if len(self.DORY_Graph[i].input_dimensions)==0:
    #                 self.DORY_Graph[i].input_dimensions = self.DORY_Graph[i-1].output_dimensions
    def update_dimensions_graph(self):
        print("\nUpdating dimensions of vectors inside the graph, if they do not match among nodes")

        for i, node in enumerate(self.DORY_Graph):
            prev_node = self.DORY_Graph[i - 1] if i > 0 else None
            next_node = self.DORY_Graph[i + 1] if i < len(self.DORY_Graph) - 1 else None

            input_channels = getattr(node, "input_channels", None)
            output_channels = getattr(node, "output_channels", None)
            input_dimensions = getattr(node, "input_dimensions", None)
            output_dimensions = getattr(node, "output_dimensions", None)

            # infer missing input channels from previous node
            if input_channels is None and prev_node is not None:
                prev_out_ch = getattr(prev_node, "output_channels", None)
                if prev_out_ch is not None:
                    node.input_channels = prev_out_ch

            # infer missing output channels from next node
            if output_channels is None:
                if next_node is not None:
                    next_in_ch = getattr(next_node, "input_channels", None)
                    if next_in_ch is not None:
                        node.output_channels = next_in_ch
                elif getattr(node, "input_channels", None) is not None:
                    node.output_channels = node.input_channels

            # infer missing input dimensions from previous node
            if input_dimensions is None and prev_node is not None:
                prev_out_dim = getattr(prev_node, "output_dimensions", None)
                if prev_out_dim is not None:
                    node.input_dimensions = prev_out_dim

            # infer missing output dimensions from next node or fallback
            if output_dimensions is None:
                if next_node is not None:
                    next_in_dim = getattr(next_node, "input_dimensions", None)
                    if next_in_dim is not None:
                        node.output_dimensions = next_in_dim
                elif getattr(node, "input_dimensions", None) is not None:
                    node.output_dimensions = node.input_dimensions

    def add_tensors_memory_occupation_and_MACs(self):
        print("\nUpdating memory occupation and MACs of tensors in layers")
        for i, node in enumerate(self.DORY_Graph):
            if "Convolution" in node.name or "FullyConnected" in node.name or "Add" in node.op_type or "Pooling" in node.name:
                node.add_memory_and_MACs()

    def adjust_data_layout(self):
        print("\nTo be implemented in the target backend")

    # Override if you want to instanciate a different type of HW_node
    def transform_nodes_to_hw_nodes(self):
        self.DORY_Graph = [HW_node(node, self.HW_description) for node in self.DORY_Graph]

    def tiling(self):
        ####################################################################################
        ###### SECTION 3: PARSING OF EACH LAYER INDEPENDENT. TILING + LAYER CREATION  ######
        ####################################################################################
        print("\nInsert tiling parameters per layer inside graph nodes")
        prev = None
        for node in self.DORY_Graph:
            ######################## NEED A  FIX ####################################################
            #### OTHERWISE ONLY WEIGHT < L2/2 GO in L2 --> much more L3 tiling not needed############
            #########################################################################################
            node.create_tiling_dimensions(prev if prev else node, self.config_file)
            prev = node

    def renaming_weights(self):
        print("\nDORY Backend: Renaming Weights tensors.")
        for i, node in enumerate(self.DORY_Graph):            
            node.rename_weights()           

    def formatting_constant_parameters_tensors_and_activations(self):
        print("\nDORY Backend: Formatting constants and adding checksums")
        for i, node in enumerate(self.DORY_Graph):            
            node.add_checksum_w_integer()           
            node.add_checksum_activations_integer(self.network_directory, i, self.n_inputs)


    # def fuse_requant_into_fullyconnected(self):
    #     fused_graph = []
    #     i = 0

    #     while i < len(self.DORY_Graph):
    #         node = self.DORY_Graph[i]

    #         if (
    #             i + 1 < len(self.DORY_Graph)
    #             and getattr(node, "name", None) == "FullyConnected"
    #             and getattr(self.DORY_Graph[i + 1], "name", None) == "Requant"
    #         ):
    #             fc = node
    #             rq = self.DORY_Graph[i + 1]

    #             # Fold requant parameters into the FC node
    #             for attr in [
    #                 "outmul",
    #                 "outadd",
    #                 "outshift",
    #                 "min",
    #                 "max",
    #                 "output_activation_bits",
    #                 "output_activation_type",
    #                 "constants_memory",
    #                 "output_activation_memory",
    #                 "check_sum_out",
    #             ]:
    #                 if hasattr(rq, attr):
    #                     setattr(fc, attr, getattr(rq, attr))

    #             # FC now produces the requantized output tensor
    #             if hasattr(rq, "output_index"):
    #                 fc.output_index = rq.output_index

    #             # Merge constant names
    #             fc_constant_names = list(getattr(fc, "constant_names", []))
    #             rq_constant_names = list(getattr(rq, "constant_names", []))
    #             for c in rq_constant_names:
    #                 if c not in fc_constant_names:
    #                     fc_constant_names.append(c)
    #             fc.constant_names = fc_constant_names

    #             # Move actual constant tensors over
    #             for key, value in rq.__dict__.items():
    #                 if isinstance(value, dict) and "value" in value:
    #                     setattr(fc, key, value)

    #             # After fusion, FC output is no longer 32-bit accumulator output
    #             # if requant specified a different output precision/type.
    #             if hasattr(rq, "output_activation_bits"):
    #                 fc.output_activation_bits = rq.output_activation_bits
    #             if hasattr(rq, "output_activation_type"):
    #                 fc.output_activation_type = rq.output_activation_type

    #             fused_graph.append(fc)
    #             i += 2
    #             continue

    #         fused_graph.append(node)
    #         i += 1

    #     self.DORY_Graph = fused_graph

    def fuse_requant_into_fullyconnected(self):

        def identify_fc_weight_and_bias(fc):
            weight_key = None
            bias_key = None

            expected_weight_size = int(fc.output_channels) * int(fc.input_channels)
            expected_bias_size = int(fc.output_channels)

            candidates = []

            for cname in getattr(fc, "constant_names", []):
                if not hasattr(fc, cname):
                    continue

                entry = getattr(fc, cname)
                if not isinstance(entry, dict) or "value" not in entry:
                    continue

                arr = np.asarray(entry["value"])
                layout = entry.get("layout", None)

                candidates.append((cname, arr, layout))

            # 1. Weight must match Cout * Cin exactly
            for cname, arr, layout in candidates:
                if arr.size == expected_weight_size:
                    weight_key = cname
                    break

            # 2. Bias must match Cout exactly, excluding the chosen weight
            for cname, arr, layout in candidates:
                if cname == weight_key:
                    continue
                if arr.size == expected_bias_size:
                    bias_key = cname
                    break

            # 3. Fallbacks only if still missing
            if weight_key is None:
                for cname, arr, layout in candidates:
                    if layout in ("CoutCin", "CinCout") and arr.ndim >= 2:
                        weight_key = cname
                        break

            if bias_key is None:
                for cname, arr, layout in candidates:
                    if cname == weight_key:
                        continue
                    if arr.ndim == 1 and arr.size == expected_bias_size:
                        bias_key = cname
                        break

            return weight_key, bias_key

        fused_graph = []
        i = 0

        while i < len(self.DORY_Graph):
            node = self.DORY_Graph[i]

            if (
                i + 1 < len(self.DORY_Graph)
                and getattr(node, "name", None) == "FullyConnected"
                and getattr(self.DORY_Graph[i + 1], "name", None) == "Requant"
            ):
                fc = node
                rq = self.DORY_Graph[i + 1]

                # Identify original FC weight and bias tensors
                weight_key, bias_key = identify_fc_weight_and_bias(fc)

                if weight_key is not None:
                    fc.weights = getattr(fc, weight_key)
                if bias_key is not None:
                    fc.bias = getattr(fc, bias_key)

                # Copy requant tensors explicitly
                for key in ["outmul", "outadd", "outshift"]:
                    if hasattr(rq, key):
                        setattr(fc, key, getattr(rq, key))

                # Fold requant scalar attributes into FC
                for attr in [
                    "min",
                    "max",
                    "output_activation_bits",
                    "output_activation_type",
                    "constants_memory",
                    "output_activation_memory",
                    "check_sum_out",
                ]:
                    if hasattr(rq, attr):
                        setattr(fc, attr, getattr(rq, attr))

                # FC now produces the requantized tensor
                if hasattr(rq, "output_index"):
                    fc.output_index = rq.output_index

                # Build a clean constant order for downstream packing
                new_constant_names = []
                if hasattr(fc, "weights"):
                    new_constant_names.append("weights")
                if hasattr(fc, "bias"):
                    new_constant_names.append("bias")
                for key in ["outshift", "outmul", "outadd"]:
                    if hasattr(fc, key):
                        new_constant_names.append(key)

                fc.constant_names = new_constant_names
                fc.number_of_input_constants = len(new_constant_names)

                fused_graph.append(fc)
                i += 2
                continue

            fused_graph.append(node)
            i += 1

        self.DORY_Graph = fused_graph

    def normalize_fullyconnected_constants(self):
       
        for node in self.DORY_Graph:
            if getattr(node, "name", None) != "FullyConnected":
                continue

            expected_weight_vals = int(node.output_channels) * int(node.input_channels)
            expected_bias_vals = int(node.output_channels)
            expected_bias_bytes = expected_bias_vals * 4

            weight_key = None
            bias_key = None

            for cname in getattr(node, "constant_names", []):
                if not hasattr(node, cname):
                    continue
                entry = getattr(node, cname)
                if not isinstance(entry, dict) or "value" not in entry:
                    continue

                arr = np.asarray(entry["value"])

                # Weight values
                if arr.size == expected_weight_vals and weight_key is None:
                    weight_key = cname
                    continue

                # Bias already packed as bytes
                if arr.size == expected_bias_bytes and bias_key is None:
                    bias_key = cname
                    continue

                # Bias as one value per output channel
                if arr.size == expected_bias_vals and bias_key is None:
                    bias_key = cname
                    continue

            if weight_key is not None:
                node.weights = getattr(node, weight_key)
            if bias_key is not None:
                node.bias = getattr(node, bias_key)

            new_constant_names = []
            if hasattr(node, "weights"):
                new_constant_names.append("weights")
            if hasattr(node, "bias"):
                new_constant_names.append("bias")
            for key in ["outshift", "outmul", "outadd"]:
                if hasattr(node, key):
                    new_constant_names.append(key)

            node.constant_names = new_constant_names
            node.number_of_input_constants = len(new_constant_names)

    def full_graph_parsing(self):
        print("#####################################################")
        print("## DORY GENERAL PARSING FROM DORY IR TO DORY HW IR ##")
        print("## FINAL RAPRESENTATION: DORY HW IR                ##")
        print("#####################################################")
        self.Printer_Frontend.print_json_from_DORY_graph("00_DORY_HW_input_graph", self.DORY_Graph)
        self.Printer_Frontend.print_onnx_from_DORY_graph("00_DORY_HW_input_graph", self.DORY_Graph)
        self.mapping_to_HW_nodes()
        self.Printer_Frontend.print_json_from_DORY_graph("01_DORY_HW_graph_raw", self.DORY_Graph)
        self.Printer_Frontend.print_onnx_from_DORY_graph("01_DORY_HW_graph_raw", self.DORY_Graph)
        self.update_branches_graph()
        self.fuse_requant_into_fullyconnected()
        self.normalize_fullyconnected_constants()
        self.Printer_Frontend.print_json_from_DORY_graph("02_DORY_HW_graph_fixed_branches", self.DORY_Graph)
        self.Printer_Frontend.print_onnx_from_DORY_graph("02_DORY_HW_graph_fixed_branches", self.DORY_Graph)
        self.update_dimensions_graph()
        self.Printer_Frontend.print_json_from_DORY_graph("03_DORY_HW_graph_fixed_dimensions", self.DORY_Graph)
        self.Printer_Frontend.print_onnx_from_DORY_graph("03_DORY_HW_graph_fixed_dimensions", self.DORY_Graph)
        self.adjust_data_layout()
        self.Printer_Frontend.print_json_from_DORY_graph("04_DORY_HW_adjusted_data_layout", self.DORY_Graph)
        self.Printer_Frontend.print_onnx_from_DORY_graph("04_DORY_HW_adjusted_data_layout", self.DORY_Graph)
        self.add_tensors_memory_occupation_and_MACs()
        self.Printer_Frontend.print_json_from_DORY_graph("05_DORY_HW_graph_added_tensors_dim", self.DORY_Graph)
        self.Printer_Frontend.print_onnx_from_DORY_graph("05_DORY_HW_graph_added_tensors_dim", self.DORY_Graph)
        self.transform_nodes_to_hw_nodes()
        self.tiling()
        self.Printer_Frontend.print_json_from_DORY_graph("06_DORY_HW_tiled_graph", self.DORY_Graph)
        self.Printer_Frontend.print_onnx_from_DORY_graph("06_DORY_HW_tiled_graph", self.DORY_Graph)
        self.renaming_weights()
        self.formatting_constant_parameters_tensors_and_activations()
        self.Printer_Frontend.print_json_from_DORY_graph("07_DORY_HW_with_checksums", self.DORY_Graph)
        self.Printer_Frontend.print_onnx_from_DORY_graph("07_DORY_HW_with_checksums", self.DORY_Graph)
        self.check_graph()
        self.check_parameters()
        return self.DORY_Graph

