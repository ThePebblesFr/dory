/*
 * network.h
 * Alessio Burrello <alessio.burrello@unibo.it>
 *
 * Copyright (C) 2019-2020 University of Bologna
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 */

#ifndef __NETWORK_H__
#define __NETWORK_H__

#include <stddef.h>
#include "pmsis.h"


struct network_run_token {
  struct pi_device cluster_dev;
};


void network_terminate();
void network_initialize();
void network_run_cluster(void * args);
struct network_run_token network_run_async(void *l2_buffer, size_t l2_buffer_size, void *l2_final_output, int exec, int initial_dir);
void network_run_wait(struct network_run_token token);
void network_run(void *l2_buffer, size_t l2_buffer_size, void *l2_final_output, int exec, int initial_dir);
void execute_layer_fork(void *arg);


#ifdef DEFINE_CONSTANTS
// allocation of buffers with parameters needed by the network execution
static const char * L3_weights_files[] = {
  "BNReluConvolution0_weights.hex", "BNReluConvolution2_weights.hex", "BNReluConvolution3_weights.hex", "BNReluConvolution4_weights.hex", "BNReluConvolution5_weights.hex", "BNReluConvolution6_weights.hex", "BNReluConvolution7_weights.hex", "FullyConnected8_weights.hex"
};
static int L3_weights_size[8];
static int layers_pointers[9];
static char * Layers_name[9] = {"BNReluConvolution0", "Pooling1", "BNReluConvolution2", "BNReluConvolution3", "BNReluConvolution4", "BNReluConvolution5", "BNReluConvolution6", "BNReluConvolution7", "FullyConnected8"};
static int L3_input_layers[9] = {1,
0, 0, 0, 0, 0, 0, 0, 0};
static int L3_output_layers[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
static int allocate_layer[9] = {1, 0, 1, 1, 1, 1, 1, 1, 1};
static int branch_input[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
static int branch_output[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
static int branch_change[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
static int weights_checksum[9] = {128644, 0, 1215436, 1221716, 2498892, 4952931, 8910623, 19745627, 615013};
static int weights_size[9] = {1056, 0, 9472, 9472, 18944, 37376, 74752, 148480, 7696};
static int activations_checksum[9][1] = {{
  2818623  },
{
  121590  },
{
  45486  },
{
  14230  },
{
  15057  },
{
  10668  },
{
  15843  },
{
  3084  },
{
  17906  }
};
static int activations_size[9] = {15360, 122880, 30720, 7680, 7680, 3840, 3840, 1920, 1920};
static int out_mult_vector[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
static int out_shift_vector[9] = {24, 0, 23, 23, 23, 23, 24, 21, 0};
static int activations_out_checksum[9][1] = {{
  121590 },
{
  45486 },
{
  14230 },
{
  15057 },
{
  10668 },
{
  15843 },
{
  3084 },
{
  17906 },
{
  1169 }
};
static int activations_out_size[9] = {122880, 30720, 7680, 7680, 3840, 3840, 1920, 1920, 16};
static int layer_with_weights[9] = {1, 0, 1, 1, 1, 1, 1, 1, 1};
#endif

#endif  // __NETWORK_H__
