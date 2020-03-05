# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
from op_test import OpTest, skip_check_grad_ci
import paddle.fluid.core as core


def gen_input_help(input, rank_offset, max_rank):
    input_row, input_col = input.shape
    input_help = np.zeros((input_row, max_rank * input_col))
    ins_rank = np.zeros((input_row, 1))
    for ins in range(input_row):
        temp = []
        ins_rank[ins] = rank_offset[ins, 0]
        for k in range(max_rank):
            rank_msg = rank_offset[ins, k * 2 + 2]
            if rank_msg >= 0:
                temp.extend(input[int(rank_msg), :])
            else:
                temp.extend(list(np.zeros(input_col)))
        input_help[ins, :] = temp
    return input_help, ins_rank


def gen_param_help(input, rank_offset, rank_para, max_rank):
    input_row, input_col = input.shape
    _, rank_para_col = rank_para.shape
    block_matrix_row = input_col * max_rank
    param_help = np.zeros((block_matrix_row * input_row, rank_para_col))

    for ins in range(input_row):
        start_index = ins * block_matrix_row
        block_temp = np.zeros((block_matrix_row, rank_para_col))
        for k in range(max_rank):
            if rank_offset[ins, k * 2 + 2] >= 0:
                start = rank_offset[ins, 0] * max_rank + k
                block_temp[int(k * input_col):int((k + 1) * input_col),:] = \
                    rank_para[int(start * input_col): int(start * input_col + input_col), :]
        param_help[start_index:start_index + block_matrix_row] = block_temp
    return param_help


def np_rank_attention(input, rank_offset, rank_para, max_rank):
    input_row, input_col = input.shape
    rank_offset_row, rank_offset_col = rank_offset.shape
    rank_para_row, rank_para_col = rank_para.shape

    assert (input_row == rank_offset_row)
    assert (max_rank == ((rank_offset_col - 1) / 2))
    assert (rank_para_row == max_rank * max_rank * input_col)

    input_help, ins_rank = gen_input_help(input, rank_offset, max_rank)
    param_help = gen_param_help(input, rank_offset, rank_para, max_rank)
    block_matrix_row = input_col * max_rank

    res = np.zeros((input_row, rank_para_col))
    for ins in range(input_row):
        res[ins, :] = \
            np.dot(input_help[ins, :],
                   param_help[int(block_matrix_row * ins):int(block_matrix_row * (ins+1)),:])
    return res, input_help, param_help, ins_rank


def gen_rank_offset(pv_nums, max_rank):
    rank_offset = []
    all_ins_num = 0
    for _ in range(pv_nums):
        ins_pv = np.random.randint(1, max_rank + 1)
        for ins in range(ins_pv):
            temp = []
            fir_col = all_ins_num
            se_col = fir_col + 1
            th_col = se_col + 1
            if ins_pv == 1:
                se_col = -1
                th_col = -1
            elif ins_pv == 2:
                th_col = -1
            temp = [ins, 0, fir_col, 1, se_col, 2, th_col]
            rank_offset.append(temp)
        all_ins_num += ins_pv
    return all_ins_num, rank_offset


class TestRankAttentionOpComplex(OpTest):
    def config(self):
        self.pv_num = 100
        self.x_feat = 10
        self.y_feat = 15
        self.max_rank = 3
        self.dtype = "float64"

    def setUp(self):
        self.op_type = "rank_attention"
        self.config()
        ins_num, rank_offset = gen_rank_offset(self.pv_num, self.max_rank)
        input = np.random.random((ins_num, self.x_feat)).astype(self.dtype)
        rank_para_shape = [
            self.max_rank * self.max_rank * self.x_feat, self.y_feat
        ]
        rank_para = np.random.random(rank_para_shape).astype(self.dtype)
        np_out, np_input_help, np_param_help, np_ins_rank = np_rank_attention(
            input, np.array(rank_offset), rank_para, self.max_rank)
        self.inputs = {
            "X": input,
            "RankOffset": np.array(rank_offset).astype(self.dtype),
            "RankParam": rank_para
        }
        self.attrs = {'MaxRank': self.max_rank}
        self.outputs = {
            "Out": np_out,
            "InputHelp": np_input_help,
            "ParamHelp": np_param_help,
            "InsRank": np_ins_rank
        }

    def test_check_output(self):
        #self.check_output()
        self.check_output_with_place(core.CUDAPlace(0), atol=0)

    def test_check_output(self):
        #self.check_output()
        self.check_grad_with_place(core.CUDAPlace(0), ["X", "RankParam"], "Out")


if __name__ == "__main__":
    unittest.main()
