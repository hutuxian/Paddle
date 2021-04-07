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
import random
from op_test import OpTest
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
from op_test import OpTest, skip_check_grad_ci
import paddle.fluid.core as core
import math
import paddle

def np_cal_matrix_mul(input, w, bias, input_scale_factor, bias_scale_factor):
    ins_num, _ = input.shape
    in_feat, w_col = w.shape
    out_feat = w_col

    res = np.zeros((ins_num, w_col))
    res = np.dot(input, w) * input_scale_factor
    
    for col in range(w_col):
        res[:, col] = res[:, col] + bias[col] * bias_scale_factor
    print(res.shape)
    return res


class TestScaleFCOp(OpTest):
    def config(self):
        self.input_scale_factor = 10.0
        self.bias_scale_factor = 10.0
        self.in_feat = 10
        self.out_feat = 100
        self.ins_num = 20
        self.dtype = "float64"

    def setUp(self):
        self.config()
        self.input = np.random.random(
            (self.ins_num, self.in_feat)).astype(self.dtype)
        self.w = np.random.random(
            (self.in_feat, self.out_feat)).astype(self.dtype)
        #self.bias = np.random.random(
        #    (1, self.out_feat)).astype(self.dtype)
        self.bias = np.random.random(
            (self.out_feat, 1)).astype(self.dtype)

        """ 
        self.op_type = "mul"
        np_out = np.dot(self.input, self.w)
        self.inputs = {
            'X': self.input,
            'Y': self.w
        }
        self.outputs = {
            'Out': np_out
        }

        """
        self.op_type = "scaled_fc"
        np_out = np_cal_matrix_mul(self.input, self.w, self.bias, self.input_scale_factor, self.bias_scale_factor)
        np_out = np_out.astype(self.dtype)
       
        print("input.shape:", self.input.shape)
        print("w.shape:", self.w.shape)
        print("bias.shape:", self.bias.shape)
        print("np_out.shape:", np_out.shape)
        self.inputs = {"Input": self.input, "W": self.w, "Bias": self.bias}
        self.outputs = {"Out": np_out}
        
        self.attrs = {"input_scale_factor": self.input_scale_factor, 
            "bias_scale_factor": self.bias_scale_factor}

    def test_check_output_gpu(self):
        if core.is_compiled_with_cuda():
            print("test fp32 or fp64.")
            self.check_output_with_place(core.CUDAPlace(0))

    def test_check_grad_gpu(self):
        if core.is_compiled_with_cuda():
            print("test fp32 or fp64 grad.")
            self.check_grad_with_place(
                core.CUDAPlace(0), ["Bias", "W", "Input"], "Out")

    #def test_check_grad_gpu(self):
    #    if core.is_compiled_with_cuda():
    #        self.check_grad_with_place(
    #            core.CUDAPlace(0), ["X", "Y"], "Out")

@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFP16ScaleFCOp(TestScaleFCOp):
    def init_dtype_type(self):
        self.dtype = np.float16

    def test_check_output(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            print("test fp16.")
            #self.check_output_with_place(place, atol=1e-1)
            self.check_output_with_place(place)

    def test_check_grad_normal(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            print("test fp16 grad.")
            self.check_grad_with_place(
                place, ["Bias", "W", "Input"], 'Out', max_relative_error=0.5)

if __name__ == "__main__":
    paddle.enable_static()    
    unittest.main()
