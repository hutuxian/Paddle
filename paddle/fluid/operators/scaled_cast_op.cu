/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/scaled_cast_op.h"
#include "paddle/fluid/platform/float16.h"

template <typename T>
using ScaledCastOpKernel =
    paddle::operators::ScaledCastOpKernel<paddle::platform::CUDADeviceContext, T>;

namespace ops = paddle::operators;
using GPUCtx = paddle::platform::CUDADeviceContext;

REGISTER_OP_CUDA_KERNEL(scaled_cast, 
                        ScaledCastOpKernel<float>, 
                        ScaledCastOpKernel<double>,
                        //ScaledCastOpKernel<int>, 
                        //ScaledCastOpKernel<int64_t>,
                        //ScaledCastOpKernel<bool>, 
                        //ScaledCastOpKernel<uint8_t>,
                        ScaledCastOpKernel<paddle::platform::float16>);

REGISTER_OP_CUDA_KERNEL(scaled_cast_back,
                        ops::ScaledCastBackOpKernel<GPUCtx, float>,
                        ops::ScaledCastBackOpKernel<GPUCtx, double>,
                        //ops::ScaledCastBackOpKernel<GPUCtx, int>,
                        //ops::ScaledCastBackOpKernel<GPUCtx, int64_t>,
                        //ops::ScaledCastBackOpKernel<GPUCtx, bool>,
                        //ops::ScaledCastBackOpKernel<GPUCtx, uint8_t>,
                        ops::ScaledCastBackOpKernel<GPUCtx, paddle::platform::float16>);

