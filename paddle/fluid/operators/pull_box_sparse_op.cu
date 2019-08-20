//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/pull_box_sparse_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;

using LoDTensor = framework::LoDTensor;

template <typename T>
class PullBoxSparseCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PullBoxSparseFunctor(ctx);
  }
};

template <typename T>
class PullBoxSparseCUDAGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PushBoxSparseFunctor(ctx);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(pull_box_sparse, ops::PullBoxSparseCUDAKernel<int>,
                        ops::PullBoxSparseCUDAKernel<int64_t>);
REGISTER_OP_CUDA_KERNEL(pull_box_sparse_grad,
                        ops::PullBoxSparseCUDAGradKernel<int>,
                        ops::PullBoxSparseCUDAGradKernel<int64_t>);
