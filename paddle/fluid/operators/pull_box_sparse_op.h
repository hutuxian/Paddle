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

#pragma once
#include <memory>
#include <vector>
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
template <typename T>
class PullBoxSparseCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    printf("paddlebox: pull box sparse int cpu ff start..\n");
    auto inputs = ctx.MultiInput<framework::Tensor>("Ids");
    auto outputs = ctx.MultiOutput<framework::Tensor>("Out");
    auto hidden_size = ctx.Attr<int>("size");
    printf("paddlebox: hidden size in op: %d\n", hidden_size);

    const auto slot_size = inputs.size();
    // std::vector<std::vector<uint64_t>> all_keys(slot_size);
    // std::vector<std::vector<float>> all_values(slot_size);
    std::vector<const uint64_t *> all_keys(slot_size);
    std::vector<float *> all_values(slot_size);
    std::vector<int64_t> slot_lengths(slot_size);
    for (auto i = 0; i < slot_size; i++) {
      const auto *slot = inputs[i];
      const uint64_t *single_slot_keys =
          reinterpret_cast<const uint64_t *>(slot->data<int64_t>());
      all_keys[i] = single_slot_keys;
      const auto key_numel = slot->numel();
      auto *output = outputs[i]->mutable_data<float>(ctx.GetPlace());
      all_values[i] = output;
      slot_lengths[i] = key_numel;
      printf("paddlebox: numel in the %d slot is %ld\n", i, key_numel);
    }

    auto box_ptr = paddle::framework::BoxWrapper::GetInstance();
    box_ptr->PullSparse(ctx.scope(), ctx.GetPlace(), all_keys, all_values,
                        slot_lengths);
  }
};

template <typename T>
class PullBoxSparseGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto inputs = ctx.MultiInput<framework::Tensor>("Ids");
    auto d_output =
        ctx.MultiInput<framework::Tensor>(framework::GradVarName("Out"));

    const auto slot_size = inputs.size();
    std::vector<const uint64_t *> all_keys(slot_size);
    std::vector<const float *> all_grad_values(slot_size);
    std::vector<int64_t> slot_lengths(slot_size);
    for (auto i = 0; i < slot_size; i++) {
      const auto *slot = inputs[i];
      const uint64_t *single_slot_keys =
          reinterpret_cast<const uint64_t *>(slot->data<int64_t>());
      all_keys[i] = single_slot_keys;
      const float *grad_value = d_output[i]->data<float>();
      const auto key_numel = slot->numel();
      all_grad_values[i] = grad_value;
      slot_lengths[i] = key_numel;
    }

    auto box_ptr = paddle::framework::BoxWrapper::GetInstance();
    box_ptr->PushSparseGrad(ctx.scope(), ctx.GetPlace(), all_keys,
                            all_grad_values, slot_lengths);
  }
};
}  // namespace operators
}  // namespace paddle
