// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <sstream>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/device_memory_aligment.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SliceMultiTensorOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto fuse_tensor = context.Input<framework::LoDTensor>("Input");
    auto in_tensors = context.MultiInput<framework::LoDTensor>("X");
    // Init the continuous space
    auto out_tensors = context.MultiOutput<framework::LoDTensor>("Output");

    int id = context.Attr<int>("id");
    int num = context.Attr<int>("num");

    size_t in_size = in_tensors.size();
    size_t out_size = out_tensors.size();
    // num data
    CHECK(in_size == out_size || out_size / num == in_size);

    // Make the outputs point to the continuous space.
    int64_t numel = fuse_tensor->numel();
    int64_t offset = (id * numel) / num;

    //    fprintf(stdout, "fuse length: %d(dim: %s), in size: %d(dim: %s),
    //    offset: %d\n",
    //            int(fused_tensor->numel()),
    //            fused_tensor->dims().to_str().c_str(),
    //            int(in_tensors[0]->numel()),
    //            in_tensors[0]->dims().to_str().c_str(),
    //            int(offset));

    auto &fuse_dim = fuse_tensor->dims();
    // adjust fuse
    if (fuse_dim.size() > 1 && fuse_dim[0] != numel) {
      paddle::framework::DDim dim(fuse_dim);
      dim[0] = numel;
      dim[1] = 1;
      const_cast<framework::LoDTensor *>(fuse_tensor)->Resize(dim);
    }

    for (size_t i = 0; i < out_tensors.size(); ++i) {
      size_t idx = i % in_size;
      auto dim = in_tensors[idx]->dims();
      size_t len = static_cast<size_t>(in_tensors[idx]->numel());
      CHECK(static_cast<int64_t>(offset + len) <= numel)
          << "fuse dim: " << fuse_dim.to_str() << ", dim:" << dim.to_str()
          << ", offset:" << offset << ", len:" << len;
      // slice tensor
      out_tensors[i]
          ->ShareDataWith(fuse_tensor->Slice(
              static_cast<int64_t>(offset), static_cast<int64_t>(offset + len)))
          .Resize(dim);
      offset += len;
    }
  }
};

class SliceMultiTensorOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {}

 protected:
  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const framework::Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   expected_kernel_type.place_,
                                   tensor.layout());
  }
};

class SliceMultiTensorOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input",
             "(LoDTensor) The input tensor of"
             " slice_multi_tensor operator.")
        .AsDuplicable();
    AddInput("X",
             "(vector<LoDTensor>) The input tensor of"
             " slice_multi_tensor operator.")
        .AsDuplicable();
    AddOutput("Output",
              "(vector<LoDTensor>) The output "
              "tensors of slice_multi_tensor operator. And the address "
              "of output tensors are continuous, they are sliced from the "
              "tensor of FusedOutput.")
        .AsDuplicable();
    AddAttr<int>("id", "split id").SetDefault(0);
    AddAttr<int>("num", "split input tensor time").SetDefault(1);
    AddComment(R"DOC(
SliceMultiTensor Operator.

slice_multi_tensor is used split one ternsor to mulit child tensor

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(slice_multi_tensor, paddle::operators::SliceMultiTensorOp,
                  paddle::operators::SliceMultiTensorOpMaker);
REGISTER_OP_CPU_KERNEL(
    slice_multi_tensor,
    ops::SliceMultiTensorOpKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SliceMultiTensorOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SliceMultiTensorOpKernel<paddle::platform::CPUDeviceContext, double>);

#ifdef PADDLE_WITH_CUDA
REGISTER_OP_CUDA_KERNEL(
    slice_multi_tensor,
    ops::SliceMultiTensorOpKernel<paddle::platform::CUDADeviceContext,
                                  plat::float16>,
    ops::SliceMultiTensorOpKernel<paddle::platform::CUDADeviceContext, int>,
    ops::SliceMultiTensorOpKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SliceMultiTensorOpKernel<paddle::platform::CUDADeviceContext, double>);
#endif
