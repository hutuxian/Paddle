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
#include <memory>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

class ScaledCastOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor of cast op");
    AddOutput("Out", "The output tensor of cast op");
    AddAttr<int>("out_dtype", "output data type");
    AddAttr<int>("in_dtype", "input data type");
    //AddAttr<float>("scale_factor", "(default 1.0) the scale of input data").SetDefault(1.0);
    AddAttr<float>("scale_factor", "(default 1.0) the scale of input data");
    AddComment(R"DOC(
Cast Operator.

This Operator casts the input tensor to another data type and
returns the Output Tensor. It's meaningless if the output dtype equals
the input dtype, but it's fine if you do so.

)DOC");
  }
};

template <typename T>
class ScaledCastOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad) const override {
    grad->SetType("scaled_cast_back");
    grad->SetInput("X", this->OutputGrad("Out"));
    grad->SetOutput("Out", this->InputGrad("X"));
    grad->SetAttr("out_dtype", this->GetAttr("in_dtype"));
    grad->SetAttr("in_dtype", this->GetAttr("out_dtype"));
    grad->SetAttr("scale_factor", this->GetAttr("scale_factor"));
  }
};

class ScaledCastOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "scaled_cast");
    OP_INOUT_CHECK(context->HasOutput("Out"), "Output", "Out", "scaled_cast");
    context->SetOutputDim("Out", context->GetInputDim("X"));
    context->ShareLoD("X", "Out");
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::OpKernelType kt = OperatorWithKernel::GetExpectedKernelType(ctx);
    // CastOp kernel's device type is decided by input tensor place
    kt.place_ = ctx.Input<framework::LoDTensor>("X")->place();
    return kt;
  }
};

///////////
// back op
class ScaledCastBackOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor of cast op");
    AddOutput("Out", "The output tensor of cast op");
    AddAttr<int>("out_dtype", "output data type");
    AddAttr<int>("in_dtype", "input data type");
    //AddAttr<float>("scale_factor", "(default 1.0) the scale of input data").SetDefault(1.0);
    AddAttr<float>("scale_factor", "(default 1.0) the scale of input data");
    AddComment(R"DOC(
Cast Operator.

This Operator casts the input tensor to another data type and
returns the Output Tensor. It's meaningless if the output dtype equals
the input dtype, but it's fine if you do so.

)DOC");
  }
};

class ScaledCastBackOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "scaled_cast_back");
    OP_INOUT_CHECK(context->HasOutput("Out"), "Output", "Out", "scaled_cast_back");
    context->SetOutputDim("Out", context->GetInputDim("X"));
    context->ShareLoD("X", "Out");
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::OpKernelType kt = OperatorWithKernel::GetExpectedKernelType(ctx);
    // CastOp kernel's device type is decided by input tensor place
    kt.place_ = ctx.Input<framework::LoDTensor>("X")->place();
    return kt;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;
REGISTER_OPERATOR(scaled_cast, ops::ScaledCastOp, ops::ScaledCastOpProtoMaker,
                  ops::ScaledCastOpGradMaker<paddle::framework::OpDesc>,
                  ops::ScaledCastOpGradMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(scaled_cast, 
                       ops::ScaledCastOpKernel<CPU, float>,
                       ops::ScaledCastOpKernel<CPU, double>,
                       //ops::ScaledCastOpKernel<CPU, int>,
                       //ops::ScaledCastOpKernel<CPU, int64_t>,
                       //ops::ScaledCastOpKernel<CPU, bool>,
                       //ops::ScaledCastOpKernel<CPU, uint8_t>,
                       ops::ScaledCastOpKernel<CPU, paddle::platform::float16>);

REGISTER_OPERATOR(scaled_cast_back, ops::ScaledCastBackOp, ops::ScaledCastBackOpProtoMaker);

REGISTER_OP_CPU_KERNEL(scaled_cast_back, 
                        ops::ScaledCastBackOpKernel<CPU, float>,
                       ops::ScaledCastBackOpKernel<CPU, double>,
                       //ops::ScaledCastBackOpKernel<CPU, int>,
                       //ops::ScaledCastBackOpKernel<CPU, int64_t>,
                       //ops::ScaledCastBackOpKernel<CPU, bool>,
                       //ops::ScaledCastBackOpKernel<CPU, uint8_t>,
                       ops::ScaledCastBackOpKernel<CPU, paddle::platform::float16>);

