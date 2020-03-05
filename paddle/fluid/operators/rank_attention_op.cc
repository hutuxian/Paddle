/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/rank_attention_op.h"
#include <memory>
#include <string>
#include <vector>

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

class RankAttentionOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::InvalidArgument(
                          "Input(X) of RankAttentionOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("RankOffset"), true,
        platform::errors::InvalidArgument(
            "Input(RankOffset) of RankAttentionOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("RankParam"), true,
        platform::errors::InvalidArgument(
            "Input(RankParam) of RankAttentionOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Out"), true,
        platform::errors::InvalidArgument(
            "Output(Out) of RankAttentionOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("InputHelp"), true,
        platform::errors::InvalidArgument(
            "Output(InputHelp) of RankAttentionOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("ParamHelp"), true,
        platform::errors::InvalidArgument(
            "Output(ParamHelp) of RankAttentionOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("InsRank"), true,
        platform::errors::InvalidArgument(
            "Output(InsRank) of RankAttentionOp should not be null."));
    auto max_rank = ctx->Attrs().Get<int>("MaxRank");

    auto x_dims = ctx->GetInputDim("X");
    auto ins_num = x_dims[0];
    auto x_fea_dim = x_dims[1];
    auto param_dims = ctx->GetInputDim("RankParam");
    auto para_col = param_dims[1];
    auto rank_offset_dims = ctx->GetInputDim("RankOffset");

    PADDLE_ENFORCE_EQ((rank_offset_dims[1] - 1) / 2, max_rank,
                      platform::errors::InvalidArgument(
                          "Input(RankOffset) has wrong columns."));

    auto block_matrix_row = max_rank * x_fea_dim;

    ctx->SetOutputDim("Out", {ins_num, para_col});
    ctx->SetOutputDim("ParamHelp", {ins_num * block_matrix_row, para_col});
    ctx->SetOutputDim("InputHelp", {ins_num, block_matrix_row});
    ctx->SetOutputDim("InsRank", {ins_num, 1});
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class RankAttentionGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::InvalidArgument("Input(X) should not be null"));
    PADDLE_ENFORCE_EQ(ctx->HasInput("RankParam"), true,
                      platform::errors::InvalidArgument(
                          "Input(RankParam) should not be null"));
    PADDLE_ENFORCE_EQ(ctx->HasInput("InputHelp"), true,
                      platform::errors::InvalidArgument(
                          "Input(InputHelp) should not be null"));
    PADDLE_ENFORCE_EQ(ctx->HasInput("ParamHelp"), true,
                      platform::errors::InvalidArgument(
                          "Input(ParamHelp) should not be null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("InsRank"), true,
        platform::errors::InvalidArgument("Input(InsRank) should not be null"));

    // ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    ctx->SetOutputDim(framework::GradVarName("RankParam"),
                      ctx->GetInputDim("RankParam"));
    // ctx->ShareLoD("X", /*-->*/ framework::GradVarName("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

class RankAttentionOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input tensor of rank_attention_Op operator.");
    AddInput("RankOffset",
             "(Tensor) Input tensor of rank_attention_Op operator.");
    AddInput("RankParam",
             "(Tensor) Input tensor of rank_attention_Op operator.");
    AddOutput("Out", "Output tensor of rank_attention_Op operator.");
    AddOutput("InputHelp", "Output tensor of rank_attention_Op operator.");
    AddOutput("ParamHelp", "Output tensor of rank_attention_Op operator.");
    AddOutput("InsRank", "Output tensor of rank_attention_Op operator.");
    AddAttr<int>("MaxRank", "(int, default 3) max rank of rank_attention_Op")
        .SetDefault(3);
    AddComment(R"DOC(
RankAttention Operator.
RankAttention  the input tensors along dimension axis.
)DOC");
  }
};

template <typename T>
class RankAttentionGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  std::unique_ptr<T> Apply() const override {
    std::unique_ptr<T> op(new T());
    op->SetType("rank_attention_grad");

    op->SetInput("X", this->Input("X"));
    op->SetInput("RankOffset", this->Input("RankOffset"));
    op->SetInput("RankParam", this->Input("RankParam"));
    op->SetInput("InputHelp", this->Output("InputHelp"));
    op->SetInput("ParamHelp", this->Output("ParamHelp"));
    op->SetInput("InsRank", this->Output("InsRank"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));

    // op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("RankParam"),
                  this->InputGrad("RankParam"));
    op->SetAttrMap(this->Attrs());
    return op;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(rank_attention, ops::RankAttentionOp,
                  ops::RankAttentionOpMaker,
                  ops::RankAttentionGradOpMaker<paddle::framework::OpDesc>,
                  ops::RankAttentionGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(rank_attention_grad, ops::RankAttentionGradOp);

REGISTER_OP_CPU_KERNEL(
    rank_attention,
    ops::RankAttentionKernel<paddle::platform::CPUDeviceContext, float>);
