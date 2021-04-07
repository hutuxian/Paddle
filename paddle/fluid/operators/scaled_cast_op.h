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

#pragma once

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/transform.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename InT, typename OutT>
struct ScaledCastOpTransformFunctor {
  HOSTDEVICE OutT operator()(InT in, float scale_factor) const { return ( (static_cast<OutT>(in)) * ( static_cast<OutT>(scale_factor)) ); }
};

template <typename DeviceContext, typename InT>
struct ScaledCastOpFunctor {
  const framework::Tensor* in_;
  framework::Tensor* out_;
  framework::Tensor* scale_factor_;
  const DeviceContext& ctx_;
  ScaledCastOpFunctor(const framework::Tensor* in, framework::Tensor* out, framework::Tensor* scale_factor,
                const DeviceContext& ctx)
      : in_(in), out_(out), scale_factor_(scale_factor), ctx_(ctx) {}

  template <typename OutT>
  void apply() const {
    auto* in_begin = in_->data<InT>();
    auto numel = in_->numel();
    auto* in_end = in_begin + numel;
    auto* out_begin = out_->mutable_data<OutT>(ctx_.GetPlace());
    ////
    //scale begin
    //auto* scale_begin = &scale_factor_;
    auto* scale_begin = scale_factor_->data<float>();
    ////
    platform::Transform<DeviceContext> trans;
    //trans(ctx_, in_begin, in_end, out_begin,
    //      CastOpTransformFunctor<InT, OutT>());
    trans(ctx_, in_begin, in_end, scale_begin, out_begin, ScaledCastOpTransformFunctor<InT, OutT>());

  }
};


// cast op func
template <typename InT, typename OutT>
struct CastOpTransformFunctor {
  HOSTDEVICE OutT operator()(InT in) const { return static_cast<OutT>(in); }
};

template <typename DeviceContext, typename InT>
struct CastOpFunctor {
  const framework::Tensor* in_;
  framework::Tensor* out_;
  const DeviceContext& ctx_;
  CastOpFunctor(const framework::Tensor* in, framework::Tensor* out,
                const DeviceContext& ctx)
      : in_(in), out_(out), ctx_(ctx) {}

  template <typename OutT>
  void apply() const {
    auto* in_begin = in_->data<InT>();
    auto numel = in_->numel();
    auto* in_end = in_begin + numel;
    auto* out_begin = out_->mutable_data<OutT>(ctx_.GetPlace());
    platform::Transform<DeviceContext> trans;
    trans(ctx_, in_begin, in_end, out_begin,
          CastOpTransformFunctor<InT, OutT>());
  }
};

// forward op
template <typename DeviceContext, typename InT>
class ScaledCastOpKernel : public framework::OpKernel<InT> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");
    
      framework::VisitDataType(
        static_cast<framework::proto::VarType::Type>(
            context.Attr<int>("out_dtype")),
        CastOpFunctor<DeviceContext, InT>(
            in, out, context.template device_context<DeviceContext>()));
  }
};

// back op
template <typename DeviceContext, typename InT>
class ScaledCastBackOpKernel : public framework::OpKernel<InT> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");

    const float scale_factor_backward = context.Attr<float>("scale_factor");
      
    framework::Tensor param_helper;
    auto input_dims = in->dims();
    param_helper.mutable_data<float>(input_dims, (context.template device_context<DeviceContext>()).GetPlace());
    math::set_constant(context.template device_context<DeviceContext>(), &param_helper, scale_factor_backward);

    framework::VisitDataType(
        static_cast<framework::proto::VarType::Type>(
            context.Attr<int>("out_dtype")),
        ScaledCastOpFunctor<DeviceContext, InT>(
            in, out, &param_helper, context.template device_context<DeviceContext>()));

  }
};


}  // namespace operators
}  // namespace paddle
