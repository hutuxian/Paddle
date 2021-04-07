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

#include <cublas.h>
#include <string>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/operators/scaled_fc_op.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace operators {
using framework::Tensor;

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = paddle::platform::PADDLE_CUDA_NUM_THREADS;
static inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// add the same row vector to all matrix rows
template <typename T>
__global__ void kernel_vec_mat_row_add(const int N, const unsigned int rown,
                                       const unsigned int coln, T* matrix,
                                       const T* vector, const T bias_scale_factor_use) {
  CUDA_KERNEL_LOOP(i, N) { matrix[i] += vector[i % coln] * bias_scale_factor_use; }
}

template <typename T>
void vec_mat_row_add(cudaStream_t stream, const unsigned int rown,
                     const unsigned int coln, T* matrix, const T* vector, const T bias_scale_factor_use) {
  int N = rown * coln;
  kernel_vec_mat_row_add<<<GET_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>(
      N, rown, coln, matrix, vector, bias_scale_factor_use);
}

// calculate col sum of a mat
template <typename T>
__global__ void kernel_add_col_sum_mat(const unsigned int rown,
                                       const unsigned int coln, const T* matrix,
                                       T* vector, const T bias_scale_factor_use) {
  CUDA_KERNEL_LOOP(i, coln) {
    for (unsigned int j = 0; j < rown; j++) {
      // vector[i] += matrix[i * rown + j];
      vector[i] += matrix[j * coln + i] * bias_scale_factor_use;
    }
  }
}

template <typename T>
void col_sum_mat(cudaStream_t stream, const unsigned int rown,
                 const unsigned int coln, const T* matrix, T* vector,
                 const T bias_scale_factor_use) {
  kernel_add_col_sum_mat<<<GET_BLOCKS(coln), CUDA_NUM_THREADS, 0, stream>>>(
      rown, coln, matrix, vector, bias_scale_factor_use);
}


template <typename DeviceContext, typename T>
class ScaledFCCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    //VLOG(0) << "begin compute.";
    auto* input = ctx.Input<framework::LoDTensor>("Input");
    auto* w = ctx.Input<Tensor>("W");
    auto* bias = ctx.Input<Tensor>("Bias");
    auto* output = ctx.Output<framework::LoDTensor>("Out");
    auto input_scale_factor = ctx.Attr<float>("input_scale_factor");
    auto bias_scale_factor = ctx.Attr<float>("bias_scale_factor");

    auto input_dims = input->dims();
    auto w_dims = w->dims();
    auto ins_num = input_dims[0];  // oriinput: ins_num*in_feat, oriweight: in_feat* out_fea, output: ins_num* out_feat
    auto in_feat = input_dims[1];
    auto out_feat = w_dims[1];

    // get data ptr
    const T* in_data = input->data<T>();
    const T* w_data = w->data<T>();
    const T* bias_data = bias->data<T>();

    output->mutable_data<T>(ctx.GetPlace());
    output->Resize({ins_num, w_dims[1]});

    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;

    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto blas = math::GetBlas<platform::CUDADeviceContext, T>(dev_ctx);

    T alpha = static_cast<T>(input_scale_factor);
    T bias_scale_factor_use = static_cast<T>(bias_scale_factor);
    T beta = static_cast<T>(0.0);
    blas.GEMM(transA, transB, ins_num, out_feat, in_feat, alpha, input->data<T>(), w->data<T>(), beta, output->data<T>());
    vec_mat_row_add<T>(ctx.cuda_device_context().stream(), ins_num, w_dims[1],
                       output->data<T>(), bias->data<T>(), bias_scale_factor_use);
  }
};

template <typename DeviceContext, typename T>
class ScaledFCGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("Input");
    auto* w = ctx.Input<Tensor>("W");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));

    auto input_scale_factor = ctx.Attr<float>("input_scale_factor");
    auto bias_scale_factor = ctx.Attr<float>("bias_scale_factor");

    T alpha = static_cast<T>(input_scale_factor);
    T bias_scale_factor_use = static_cast<T>(bias_scale_factor);
    T beta = static_cast<T>(0.0);

    auto* dx = ctx.Output<Tensor>(framework::GradVarName("Input"));
    auto* dw = ctx.Output<Tensor>(framework::GradVarName("W"));
    auto* db = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    auto input_dims = input->dims(); //ins_num*in_feat
    auto dout_dims = dout->dims(); //ins_num*out_feat
    auto w_dims = w->dims(); //in_feat*out_feat

    auto dout_coln = dout_dims[1];
    auto ins_num = dout_dims[0];

    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto& place = *ctx.template device_context<platform::CUDADeviceContext>()
                       .eigen_device();
    auto stream = ctx.cuda_device_context().stream();
    // initialize
    dx->mutable_data<T>(ctx.GetPlace());
    auto dx_eigen = framework::EigenVector<T>::Flatten(*dx);
    dx_eigen.device(place) = dx_eigen.constant(static_cast<T>(0));

    dw->mutable_data<T>(ctx.GetPlace());
    auto dw_eigen = framework::EigenVector<T>::Flatten(*dw);
    dw_eigen.device(place) = dw_eigen.constant(static_cast<T>(0));

    db->mutable_data<T>(ctx.GetPlace());
    auto db_eigen = framework::EigenVector<T>::Flatten(*db);
    db_eigen.device(place) = db_eigen.constant(static_cast<T>(0));

    // get bias grad
    col_sum_mat(stream, ins_num, dout_coln, dout->data<T>(), db->data<T>(), bias_scale_factor_use);

    // input & weight grad
    auto blas = math::GetBlas<platform::CUDADeviceContext, T>(dev_ctx);
    
    auto input_mat = static_cast<const Tensor&>(*input);
    auto w_mat = static_cast<const Tensor&>(*w);
    auto dout_mat = static_cast<const Tensor&>(*dout);

    // dx = dout_data * y^T, ins_num*in_feat
    //dout_data : ins_num*out_feat
    //blas.GEMM(CblasNoTrans, CblasTrans, dout_dims[0], w_dims[0], w_dims[1], alpha, dout.data<T>(), w.data<T>(), beta, dx.data<T>());
    blas.GEMM(CblasNoTrans, CblasTrans, dout_dims[0], w_dims[0], w_dims[1], alpha, dout_mat.data<T>(), w_mat.data<T>(), beta, dx->data<T>());
    
    // dy = x^T * dout_data
    //blas.GEMM(CblasTrans, CblasNoTrans, input_dims[1], dout_dims[1], input_dims[0], alpha, input.data<T>(), dout.data<T>(), beta, dw->data<T>());
    blas.GEMM(CblasTrans, CblasNoTrans, input_dims[1], dout_dims[1], input_dims[0], alpha, input_mat.data<T>(), dout_mat.data<T>(), beta, dw->data<T>());

  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using GPUCtx = paddle::platform::CUDADeviceContext;
REGISTER_OP_CUDA_KERNEL(scaled_fc, ops::ScaledFCCUDAKernel<GPUCtx, float>,
                        ops::ScaledFCCUDAKernel<GPUCtx, double>,
                        ops::ScaledFCCUDAKernel<GPUCtx, paddle::platform::float16>);

REGISTER_OP_CUDA_KERNEL(scaled_fc_grad,
                        ops::ScaledFCGradOpCUDAKernel<GPUCtx, float>,
                        ops::ScaledFCGradOpCUDAKernel<GPUCtx, double>,
                        ops::ScaledFCGradOpCUDAKernel<GPUCtx, paddle::platform::float16>);
                      
