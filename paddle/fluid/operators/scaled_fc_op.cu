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

// cast to fp16 & padding
template <typename T>
__global__ void kernel_cast_and_padding(const int N, 
                      const unsigned int rown_ori, const unsigned int coln_ori,
                      const unsigned int rown_pad, const unsigned int coln_pad,
                      const T* matrix, paddle::platform::float16* matrix_pad,
                      T grad_scale_factor) {
  CUDA_KERNEL_LOOP(i, N) { 
      int col_idx = i % coln_pad;
      int row_idx = i / coln_pad;
      if (row_idx < rown_ori && col_idx < coln_ori) {
          int idx = row_idx * coln_ori + col_idx;
          //matrix_pad[i] = static_cast<paddle::platform::float16>(matrix[idx]);
          matrix_pad[i] = static_cast<paddle::platform::float16>(matrix[idx] * grad_scale_factor);
      } else {
          matrix_pad[i] = static_cast<paddle::platform::float16>(0.0);
      }
  }
}

template <typename T>
void cast_and_padding(cudaStream_t stream, 
                      const unsigned int rown_ori, const unsigned int coln_ori,
                      const unsigned int rown_pad, const unsigned int coln_pad, 
                      const T* matrix, paddle::platform::float16* matrix_pad,
                      T grad_scale_factor) {
                      //T grad_scale_factor = static_cast<T>(1.0)) {
  int N = rown_pad * coln_pad;
  kernel_cast_and_padding<<<GET_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>(
      N, rown_ori, coln_ori, rown_pad, coln_pad, matrix, matrix_pad, grad_scale_factor);
}
// end cast to fp16 & padding


// cast to fp32 & cut
template <typename T>
__global__ void kernel_cast_and_cut(const int N, 
                      const unsigned int rown_ori, const unsigned int coln_ori,
                      const unsigned int rown_pad, const unsigned int coln_pad,
                      T* matrix, paddle::platform::float16* matrix_pad,
                      T scale_factor) {
  CUDA_KERNEL_LOOP(i, N) { 
      int col_idx = i % coln_ori;
      int row_idx = i / coln_ori;
      int idx = row_idx * coln_pad + col_idx;
      matrix[i] = static_cast<T>(matrix_pad[idx]);
      matrix[i] *= scale_factor;
  }
}

template <typename T>
void cast_and_cut(cudaStream_t stream, 
                      const unsigned int rown_ori, const unsigned int coln_ori,
                      const unsigned int rown_pad, const unsigned int coln_pad, 
                      T* matrix, paddle::platform::float16* matrix_pad,
                      T scale_factor) {
  int N = rown_ori * coln_ori;
  kernel_cast_and_cut<<<GET_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>(
      N, rown_ori, coln_ori, rown_pad, coln_pad, matrix, matrix_pad, scale_factor);
}
// end cast to fp32 & cut

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
      ////vector[i] += matrix[i * rown + j];
      //vector[i] += matrix[j * coln + i] * bias_scale_factor_use;
      vector[i] += matrix[j * coln + i];
    }
  }
}

//col_sum_mat(stream, ins_num, dout_coln, dout->data<T>(), db->data<T>(), bias_scale_factor_use);
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
    auto* input = ctx.Input<framework::LoDTensor>("Input"); // framework::Tensor*
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

    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    // cast and pad
    const unsigned int insnum_ori = ins_num;
    const unsigned int infea_ori = in_feat;
    const unsigned int outfea_ori = out_feat;
    
    const unsigned int insnum_pad = (insnum_ori % 8) == 0 ? insnum_ori : insnum_ori + (8 - insnum_ori % 8);
    const unsigned int infea_pad = (infea_ori % 8) == 0 ? infea_ori : infea_ori + (8 - infea_ori % 8);
    const unsigned int outfea_pad = (outfea_ori % 8) == 0 ? outfea_ori : outfea_ori + (8 - outfea_ori % 8);

    framework::Tensor input_help;
    input_help = ctx.AllocateTmpTensor<paddle::platform::float16, DeviceContext>({insnum_pad, infea_pad}, dev_ctx);

    framework::Tensor w_help;
    w_help = ctx.AllocateTmpTensor<paddle::platform::float16, DeviceContext>({infea_pad, outfea_pad}, dev_ctx);

    framework::Tensor bias_help;
    bias_help = ctx.AllocateTmpTensor<paddle::platform::float16, DeviceContext>({outfea_pad, 1}, dev_ctx);

    framework::Tensor output_help;
    output_help = ctx.AllocateTmpTensor<paddle::platform::float16, DeviceContext>({insnum_pad, outfea_pad}, dev_ctx);

    T scale = static_cast<T>(1.0);
    cast_and_padding<T>(ctx.cuda_device_context().stream(), insnum_ori, infea_ori, insnum_pad, infea_pad, input->data<T>(), input_help.mutable_data<paddle::platform::float16>(ctx.GetPlace()), scale);
    cast_and_padding<T>(ctx.cuda_device_context().stream(), infea_ori, outfea_ori, infea_pad, outfea_pad, w->data<T>(), w_help.mutable_data<paddle::platform::float16>(ctx.GetPlace()), scale);
    cast_and_padding<T>(ctx.cuda_device_context().stream(), outfea_ori, 1, outfea_pad, 1, bias->data<T>(), bias_help.mutable_data<paddle::platform::float16>(ctx.GetPlace()), scale);
    VLOG(3) << "input dim0=" << input->dims()[0] << ", input dim1=" << input->dims()[1]
            << ", input_help dim0=" << input_help.dims()[0] << ", input_help dim1=" << input_help.dims()[1];
    VLOG(3) << "w dim0=" << w->dims()[0] << ", w dim1=" << w->dims()[1]
            << ", w_help dim0=" << w_help.dims()[0] << ", w_help dim1=" << w_help.dims()[1];
    VLOG(3) << "bias dim0=" << bias->dims()[0] << ", bias dim1=" << bias->dims()[1]
            << ", bias_help dim0=" << bias_help.dims()[0] << ", bias_help dim1=" << bias_help.dims()[1];

    // end cast and pad

    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;

    //auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    //auto blas = math::GetBlas<platform::CUDADeviceContext, T>(dev_ctx);
    auto blas = math::GetBlas<platform::CUDADeviceContext, paddle::platform::float16>(dev_ctx);

    //T alpha = static_cast<T>(input_scale_factor);
    //T bias_scale_factor_use = static_cast<T>(bias_scale_factor);
    //T beta = static_cast<T>(0.0);
    paddle::platform::float16 alpha = static_cast<paddle::platform::float16>(input_scale_factor);
    paddle::platform::float16 bias_scale_factor_use = static_cast<paddle::platform::float16>(bias_scale_factor);
    paddle::platform::float16 beta = static_cast<paddle::platform::float16>(0.0);

    //blas.GEMM(transA, transB, ins_num, out_feat, in_feat, alpha, input->data<T>(), w->data<T>(), beta, output->data<T>());
    //blas.GEMM(transA, transB, insnum_pad, outfea_pad, infea_pad, alpha, input_help->data<paddle::platform::float16>(), w_help->data<paddle::platform::float16>(), beta, output_help->data<paddle::platform::float16>());
    blas.GEMM(transA, transB, insnum_pad, outfea_pad, infea_pad, alpha, input_help.data<paddle::platform::float16>(), w_help.data<paddle::platform::float16>(), beta, output_help.mutable_data<paddle::platform::float16>(ctx.GetPlace()));
    //blas.GEMM(transA, transB, insnum_pad, outfea_pad, infea_pad, alpha, &input_help, &w_help, beta, &output_help);
    //vec_mat_row_add<T>(ctx.cuda_device_context().stream(), ins_num, w_dims[1],
    //                   output->data<T>(), bias->data<T>(), bias_scale_factor_use);
    //vec_mat_row_add<paddle::platform::float16>(ctx.cuda_device_context().stream(), insnum_pad, outfea_pad,
    //                   output_help->data<paddle::platform::float16>(), bias_help->data<paddle::platform::float16>(), bias_scale_factor_use);
    vec_mat_row_add<paddle::platform::float16>(ctx.cuda_device_context().stream(), insnum_pad, outfea_pad,
                       output_help.data<paddle::platform::float16>(), bias_help.data<paddle::platform::float16>(), bias_scale_factor_use);

    //cast_and_cut<T>(ctx.cuda_device_context().stream(), insnum_ori, outfea_ori, insnum_pad, outfea_pad, output->data<T>(), output_help->data<paddle::platform::float16>());
    T scale_factor = static_cast<T>(1 / input_scale_factor);
    VLOG(3) << "input_scale_factor=" << input_scale_factor
            << ", bias_scale_factor_use=" << bias_scale_factor_use
            << ", output scale_factor=" << scale_factor;
    cast_and_cut<T>(ctx.cuda_device_context().stream(), insnum_ori, outfea_ori, insnum_pad, outfea_pad, output->data<T>(), output_help.data<paddle::platform::float16>(), scale_factor);
    VLOG(3) << "output_help dim0=" << output_help.dims()[0] << ", output_help dim1=" << output_help.dims()[1]
            << ", output dim0=" << output->dims()[0] << ", output dim1=" << output->dims()[1];
    //cast_and_cut<T>(ctx.cuda_device_context().stream(), insnum_ori, outfea_ori, insnum_pad, outfea_pad, output->data<T>(), &output_help);

  }
};

template <typename DeviceContext, typename T>
class ScaledFCGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("Input");
    auto* w = ctx.Input<Tensor>("W");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out")); // insnum * outfea

    auto input_scale_factor = ctx.Attr<float>("input_scale_factor");
    auto bias_scale_factor = ctx.Attr<float>("bias_scale_factor");
    auto grad_scale_factor = ctx.Attr<float>("grad_scale_factor");

    //T alpha = static_cast<T>(input_scale_factor);
    T bias_scale_factor_use = static_cast<T>(bias_scale_factor);
    //T beta = static_cast<T>(0.0);
    paddle::platform::float16 alpha = static_cast<paddle::platform::float16>(input_scale_factor);
    //paddle::platform::float16 bias_scale_factor_use = static_cast<paddle::platform::float16>(bias_scale_factor);
    paddle::platform::float16 beta = static_cast<paddle::platform::float16>(0.0);

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

    // fp16: cast and pad
    const unsigned int insnum_ori = input_dims[0];
    const unsigned int infea_ori = input_dims[1];
    const unsigned int outfea_ori = w_dims[1];
    
    const unsigned int insnum_pad = (insnum_ori % 8) == 0 ? insnum_ori : insnum_ori + (8 - insnum_ori % 8);
    const unsigned int infea_pad = (infea_ori % 8) == 0 ? infea_ori : infea_ori + (8 - infea_ori % 8);
    const unsigned int outfea_pad = (outfea_ori % 8) == 0 ? outfea_ori : outfea_ori + (8 - outfea_ori % 8);
    VLOG(3) << "input dim0=" << input_dims[0] << ", input dim1=" << input_dims[1]
            << ", dout dim0=" << dout_dims[0] << ", dout dim1=" << dout_dims[1]
            << ", w dim0=" << w_dims[0] << ", w dim1=" << w_dims[1]
            << ", insnum_ori=" << insnum_ori << ", insnum_pad=" << insnum_pad
            << ", infea_ori=" << infea_ori << ", infea_pad=" << infea_pad
            << ", outfea_ori=" << outfea_ori << ", outfea_pad=" << outfea_pad;

    framework::Tensor dx_help;
    dx_help = ctx.AllocateTmpTensor<paddle::platform::float16, DeviceContext>({insnum_pad, infea_pad}, dev_ctx);

    framework::Tensor dw_help;
    dw_help = ctx.AllocateTmpTensor<paddle::platform::float16, DeviceContext>({infea_pad, outfea_pad}, dev_ctx);

    framework::Tensor dout_help;
    dout_help = ctx.AllocateTmpTensor<paddle::platform::float16, DeviceContext>({insnum_pad, outfea_pad}, dev_ctx);

    framework::Tensor input_help;
    input_help = ctx.AllocateTmpTensor<paddle::platform::float16, DeviceContext>({insnum_pad, infea_pad}, dev_ctx);

    framework::Tensor w_help;
    w_help = ctx.AllocateTmpTensor<paddle::platform::float16, DeviceContext>({infea_pad, outfea_pad}, dev_ctx);

    T scale = static_cast<T>(1.0);
    cast_and_padding<T>(ctx.cuda_device_context().stream(), insnum_ori, infea_ori, insnum_pad, infea_pad, input->data<T>(), input_help.mutable_data<paddle::platform::float16>(ctx.GetPlace()), scale);
    cast_and_padding<T>(ctx.cuda_device_context().stream(), infea_ori, outfea_ori, infea_pad, outfea_pad, w->data<T>(), w_help.mutable_data<paddle::platform::float16>(ctx.GetPlace()), scale);
    //cast_and_padding<T>(ctx.cuda_device_context().stream(), insnum_ori, outfea_ori, insnum_pad, outfea_pad, dout->data<T>(), dout_help.mutable_data<paddle::platform::float16>(ctx.GetPlace()));
    //T grad_scale_factor = static_cast<T>(256.0) * static_cast<T>(1 / input_scale_factor);
    T dout_grad_scale_factor = static_cast<T>(grad_scale_factor) * static_cast<T>(1 / input_scale_factor);
    cast_and_padding<T>(ctx.cuda_device_context().stream(), insnum_ori, outfea_ori, insnum_pad, outfea_pad, dout->data<T>(), dout_help.mutable_data<paddle::platform::float16>(ctx.GetPlace()), dout_grad_scale_factor);

    
    auto blas = math::GetBlas<platform::CUDADeviceContext, paddle::platform::float16>(dev_ctx);
    //dx = dy * w^T
    blas.GEMM(CblasNoTrans, CblasTrans, insnum_pad, infea_pad, outfea_pad, alpha, dout_help.data<paddle::platform::float16>(), w_help.data<paddle::platform::float16>(), beta, dx_help.mutable_data<paddle::platform::float16>(ctx.GetPlace()));
    //dw = x^T * dy
    blas.GEMM(CblasTrans, CblasNoTrans, infea_pad, outfea_pad, insnum_pad, alpha, input_help.data<paddle::platform::float16>(), dout_help.data<paddle::platform::float16>(), beta, dw_help.mutable_data<paddle::platform::float16>(ctx.GetPlace()));

    //cast dx dw to fp32 and cut
    //T scale_factor = static_cast<T>(1 / input_scale_factor);
    //T scale_factor = static_cast<T>(1.0 / 256.0);
    T scale_factor = static_cast<T>(1.0 / grad_scale_factor);
    VLOG(3) << "input_scale_factor=" << input_scale_factor
            << ", bias_scale_factor_use=" << bias_scale_factor_use
            << ", dout_grad_scale_factor=" << dout_grad_scale_factor
            << ", dx_grad dw_grad scale_factor=" << scale_factor;

    cast_and_cut<T>(ctx.cuda_device_context().stream(), insnum_ori, infea_ori, insnum_pad, infea_pad, dx->data<T>(), dx_help.data<paddle::platform::float16>(), scale_factor);
    cast_and_cut<T>(ctx.cuda_device_context().stream(), infea_ori, outfea_ori, infea_pad, outfea_pad, dw->data<T>(), dw_help.data<paddle::platform::float16>(), scale_factor);

    // end cast and pad

    /*
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
    */
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using GPUCtx = paddle::platform::CUDADeviceContext;
REGISTER_OP_CUDA_KERNEL(scaled_fc, ops::ScaledFCCUDAKernel<GPUCtx, float>,
                        ops::ScaledFCCUDAKernel<GPUCtx, double>);
                        //ops::ScaledFCCUDAKernel<GPUCtx, paddle::platform::float16>);

REGISTER_OP_CUDA_KERNEL(scaled_fc_grad,
                        ops::ScaledFCGradOpCUDAKernel<GPUCtx, float>,
                        ops::ScaledFCGradOpCUDAKernel<GPUCtx, double>);
                        //ops::ScaledFCGradOpCUDAKernel<GPUCtx, paddle::platform::float16>);
                      
