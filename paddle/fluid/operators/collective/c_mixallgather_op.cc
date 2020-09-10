/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device_memory_aligment.h"
#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/nccl_helper.h"
#endif
#include "paddle/fluid/operators/tensor_formatter.h"

namespace paddle {
namespace operators {

class CMixAllGatherOp : public framework::OperatorWithKernel {
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

template <typename T>
class CMixAllGatherOpCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_THROW("CMixAllGather op do not support CPUKernel for now.");
  }
};

// template<typename T>
// int check_illegal_count(const T* a, int size, char *buf) {
//    int zero = 0;
//    int nan = 0;
//    int inf = 0;
//    for (int i = 0; i < size; ++i) {
//        if (a[i] == 0) {
//            zero = zero + 1;
//        } else if (isnan(a[i])) {
//            nan = nan + 1;
//        } else if (isinf(a[i])) {
//            inf = inf + 1;
//        }
//    }
//    return snprintf(buf, 2048, "(SIZE:%d,NA:%d,INF:%d,ZERO:%d),", size, nan,
//    inf, zero);
//}
// void print_cpu_data(const char *name, int device, const void *address, const
// paddle::platform::float16 *a, int size) {
//
//}
// void print_cpu_data(const char *name, int device, const void *address, const
// double *a, int size) {
//
//}
// void print_cpu_data(const char *name, int device, const void *address, const
// int *a, int size) {
//
//}
// void print_cpu_data(const char *name, int device, const void *address, const
// int64_t *a, int size) {
//
//}
// template<typename T>
// void print_cpu_data(const char *name, int device, const void *address, const
// T *a, int size) {
//    char szbuf[8193] = {0};
//    int offset = check_illegal_count(a, size, szbuf);
//    if (size > 100) {
//        int step = size / 100;
//        for (int i = 0; i < size; i = i + step) {
//            offset += snprintf(szbuf + offset, 8192 - offset, "%f,", a[i]);
//        }
//    } else {
//        for (int i = 0; i < size; ++ i) {
//            offset += snprintf(szbuf + offset, 8192 - offset, "%f,", a[i]);
//        }
//    }
//    fprintf(stdout, "[%d]%s(%p):%s\n", device, name, address, szbuf);
//}
//
// template<typename T>
// void print_gpu_data(const char *name, const T *a, int size, int device,
// cudaStream_t stream) {
//    T *buf = 0;
//    cudaHostAlloc((void **)&buf, sizeof(T) * size, cudaHostAllocDefault);
//    cudaMemcpyAsync(buf, a, size * sizeof(float), cudaMemcpyDeviceToHost,
//    stream);
//    cudaStreamSynchronize(stream);
//    print_cpu_data(name, device, a, buf, size);
//    cudaFreeHost(buf);
//}

template <typename T>
class CMixAllGatherOpCUDAKernel : public framework::OpKernel<T> {
  static const int NCCL_MIXALLGATHER = 1;
  static const int NCCL_ALLGATHER = 2;

 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
#if defined(PADDLE_WITH_NCCL)
    auto in_tensors = ctx.MultiInput<framework::LoDTensor>("Input");
    auto fused_tensor = ctx.Output<framework::LoDTensor>("Output");
    //    auto in_var_names = ctx.InputNames("Input");

    int nranks = ctx.Attr<int>("nranks");
    int rank_id = ctx.Attr<int>("rankid");
    int nccl_mode = ctx.Attr<int>("nccl_mode");
    int ring_id = ctx.Attr<int>("ring_id");

    auto place = ctx.GetPlace();

    int device_id = boost::get<platform::CUDAPlace>(place).GetDeviceId();

    size_t numel = 0;
    auto dtype =
        static_cast<framework::proto::VarType::Type>(in_tensors[0]->type());
    GetTensorMemSize(in_tensors, &numel);

    int64_t offset = 0;
    size_t recv_len = 0;
    T *recvbuff = nullptr;
    T *sendbuff = nullptr;

    auto comm = platform::NCCLCommContext::Instance().Get(0, device_id);
    int device_num = comm->nranks();

    if (nccl_mode == NCCL_MIXALLGATHER) {  // mixallgather
      offset = numel * rank_id;
      recvbuff = fused_tensor->mutable_data<T>(
          {static_cast<int>(numel * nranks), 1}, place);
      sendbuff = &recvbuff[offset];
      recv_len = numel * nranks;
    } else if (nccl_mode == NCCL_ALLGATHER) {  // allgather
      offset = numel * (device_num * rank_id + device_id);
      recvbuff = fused_tensor->mutable_data<T>(
          {static_cast<int>(numel * nranks * device_num), 1}, place);
      sendbuff = &recvbuff[offset];
      recv_len = numel * nranks * device_num;
    } else {  // allreduce
      recvbuff =
          fused_tensor->mutable_data<T>({static_cast<int>(numel), 1}, place);
      sendbuff = recvbuff;
      recv_len = numel;
    }
    CHECK(static_cast<int64_t>(recv_len) == fused_tensor->numel());

    auto dev_ctx = paddle::platform::DeviceContextPool::Instance().Get(place);
    // copy input datas
    for (size_t i = 0; i < in_tensors.size(); ++i) {
      size_t len = static_cast<size_t>(in_tensors[i]->numel());
      auto sub_tensor = fused_tensor->Slice(static_cast<int64_t>(offset),
                                            static_cast<int64_t>(offset + len));
      framework::TensorCopy(*in_tensors[i], place, *dev_ctx, &sub_tensor);
      offset += len;
    }

    cudaStream_t stream = nullptr;
    if (ctx.Attr<bool>("use_calc_stream")) {
      stream = static_cast<platform::CUDADeviceContext *>(dev_ctx)->stream();
    } else {
      stream = static_cast<platform::CUDADeviceContext *>(dev_ctx)->stream();
      PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(stream));
      stream = comm->stream();
    }

    ncclDataType_t nccl_dtype = platform::ToNCCLDataType(dtype);
    // reduce device 0
    if (nranks > 1) {  // multi node
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclGroupStart());
      if (nccl_mode == NCCL_ALLGATHER) {  // allgather
        PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclAllGather(
            sendbuff, &recvbuff[numel * device_num * rank_id], numel,
            nccl_dtype, comm->comm(), stream));
        if (device_id == 0) {
          // node allgather
          auto node_comm =
              platform::NCCLCommContext::Instance().Get(ring_id, 0);
          PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclAllGather(
              &recvbuff[numel * device_num * rank_id], recvbuff,
              numel * device_num, nccl_dtype, node_comm->comm(), stream));
        }
      } else {  // mixallgather allreduce
        PADDLE_ENFORCE_CUDA_SUCCESS(
            platform::dynload::ncclReduce(sendbuff, sendbuff, numel, nccl_dtype,
                                          ncclSum, 0, comm->comm(), stream));
        if (device_id == 0) {
          auto node_comm =
              platform::NCCLCommContext::Instance().Get(ring_id, 0);
          if (nccl_mode == NCCL_MIXALLGATHER) {
            // allgather
            PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclAllGather(
                sendbuff, recvbuff, numel, nccl_dtype, node_comm->comm(),
                stream));
          } else {
            // allreduce
            PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclAllReduce(
                sendbuff, recvbuff, numel, nccl_dtype, ncclSum,
                node_comm->comm(), stream));
          }
        }
      }
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclGroupEnd());
      // broadcast to all device
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclBcast(
          recvbuff, recv_len, nccl_dtype, 0, comm->comm(), stream));
    } else {  // single node
      if (nccl_mode == NCCL_ALLGATHER) {
        PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclAllGather(
            sendbuff, recvbuff, numel, nccl_dtype, comm->comm(), stream));
      } else {
        PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclAllReduce(
            sendbuff, recvbuff, numel, nccl_dtype, ncclSum, comm->comm(),
            stream));
      }
    }
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(stream));
//    print_gpu_data("fuse_nccl", recvbuff, static_cast<int>(recv_len),
//    device_id, stream);
#else
    PADDLE_THROW("PaddlePaddle should compile with GPU.");
#endif
  }

 protected:
  void GetTensorMemSize(
      const std::vector<const framework::LoDTensor *> &lod_tensors,
      size_t *numel) const {
    *numel = 0;
    for (size_t i = 0; i < lod_tensors.size(); ++i) {
      CHECK(lod_tensors[i]->IsInitialized());
      *numel += lod_tensors[i]->numel();
    }
  }
};

class CMixAllGatherOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("Input",
             "(vector<LoDTensor>) The input tensors of mixallgather_tensor "
             "operator.")
        .AsDuplicable();
    AddOutput("Output",
              "(LoDTensor) The output tensor "
              "of mixallgather_tensor operator. And the tensors of"
              " Output is sliced from the tensor of FusedOutput.");
    AddAttr<int>("rankid", "(int default 0) communication node id.")
        .SetDefault(0);
    AddAttr<int>("nranks", "(int default 1) communication node num.")
        .SetDefault(1);
    AddAttr<int>("nccl_mode",
                 "(int default 0) one node 0 allreduce, 1 mixallgather mode , "
                 "2 allgather mode.")
        .SetDefault(0);
    AddAttr<int>("ring_id", "(int default -1) nccl ring id num.")
        .SetDefault(-1);
    AddAttr<bool>(
        "use_calc_stream",
        "(bool default false) eject CUDA operations to calculation stream.")
        .SetDefault(true);
    AddComment(string::Sprintf(R"DOC(
MixAllGather %s Operator

Call collective MixAllGather with reduce type %s. If input and output are
the same variable, in-place allreduce will be used.
Reference: https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/usage/operations.html#allreduce
)DOC",
                               GetName(), GetName()));
  }

 protected:
  virtual std::string GetName() { return "MixAllGather"; }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(c_mixallgather, ops::CMixAllGatherOp,
                  ops::CMixAllGatherOpMaker);
REGISTER_OP_CPU_KERNEL(c_mixallgather, ops::CMixAllGatherOpCPUKernel<float>,
                       ops::CMixAllGatherOpCPUKernel<double>,
                       ops::CMixAllGatherOpCPUKernel<int>,
                       ops::CMixAllGatherOpCPUKernel<int64_t>,
                       ops::CMixAllGatherOpCPUKernel<plat::float16>);
#ifdef PADDLE_WITH_CUDA
REGISTER_OP_CUDA_KERNEL(c_mixallgather, ops::CMixAllGatherOpCUDAKernel<float>,
                        ops::CMixAllGatherOpCUDAKernel<double>,
                        ops::CMixAllGatherOpCUDAKernel<int>,
                        ops::CMixAllGatherOpCUDAKernel<int64_t>,
                        ops::CMixAllGatherOpCUDAKernel<plat::float16>);
#endif
