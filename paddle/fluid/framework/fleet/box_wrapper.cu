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

#ifdef PADDLE_WITH_BOX_PS
#include <algorithm>
#include <ctime>
#include <memory>
#include <numeric>
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace framework {
#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

template <size_t EMBEDX_DIM, size_t EXPAND_EMBED_DIM>
__global__ void PullCopy(
    float** dest,
    const boxps::FeatureValueGpu<EMBEDX_DIM, EXPAND_EMBED_DIM>* src,
    const int64_t* len, int hidden, int expand_dim, int slot_num, int total_len,
    uint64_t** keys) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int low = 0;
    int high = slot_num - 1;
    while (low < high) {
      int mid = (low + high) / 2;
      if (i < len[mid])
        high = mid;
      else
        low = mid + 1;
    }
    int x = low;
    int y = i - (x ? len[x - 1] : 0);
    if (*(keys[x] + y) == 0) {
      *(dest[x] + y * hidden) = 0;
      *(dest[x] + y * hidden + 1) = 0;
      *(dest[x] + y * hidden + 2) = 0;
    } else {
      *(dest[x] + y * hidden) = (src + i)->show;
      *(dest[x] + y * hidden + 1) = (src + i)->clk;
      *(dest[x] + y * hidden + 2) = (src + i)->embed_w;
    }
    if ((src + i)->embedding_size == 0 || *(keys[x] + y) == 0) {
      for (int j = 0; j < hidden - 3; j++) {
        *(dest[x] + y * hidden + 3 + j) = 0;
      }
    } else {
      for (int j = 0; j < hidden - 3; j++) {
        *(dest[x] + y * hidden + 3 + j) = (src + i)->embedx[1 + j];
      }
    }
    // process embed_expand
    if (expand_dim > 0) {
      int z = x + slot_num;
      if ((src + i)->embed_expand_size[0] == 0 || *(keys[x] + y) == 0) {
        for (int j = 0; j < expand_dim; j++) {
          *(dest[z] + y * expand_dim + j) = 0;
        }
      } else {
        for (int j = 0; j < expand_dim; j++) {
          *(dest[z] + y * expand_dim + j) = (src + i)->embed_expand[1 + j];
        }
      }
    }
  }  // end kernel loop
}

__global__ void CopyKeysKernel(uint64_t** src_keys, uint64_t* dest_total_keys,
                               const int64_t* len, int slot_num,
                               int total_len) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int low = 0;
    int high = slot_num - 1;
    while (low < high) {
      int mid = (low + high) / 2;
      if (i < len[mid])
        high = mid;
      else
        low = mid + 1;
    }
    int x = low;
    int y = i - (x ? len[x - 1] : 0);
    dest_total_keys[i] = src_keys[x][y];
  }
}

template <size_t EMBEDX_DIM, size_t EXPAND_EMBED_DIM>
__global__ void PushCopy(
    boxps::FeaturePushValueGpu<EMBEDX_DIM, EXPAND_EMBED_DIM>* dest, float** src,
    int64_t* len, int hidden, int expand_dim, int slot_num, int total_len,
    int bs, int* slot_vector) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int low = 0;
    int high = slot_num - 1;
    while (low < high) {
      int mid = (low + high) / 2;
      if (i < len[mid])
        high = mid;
      else
        low = mid + 1;
    }
    int x = low;
    int y = i - (x ? len[low - 1] : 0);
    (dest + i)->slot = slot_vector[x];
    (dest + i)->show = *(src[x] + y * hidden);
    (dest + i)->clk = *(src[x] + y * hidden + 1);
    (dest + i)->embed_g = *(src[x] + y * hidden + 2) * -1. * bs;
    for (int j = 0; j < hidden - 3; j++) {
      (dest + i)->embedx_g[j] = *(src[x] + y * hidden + 3 + j) * -1. * bs;
    }
    if (expand_dim > 0) {
      int z = x + slot_num;
      for (int j = 0; j < expand_dim; j++) {
        (dest + i)->embed_expand_g[j] =
            *(src[z] + y * expand_dim + j) * -1. * bs;
      }
    }
  }
}

__global__ void AddBasicCalculator(const float* pred, const int64_t* label,
                                   double* positive, double* negative,
                                   double* abs_error, double* sqr_error,
                                   double* local_pred, int len,
                                   int table_size) {
  CUDA_KERNEL_LOOP(ins_idx, len) {
    int pos = static_cast<int>(pred[ins_idx] * table_size);
    if (pos >= table_size) {
      pos = table_size - 1;
    }
    if (label[ins_idx] == 0) {
      atomicAdd(negative + pos, 1.0);
      // negative[pos]++;
    } else {
      atomicAdd(positive + pos, 1.0);
      // positive[pos]++;
    }
    double err = pred[ins_idx] - label[ins_idx];
    abs_error[ins_idx] += fabs(err);
    sqr_error[ins_idx] += err * err;
    local_pred[ins_idx] += pred[ins_idx];
  }
}

void BoxWrapper::CopyForPull(const paddle::platform::Place& place,
                             uint64_t** gpu_keys,
                             const std::vector<float*>& values,
                             void* total_values_gpu, const int64_t* gpu_len,
                             const int slot_num, const int hidden_size,
                             const int expand_embed_dim,
                             const int64_t total_length) {
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(
                        BOOST_GET_CONST(platform::CUDAPlace, place)))
                    ->stream();
  auto buf_value = memory::AllocShared(place, values.size() * sizeof(float*));
  float** gpu_values = reinterpret_cast<float**>(buf_value->ptr());
  cudaMemcpy(gpu_values, values.data(), values.size() * sizeof(float*),
             cudaMemcpyHostToDevice);
#define EMBEDX_CASE(i, ...)                                                  \
  case i: {                                                                  \
    constexpr size_t EmbedxDim = i;                                          \
    switch (expand_embed_dim) {                                              \
      __VA_ARGS__                                                            \
      default:                                                               \
        PADDLE_THROW(platform::errors::InvalidArgument(                      \
            "Unsupport this expand embedding size [%d]", expand_embed_dim)); \
    }                                                                        \
  } break

#define EXPAND_EMBED_PULL_CASE(i, ...)                                       \
  case i: {                                                                  \
    constexpr size_t ExpandDim = i;                                          \
    PullCopy<EmbedxDim,                                                      \
             ExpandDim><<<(total_length + 512 - 1) / 512, 512, 0, stream>>>( \
        gpu_values,                                                          \
        reinterpret_cast<boxps::FeatureValueGpu<EmbedxDim, ExpandDim>*>(     \
            total_values_gpu),                                               \
        gpu_len, hidden_size, expand_embed_dim, slot_num, total_length,      \
        gpu_keys);                                                           \
  } break

  switch (hidden_size - 3) {
    EMBEDX_CASE(8, EXPAND_EMBED_PULL_CASE(0); EXPAND_EMBED_PULL_CASE(8);
                EXPAND_EMBED_PULL_CASE(64););
    EMBEDX_CASE(16, EXPAND_EMBED_PULL_CASE(0););
    EMBEDX_CASE(256, EXPAND_EMBED_PULL_CASE(0););
    EMBEDX_CASE(128, EXPAND_EMBED_PULL_CASE(0););
    default:
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unsupport this embedding size [%d]", hidden_size - 3));
  }
  cudaStreamSynchronize(stream);
#undef EXPAND_EMBED_PULL_CASE
#undef EMBEDX_CASE
}

void BoxWrapper::CopyKeys(const paddle::platform::Place& place,
                          uint64_t** origin_keys, uint64_t* total_keys,
                          const int64_t* gpu_len, int slot_num, int total_len) {
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(
                        BOOST_GET_CONST(platform::CUDAPlace, place)))
                    ->stream();
  CopyKeysKernel<<<(total_len + 512 - 1) / 512, 512, 0, stream>>>(
      origin_keys, total_keys, gpu_len, slot_num, total_len);
  cudaStreamSynchronize(stream);
}

void BoxWrapper::CopyForPush(const paddle::platform::Place& place,
                             const std::vector<const float*>& grad_values,
                             void* total_grad_values_gpu,
                             const std::vector<int64_t>& slot_lengths,
                             const int hidden_size, const int expand_embed_dim,
                             const int64_t total_length, const int batch_size) {
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(
                        BOOST_GET_CONST(platform::CUDAPlace, place)))
                    ->stream();
  auto slot_lengths_lod = slot_lengths;
  for (int i = 1; i < slot_lengths_lod.size(); i++) {
    slot_lengths_lod[i] += slot_lengths_lod[i - 1];
  }
  auto buf_grad_value =
      memory::AllocShared(place, grad_values.size() * sizeof(float*));
  auto buf_length =
      memory::AllocShared(place, slot_lengths.size() * sizeof(int64_t));
  auto buf_slot_vector =
      memory::AllocShared(place, slot_lengths_lod.size() * sizeof(int));

  float** gpu_values = reinterpret_cast<float**>(buf_grad_value->ptr());
  int64_t* gpu_len = reinterpret_cast<int64_t*>(buf_length->ptr());
  int* d_slot_vector = reinterpret_cast<int*>(buf_slot_vector->ptr());

  cudaMemcpy(gpu_values, grad_values.data(),
             grad_values.size() * sizeof(float*), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_len, slot_lengths_lod.data(),
             slot_lengths.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_slot_vector, slot_vector_.data(),
             slot_lengths_lod.size() * sizeof(int), cudaMemcpyHostToDevice);

#define EMBEDX_CASE(i, ...)                                                  \
  case i: {                                                                  \
    constexpr size_t EmbedxDim = i;                                          \
    switch (expand_embed_dim) {                                              \
      __VA_ARGS__                                                            \
      default:                                                               \
        PADDLE_THROW(platform::errors::InvalidArgument(                      \
            "Unsupport this expand embedding size [%d]", expand_embed_dim)); \
    }                                                                        \
  } break

#define EXPAND_EMBED_PUSH_CASE(i, ...)                                       \
  case i: {                                                                  \
    constexpr size_t ExpandDim = i;                                          \
    PushCopy<EmbedxDim,                                                      \
             ExpandDim><<<(total_length + 512 - 1) / 512, 512, 0, stream>>>( \
        reinterpret_cast<boxps::FeaturePushValueGpu<EmbedxDim, ExpandDim>*>( \
            total_grad_values_gpu),                                          \
        gpu_values, gpu_len, hidden_size, expand_embed_dim,                  \
        slot_lengths.size(), total_length, batch_size, d_slot_vector);       \
  } break

  switch (hidden_size - 3) {
    EMBEDX_CASE(8, EXPAND_EMBED_PUSH_CASE(0); EXPAND_EMBED_PUSH_CASE(8);
                EXPAND_EMBED_PUSH_CASE(64););
    EMBEDX_CASE(16, EXPAND_EMBED_PUSH_CASE(0););
    EMBEDX_CASE(256, EXPAND_EMBED_PUSH_CASE(0););
    EMBEDX_CASE(128, EXPAND_EMBED_PUSH_CASE(0););
    default:
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unsupport this embedding size [%d]", hidden_size - 3));
  }

  cudaStreamSynchronize(stream);
#undef EXPAND_EMBED_PUSH_CASE
#undef EMBEDX_CASE
}

void BasicAucCalculator::cuda_add_data(const paddle::platform::Place& place,
                                       const int64_t* label, const float* pred,
                                       int len) {

  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(
                        BOOST_GET_CONST(platform::CUDAPlace, place)))
                    ->stream();

  int i = BOOST_GET_CONST(platform::CUDAPlace, place).GetDeviceId();

  cudaSetDevice(i);

  AddBasicCalculator<<<(len + 512 - 1) / 512, 512, 0, stream>>>(
      pred, label, reinterpret_cast<double*>(_d_positive[i]->ptr()),
      reinterpret_cast<double*>(_d_negative[i]->ptr()),
      reinterpret_cast<double*>(_d_abserr[i]->ptr()),
      reinterpret_cast<double*>(_d_sqrerr[i]->ptr()),
      reinterpret_cast<double*>(_d_pred[i]->ptr()), len, _table_size);
}

__global__
void pull_query_emb_kernel(int len, int dim, uint64_t* key, float* val, float* table) {
    CUDA_KERNEL_LOOP(i, len) {
        val[i] = table[key[i / dim] * dim + i % dim];
    }
}

void QueryEmbSet::PullQueryEmb(uint64_t* d_keys, float* d_vals, int num, int gpu_id) {
  auto place = platform::CUDAPlace(gpu_id);
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(place))
                    ->stream();
  int len = emb_dim * num;
  const int BLOCK_SIZE_ = 256;
  pull_query_emb_kernel<<<(len + BLOCK_SIZE_ - 1) / BLOCK_SIZE_, BLOCK_SIZE_, 0, stream>>>(len, emb_dim, d_keys, d_vals, d_embs[gpu_id]);
  //std::vector<float> h;
  //h.resize(128);
  
  //h_emb_mtx.lock();
  //std::cout << "val:";
  //cudaMemcpyAsync(h.data(), d_vals + 4 * 128, sizeof(float) * 128, cudaMemcpyDeviceToHost, stream);
  //cudaStreamSynchronize(stream);
  //for (int i = 0; i < 128; ++i) {
  //  std::cout << " " << h[i];
  //}
  //std::cout << std::endl;
  //h_emb_mtx.unlock();
}

}  // end namespace framework
}  // end namespace paddle
#endif
