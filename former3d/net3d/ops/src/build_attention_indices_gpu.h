
#ifndef BUILD_ATTENTION_INDICES_GPU_H
#define BUILD_ATTENTION_INDICES_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int subm_local_attention_with_tensor_wrapper(int x_max, int y_max, int z_max, int num_voxels, int attend_size, int attend_range,
                                                at::Tensor attend_indices_tensor, at::Tensor v_indices_tensor, at::Tensor xyz_to_vidx_tensor);

int subm_local_attention_with_hash_wrapper(int x_max, int y_max, int z_max, int num_voxels, int attend_size, int attend_range, int hash_size,
                                                at::Tensor attend_indices_tensor, at::Tensor v_indices_tensor, at::Tensor xyz_to_vidx_tensor);

int subm_strided_attention_with_tensor_wrapper(int x_max, int y_max, int z_max, int num_voxels, int attend_size, int num_range,
                                                at::Tensor attend_indices_tensor, at::Tensor v_indices_tensor,
                                                at::Tensor xyz_to_vidx_tensor, at::Tensor range_spec_tensor);

int subm_strided_attention_with_hash_wrapper(int x_max, int y_max, int z_max, int num_voxels, int attend_size, int num_range, int hash_size,
                                                at::Tensor attend_indices_tensor, at::Tensor v_indices_tensor,
                                                at::Tensor xyz_to_vidx_tensor, at::Tensor range_spec_tensor);

int subm_strided_attention_with_hash_sort_wrapper(int x_max, int y_max, int z_max, int num_voxels, int attend_size, int num_range, int hash_size,
                                                int search1_num, int search2_num, int search3_num, int search4_num, 
                                                at::Tensor attend_indices_tensor, at::Tensor v_indices_tensor,
                                                at::Tensor xyz_to_vidx_tensor, at::Tensor search1_tensor, 
                                                at::Tensor search2_tensor, at::Tensor search3_tensor, at::Tensor search4_tensor);

void subm_local_attention_with_tensor_kernel_launcher(int x_max, int y_max, int z_max, int num_voxels, int attend_size, int attend_range,
                                                        int *attend_indices, const int *v_indices, const int *xyz_to_vidx);

void subm_local_attention_with_hash_kernel_launcher(int x_max, int y_max, int z_max, int num_voxels, int attend_size, int attend_range, int hash_size,
                                                        int *attend_indices, const int *v_indices, const int *xyz_to_vidx);

void subm_strided_attention_with_tensor_kernel_launcher(int x_max, int y_max, int z_max, int num_voxels, int attend_size, int num_range,
                                                            int *attend_indices, const int *v_indices, const int *xyz_to_vidx, const int *range_spec);

void subm_strided_attention_with_hash_kernel_launcher(int x_max, int y_max, int z_max, int num_voxels, int attend_size, int num_range, int hash_size,
                                                            int *attend_indices, const int *v_indices, const int *xyz_to_vidx, const int *range_spec);

void subm_strided_attention_with_hash_sort_kernel_launcher(int x_max, int y_max, int z_max, int num_voxels, int attend_size, int num_range, int hash_size,
                                                            int search1_num, int search2_num, int search3_num, int search4_num, 
                                                            int *attend_indices, const int *v_indices, const int *xyz_to_vidx, const int *search1,
                                                            const int *search2, const int *search3, const int *search4);            

int sparse_local_attention_with_tensor_wrapper(int x_max, int y_max, int z_max, int x_stride, int y_stride, int z_stride,
                                                int num_voxels, int attend_size, int attend_range,
                                                at::Tensor attend_indices_tensor, at::Tensor v_indices_tensor, at::Tensor xyz_to_vidx_tensor);

int sparse_local_attention_with_hash_wrapper(int x_max, int y_max, int z_max, int x_stride, int y_stride, int z_stride,
                                                int num_voxels, int attend_size, int attend_range, int hash_size,
                                                at::Tensor attend_indices_tensor, at::Tensor v_indices_tensor, at::Tensor xyz_to_vidx_tensor);

int sparse_strided_attention_with_tensor_wrapper(int x_max, int y_max, int z_max, int x_stride, int y_stride, int z_stride,
                                                    int num_voxels, int attend_size, int num_range,
                                                    at::Tensor attend_indices_tensor, at::Tensor v_indices_tensor,
                                                    at::Tensor xyz_to_vidx_tensor, at::Tensor range_spec_tensor);

int sparse_strided_attention_with_hash_wrapper(int x_max, int y_max, int z_max, int x_stride, int y_stride, int z_stride,
                                                    int num_voxels, int attend_size, int num_range, int hash_size,
                                                    at::Tensor attend_indices_tensor, at::Tensor v_indices_tensor,
                                                    at::Tensor xyz_to_vidx_tensor, at::Tensor range_spec_tensor);

void sparse_local_attention_with_tensor_kernel_launcher(int x_max, int y_max, int z_max, int x_stride, int y_stride, int z_stride,
                                                        int num_voxels, int attend_size, int attend_range,
                                                        int *attend_indices, const int *v_indices, const int *xyz_to_vidx);

void sparse_local_attention_with_hash_kernel_launcher(int x_max, int y_max, int z_max, int x_stride, int y_stride, int z_stride,
                                                        int num_voxels, int attend_size, int attend_range, int hash_size,
                                                        int *attend_indices, const int *v_indices, const int *xyz_to_vidx);

void sparse_strided_attention_with_tensor_kernel_launcher(int x_max, int y_max, int z_max, int x_stride, int y_stride, int z_stride,
                                                            int num_voxels, int attend_size, int num_range,
                                                            int *attend_indices, const int *v_indices, const int *xyz_to_vidx, const int *range_spec);

void sparse_strided_attention_with_hash_kernel_launcher(int x_max, int y_max, int z_max, int x_stride, int y_stride, int z_stride,
                                                            int num_voxels, int attend_size, int num_range, int hash_size,
                                                            int *attend_indices, const int *v_indices, const int *xyz_to_vidx, const int *range_spec);
#endif