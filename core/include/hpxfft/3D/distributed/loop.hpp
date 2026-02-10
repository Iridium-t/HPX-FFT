#pragma once
#ifndef hpxfft_distributed_loop_3D_H_INCLUDED
#define hpxfft_distributed_loop_3D_H_INCLUDED

#include "../../util/vector_3d.hpp"  // for hpxfft::util::vector_3d
#include "distributed_base.hpp"
#include <hpx/parallel/algorithms/for_loop.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/collectives.hpp>

typedef double real;

namespace hpxfft::fft3D::distributed::loop
{
using vector_3d = hpxfft::util::vector_3d<real>;

struct slap : public base
{
  public:
    slap() = default;

    void initialize(vector_3d values_vec, const std::string COMM_FLAG, const std::string PLAN_FLAG);

    vector_3d fft_3d_r2c();

  private:
    //permute (only local data)
    void permute_distributed_x_z_y(const std::size_t slice_x, const std::size_t i) override;
    void permute_distributed_z_y_x(const std::size_t slice_y, const std::size_t i) override;
    void permute_distributed_z_x_y(const std::size_t slice_x, const std::size_t i) override;

    // split data for communication
    void split_vec(const std::size_t i, const std::size_t j);
    void split_permuted_vec(const std::size_t i, const std::size_t j);

    // scatter communication
    void communicate_scatter_vec(const std::size_t i);
    void communicate_scatter_permuted_vec(const std::size_t i);

    // all to all communication
    void communicate_all_to_all_vec();
    void communicate_all_to_all_permuted_vec();
};

/*
struct pencil : public base
{
  public:
    pencil() = default;

    void initialize(vector_3d values_vec, const std::string COMM_FLAG, const std::string PLAN_FLAG);

    vector_3d fft_3d_r2c();

  private:
    //permute (only local data)
    void permute_distributed_x_z_y(const std::size_t slice_x) override;
    void permute_distributed_z_y_x(const std::size_t slice_y) override;
    void permute_distributed_z_x_y(const std::size_t slice_x) override;

    // split data for communication
    void split_vec(const std::size_t i, const std::size_t j);
    void split_permuted_vec(const std::size_t i, const std::size_t j);

    // scatter communication
    void communicate_scatter_vec(const std::size_t i);
    void communicate_scatter_permuted_vec(const std::size_t i);

    // all to all communication
    void communicate_all_to_all_vec();
    void communicate_all_to_all_permuted_vec();
};
*/

}  // namespace hpxfft::fft3D::distributed::loop
#endif  // hpxfft_distributed_loop_3D_H_INCLUDED
