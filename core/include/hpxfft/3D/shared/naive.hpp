#pragma once
#ifndef hpxfft_shared_naive_3D_H_INCLUDED
#define hpxfft_shared_naive_3D_H_INCLUDED

#include "shared_base.hpp"
#include "../../util/vector_3d.hpp"             // for hpxfft::util::vector_3d
#include <hpx/timing/high_resolution_timer.hpp> // for hpx::chrono::high_resolution_timer
#include <hpx/future.hpp>

typedef double real;

namespace hpxfft::fft3D::shared
{
using vector_3d = hpxfft::util::vector_3d<real>;

struct naive : public base
{
    typedef std::vector<hpx::future<void>> vector_future;

  public:
    naive() = default;

    void initialize(vector_3d values_vec, const std::string PLAN_FLAG);

    vector_3d fft_3d_r2c();

    void write_plans_to_file(std::string file_path);

  private:
    // future vectors
    vector_future fft_z_r2c_futures_;
    vector_future permute_first_futures_;
    vector_future fft_y_c2c_futures_;
    vector_future permute_second_futures_;
    vector_future fft_x_c2c_futures_;
    vector_future permute_third_futures_;
};
} // namespace hpxfft::fft3D::shared
#endif  // hpxfft_shared_naive_3D_H_INCLUDED
