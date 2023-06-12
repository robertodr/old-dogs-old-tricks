#pragma once

#include <cmath>
#include <vector>

#include "Approximants.hpp"
#include "Utilities.hpp"

// FIXME
#define __host__
#define __device__

namespace pade {
namespace cpu {
template <auto M, auto N, auto order>
auto
boys_function(const std::vector<double>& xs) -> std::vector<double>
{
    static_assert(order > 0, "Order should be greater than 0!");

    constexpr auto ncols = order + 1;

    auto n_xs = xs.size();

    auto ys = std::vector<double>(n_xs * ncols);

    const double SQRT_M_PI = std::sqrt(M_PI);

    constexpr auto odds = odd_numbers<order + 1>();

    for (auto i = 0; i < n_xs; ++i)
    {
        auto offset = i * ncols;
        auto x      = xs[i];
        auto sqrt_x = std::sqrt(x);

        if (is_zero(x))
        {
            // compute analytically at the origin
            detail::constexpr_for<0, order + 1, 1>([&ys, &odds, offset](auto o) { ys[o + offset] = 1.0 / odds[o]; });
        }
        else
        {
            // zero-th order Boys' function
            ys[0 + offset] = SQRT_M_PI * std::erf(sqrt_x) / (2 * sqrt_x);

            if (x <= 12.0)
            {
                // [M/N] Padé approximant for all orders
                compute_approximants<M, N>(make_index_range<0, order>{}, ys.data(), x, offset);
            }
            else
            {
                auto x_m1   = 1.0 / x;
                auto exp_mx = std::exp(-x);
                // compile-time unrolled upward recursion
                detail::constexpr_for<0, order, 1>(
                    [&ys, &odds, x_m1, exp_mx, offset](auto o) { ys[(o + 1) + offset] = 0.5 * x_m1 * std::fma(odds[o], ys[o + offset], -exp_mx); });
            }
        }
    }

    return ys;
}
}  // namespace cpu

auto boys_function_5_6(size_t order, const std::vector<double>& xs) -> std::vector<double>;

auto boys_function_9_10(size_t order, const std::vector<double>& xs) -> std::vector<double>;

auto boys_function_15_16(size_t order, const std::vector<double>& xs) -> std::vector<double>;

auto boys_function_25_26(size_t order, const std::vector<double>& xs) -> std::vector<double>;
}  // namespace pade
