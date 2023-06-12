#pragma once

#include <cmath>
#include <iostream>
#include <vector>

#include "HornerScheme.hpp"
#include "Pretabulated.hpp"
#include "Utilities.hpp"

namespace mcmurchie_davidson {
auto boys_function(size_t order, const std::vector<double>& xs) -> std::vector<double>;

namespace cpu {
template <auto order>
auto
boys_function(const std::vector<double>& xs) -> std::vector<double>
{
    static_assert(order > 0, "Order should be greater than 0!");

    constexpr auto ncols = order + 1;

    constexpr auto table = pretabulated<order>();

    auto n_xs = xs.size();

    auto ys = std::vector<double>(n_xs * ncols);

    const double SQRT_M_PI = std::sqrt(M_PI);

    constexpr auto odds = odd_numbers<order + 1>();

    for (auto i = 0; i < n_xs; ++i)
    {
        auto offset = i * ncols;
        auto x      = xs[i];
        auto sqrt_x = std::sqrt(x);
        auto exp_mx = std::exp(-x);

        if (is_zero(x))
        {
            // compute analytically at the origin
            detail::constexpr_for<0, order + 1, 1>([&ys, &odds, offset](auto o) { ys[o + offset] = 1.0 / odds[o]; });
        }
        else
        {
            // zero-th order Boys' function
            ys[0 + offset] = SQRT_M_PI * std::erf(sqrt_x) / (2 * sqrt_x);

            auto p = grid_point(x);

            if (p < 121)
            {
                auto w = x - 0.1 * p;
                auto y = horner(w, table[p]);

                ys[order + offset] = y;

                // compile-time unrolled downward recursion
                detail::constexpr_for<order - 1, 0, -1>(
                    [&ys, &odds, x, exp_mx, offset](auto o) { ys[o + offset] = std::fma(2 * x, ys[(o + 1) + offset], exp_mx) / odds[o]; });
            }
            else
            {
                auto x_m1 = 1.0 / x;
                // compile-time unrolled upward recursion
                detail::constexpr_for<0, order, 1>(
                    [&ys, &odds, x_m1, exp_mx, offset](auto o) { ys[(o + 1) + offset] = 0.5 * x_m1 * std::fma(odds[o], ys[o + offset], -exp_mx); });
            }
        }
    }

    return ys;
}
}  // namespace cpu
}  // namespace mcmurchie_davidson
