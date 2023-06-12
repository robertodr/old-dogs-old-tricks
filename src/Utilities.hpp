#pragma once

#include <array>
#include <cmath>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

// FIXME
#define __host__
#define __device__

/** Numpy-style check for floating-point equality.
 *
 * @tparam T type of the arguments.
 * @param a left operand.
 * @param b right operand.
 * @param rtol relative tolerance.
 * @param atol absolute tolerance.
 *
 * See: https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
 */
__host__ __device__ template <typename T>
inline auto
all_close(T a, T b, T rtol, T atol) -> bool
{
    static_assert(std::is_floating_point_v<T>, "all_close is valid for floating-point numbers");
    return (std::abs(a - b) <= atol + rtol * std::abs(b));
}

/** Whether a floating-point number is zero.
 *
 * @tparam T type of the arguments.
 * @param x number to be checked.
 * @param atol absolute tolerance.
 */
__host__ __device__ template <typename T>
inline auto
is_zero(T x, T atol = std::numeric_limits<T>::epsilon()) -> bool
{
    static_assert(std::is_floating_point_v<T>, "is_zero is valid for floating-point numbers");
    return all_close(x, T{0.0}, 0.0, atol);
}

// FIXME device function as well
__host__ inline auto
boys_function_0(const std::vector<double>& xs) -> std::vector<double>
{
    auto n_xs = xs.size();
    auto ys   = std::vector<double>(n_xs, 0.0);

    const double SQRT_M_PI = std::sqrt(M_PI);

    for (auto i = 0; i < n_xs; ++i)
    {
        auto x = xs[i];
        if (is_zero(x))
        {
            ys[i] = 1.0;
        }
        else
        {
            auto sqrt_x = std::sqrt(x);
            ys[i]       = SQRT_M_PI * std::erf(sqrt_x) / (2 * sqrt_x);
        }
    }

    return ys;
}

namespace detail {
/** Compile-time loop-application of a function.
 *
 * @tparam Start loop start index.
 * @tparam End loop end index.
 * @tparam Increment loop increment.
 * @tparam F function to apply.
 *
 * @param[in] F function to apply with signature
 *
 * auto F(auto i) -> void;
 *
 * with i the loop index.
 */
template <auto Start, auto End, auto Increment, typename F>
__host__ __device__ constexpr auto
constexpr_for(F&& f) -> void
{
    constexpr auto cond = (Increment > 0) ? (Start < End) : (Start > End);

    if constexpr (cond)
    {
        f(std::integral_constant<decltype(Start), Start>());
        constexpr_for<Start + Increment, End, Increment>(f);
    }
}
/** Compile-time fill an array of N elements with results of a function of the
 * index.
 *
 * @tparam T output scalar type
 * @tparam N size of the array
 * @tparam Generator function to apply on each index. Signature: T op(std::size_t)
 * @tparam Is indices
 * @param op generator function
 * @param index sequence
 */
template <size_t N, typename Callable, size_t... Is>
__host__ __device__ constexpr auto
fill_array_impl(Callable op, std::index_sequence<Is...>) -> std::array<std::result_of_t<Callable(size_t)>, N>
{
    return {{op(Is)...}};
}
}  // namespace detail

/** Compile-time fill an array of N elements with results of a function of the
 * index.
 *
 * @tparam N size of the array
 * @tparam Callable function to apply on each index.
 * Signature: T op(std::size_t)
 * @tparam Is indices
 * @param op generator function
 */
template <size_t N, typename Callable, typename Is = std::make_index_sequence<N>>
__host__ __device__ constexpr auto
fill_array(Callable op) -> std::array<std::result_of_t<Callable(size_t)>, N>
{
    return detail::fill_array_impl<N>(op, Is{});
}

/** Compile-time array of the N first inverse odd numbers.
 *
 * @tparam T output scalar type
 * @tparam N size of the array
 */
template <size_t N>
__host__ __device__ constexpr auto
odd_numbers() -> std::array<size_t, N>
{
    return fill_array<N>([](auto o) { return (2 * o + 1); });
}

// FIXME on the device it should be fma, not std::fma!

namespace detail {
template <typename T, typename, T Begin, bool Increasing>
struct integer_range_impl;

template <typename T, T... Ns, T Begin>
struct integer_range_impl<T, std::integer_sequence<T, Ns...>, Begin, true>
{
    using type = std::integer_sequence<T, Ns + Begin...>;
};

template <typename T, T... Ns, T Begin>
struct integer_range_impl<T, std::integer_sequence<T, Ns...>, Begin, false>
{
    using type = std::integer_sequence<T, Begin - Ns...>;
};
}  // namespace detail

template <typename T, T Begin, T End>
using make_integer_range =
    typename detail::integer_range_impl<T, std::make_integer_sequence<T, (Begin < End) ? End - Begin : Begin - End>, Begin, (Begin < End)>::type;

template <std::size_t Begin, std::size_t End>
using make_index_range = make_integer_range<std::size_t, Begin, End>;
