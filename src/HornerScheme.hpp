#pragma once

#include <array>
#include <cstdint>
#include <type_traits>
#include <utility>

// TODO
//#include "Device.hpp"
#define __host__
#define __device__

/** Base case of Horner's scheme compile-time recursion (variadic
 * implementation). */
template <typename X, typename T>
__host__ __device__ constexpr X
horner(X /* x */, T v)
{
    return static_cast<X>(v);
}

/**
 * Horner's scheme for evaluation of 1D polynomial (variadic implementation)
 *
 * @f{align}
 *   a_0 &+ a_1x + a_2x^2 + a_3x^3 + \cdots + a_nx^n \\
 *   &= a_0 + x \bigg(a_1 + x \Big(a_2 + x \big(a_3 + \cdots + x(a_{n-1} + x \,
 * a_n) \cdots \big) \Big) \bigg)
 * @f}
 *
 * @tparam X Scalar type of output
 * @tparam T Scalar type of first coefficient
 * @tparam Args Types of all other coefficients
 * @param[in] x evaluation point
 * @param[in] c0 zeroth order polynomial coefficient
 * @param[in] cs polynomial coefficients, in *reverse lexicographical order*
 * (lowest to highest degree)
 * @return value of polynomial at point
 * @note we use fused-multiply-add (FMA) instead of `static_cast<X>(c0) + x * horner(x, cs...);`
 */
template <typename X, typename T, typename... Args>
__host__ constexpr X
horner(X x, T c0, Args... cs)
{
    return std::fma(x, horner(x, cs...), static_cast<X>(c0));
}

/**
 * Horner's scheme for evaluation of 1D polynomial (variadic implementation)
 *
 * @f{align}
 *   a_0 &+ a_1x + a_2x^2 + a_3x^3 + \cdots + a_nx^n \\
 *   &= a_0 + x \bigg(a_1 + x \Big(a_2 + x \big(a_3 + \cdots + x(a_{n-1} + x \,
 * a_n) \cdots \big) \Big) \bigg)
 * @f}
 *
 * @tparam X Scalar type of output
 * @tparam T Scalar type of first coefficient
 * @tparam Args Types of all other coefficients
 * @param[in] x evaluation point
 * @param[in] c0 zeroth order polynomial coefficient
 * @param[in] cs polynomial coefficients, in *reverse lexicographical order*
 * (lowest to highest degree)
 * @return value of polynomial at point
 * @note we use fused-multiply-add (FMA) instead of `static_cast<X>(c0) + x * horner(x, cs...);`
 */
#ifdef USE_DEVICE
template <typename X, typename T, typename... Args>
__device__ constexpr X
horner(X x, T c0, Args... cs)
{
    return fma(x, horner(x, cs...), static_cast<X>(c0));
}
#endif

namespace detail {
/** Implementation of Horner's scheme with an array of coefficients.
 *
 * @note This uses the variadic implementation internally.
 */
template <typename T, std::size_t N, std::size_t... Is>
__host__ __device__ constexpr T
horner_impl(T x, const std::array<T, N> &coefs, std::index_sequence<Is...>)
{
    return horner(x, coefs[Is]...);
}
}  // namespace detail

/**
 * Horner's scheme for evaluation of 1D polynomial.
 *
 * @f{align}
 *   a_0 &+ a_1x + a_2x^2 + a_3x^3 + \cdots + a_nx^n \\
 *   &= a_0 + x \bigg(a_1 + x \Big(a_2 + x \big(a_3 + \cdots + x(a_{n-1} + x \,
 * a_n) \cdots \big) \Big) \bigg)
 * @f}
 *
 * @tparam T Scalar type of coefficients
 * @tparam N Number of coefficients
 * @tparam Is sequence of indices in the coefficients array
 * @param[in] x evaluation point
 * @param[in] c polynomial coefficients, in *reverse lexicographical order*
 * (lowest to highest degree)
 * @return value of polynomial at point
 */
template <typename T, std::size_t N, typename Is = std::make_index_sequence<N>>
__host__ __device__ constexpr T
horner(T x, const std::array<T, N> &c)
{
    return detail::horner_impl(x, c, Is{});
}
