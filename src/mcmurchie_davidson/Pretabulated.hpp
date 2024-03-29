/* This file was autogenerated on 2023-06-11T16:57 DO NOT EDIT! */

#pragma once

#include <array>

namespace mcmurchie_davidson {
/**
 * Obtain index of point within pretabulated grid.
 *
 * @param[in] x the point
 * @return The index of the point in the pretabulated grid.
 */
inline constexpr auto
grid_point(double x) -> size_t
{
    return (x > 1.0e5) ? static_cast<size_t>(1.0e6) : static_cast<size_t>(10.0 * x + 0.5);
}

/**
 * Pretabulated values of given order.
 *
 * @tparam order Order of the table to fetch.
 * @return table of values.
 */
template <auto order>
inline constexpr std::array<std::array<double, 7>, 121> pretabulated();
}  // namespace mcmurchie_davidson

#include "tables/BFunc_01.hpp"
#include "tables/BFunc_02.hpp"
#include "tables/BFunc_03.hpp"
#include "tables/BFunc_04.hpp"
#include "tables/BFunc_05.hpp"
#include "tables/BFunc_06.hpp"
#include "tables/BFunc_07.hpp"
#include "tables/BFunc_08.hpp"
#include "tables/BFunc_09.hpp"
#include "tables/BFunc_10.hpp"
#include "tables/BFunc_11.hpp"
#include "tables/BFunc_12.hpp"
#include "tables/BFunc_13.hpp"
#include "tables/BFunc_14.hpp"
#include "tables/BFunc_15.hpp"
#include "tables/BFunc_16.hpp"
#include "tables/BFunc_17.hpp"
#include "tables/BFunc_18.hpp"
#include "tables/BFunc_19.hpp"
#include "tables/BFunc_20.hpp"
