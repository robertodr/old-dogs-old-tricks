import datetime
from pathlib import Path

from sympy import Matrix, Pow, Range, factorial, lowergamma, symbols
from tqdm import trange

_MD_TEMPLATE = """/* This file was autogenerated on {date_and_time} DO NOT EDIT! */

#pragma once

#include <array>

namespace mcmurchie_davidson {{
/** Pretabulated coefficients for the seven-term Taylor expansion of the Boys function of order {order}.
 *  See: McMurchie, L. E.; Davidson, E. R. J. Comput. Phys. 1978, 26, 218. https://doi.org/10.1016/0021-9991(78)90092-X
 *  Row i contains the coefficients for the expansion at point i. The
 *  coefficients are in natural order: from 0-th to 6-th power.
 */
template <> inline constexpr std::array<std::array<double, 7>, 121> pretabulated<{order}>() {{
  // clang-format off
  return {{{{{lines}}}}};
  // clang-format on
}}
}} // namespace mcmurchie_davidson
"""

_PRETABULATED = """/* This file was autogenerated on {date_and_time} DO NOT EDIT! */

#pragma once

#include <array>

namespace mcmurchie_davidson {{
/**
 * Obtain index of point within pretabulated grid.
 *
 * @param[in] x the point
 * @return The index of the point in the pretabulated grid.
 */
inline constexpr auto
grid_point(double x) -> size_t
{{
    return (x > 1.0e5) ? static_cast<size_t>(1.0e6) : static_cast<size_t>(10.0 * x + 0.5);
}}

/**
 * Pretabulated values of given order.
 *
 * @tparam order Order of the table to fetch.
 * @return table of values.
 */
template <auto order>
inline constexpr std::array<std::array<double, 7>, 121> pretabulated();
}}  // namespace mcmurchie_davidson

{lines}
"""

_CPP = """/* This file was autogenerated on {date_and_time} DO NOT EDIT! */

#include <stdexcept>
#include <string>
#include <vector>

#include "McMurchieDavidson.hpp"
#include "Utilities.hpp"

namespace mcmurchie_davidson {{
auto
boys_function(size_t order, const std::vector<double>& xs) -> std::vector<double>
{{
    switch (order)
    {{
       case 0:
           return boys_function_0(xs);
       {lines}
       default:
           throw std::invalid_argument("Maximum supported order is {max_order}. Requested order is " + std::to_string(order));
    }}
}}
}}  // namespace mcmurchie_davidson
"""


def generate(folder: Path, *, max_order: int) -> None:
    date_and_time = datetime.datetime.now().isoformat(timespec="minutes")

    # generate CMakeLists.txt
    with Path(folder / "CMakeLists.txt").open("w") as fh:
        fh.write(
            f"""# This file was autogenerated on {date_and_time} DO NOT EDIT!

target_sources(run PRIVATE McMurchieDavidson.cpp)
        """
        )

    # generate include file with interface
    with Path(folder / "Pretabulated.hpp").open("w") as fh:
        fh.write(
            _PRETABULATED.format(
                date_and_time=date_and_time,
                lines="\n".join(
                    [
                        f'#include "tables/BFunc_{n:02d}.hpp"'
                        for n in range(1, max_order + 1)
                    ]
                ),
            )
        )

    # generate source file
    with Path(folder / "McMurchieDavidson.cpp").open("w") as fh:
        fh.write(
            _CPP.format(
                date_and_time=date_and_time,
                max_order=max_order,
                lines="\n".join(
                    [
                        f"case {n}:\n    return cpu::boys_function<{n}>(xs);"
                        for n in range(1, max_order + 1)
                    ]
                ),
            )
        )

    # declare sympy symbols
    x, h = symbols("x h", real=True, positive=True)
    h = 0.1
    # assemble the factorial factors for the Taylor expansion beforehand
    taylors = Matrix([(Pow(-1, i) / factorial(i)) for i in Range(7)])

    (folder / "tables").mkdir(parents=True, exist_ok=True)

    for n in trange(1, max_order + 1):
        ls = []
        x = 0
        for i in range(121):
            vs = []
            if i == 0:
                # at x = 0.0, we can evaluate the Boys function analytically
                vs = Matrix(
                    [(1 / (2 * i + 1)) for i in Range(n, n + 7)]
                ).multiply_elementwise(taylors)
            else:
                # away from 0.0, we use the lower gamma function
                vs = Matrix(
                    [
                        lowergamma(i + 1 / 2, x) / (2 * Pow(x, i + 1 / 2))
                        for i in Range(n, n + 7)
                    ]
                ).multiply_elementwise(taylors)
            ls.append(f"{{{', '.join([format(v.evalf(30), '3.20e') for v in vs])}}}")
            # next point
            x += h

        with (folder / f"tables/BFunc_{n:02d}.hpp").open("w") as fh:
            fh.write(
                _MD_TEMPLATE.format(
                    date_and_time=date_and_time,
                    order=n,
                    lines=f",\n{' '*11}".join(ls),
                )
            )
