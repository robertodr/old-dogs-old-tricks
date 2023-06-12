#include <cstdlib>
#include <filesystem>
#include <highfive/H5Easy.hpp>
#include <iostream>
#include <vector>

#include "mcmurchie_davidson/McMurchieDavidson.hpp"
#include "pade/Pade.hpp"

namespace fs = std::filesystem;

inline void
dump_to_h5(size_t n_points, size_t max_order, const std::string& label, const std::vector<double>& data, H5Easy::File& h5file)
{
    auto col = std::vector<double>(n_points);

    // save order by order
    for (auto o = 0; o <= max_order; ++o)
    {
        // collect column
        for (auto i = 0; i < n_points; ++i)
        {
            col[i] = data[o + i * (max_order + 1)];
        }

        H5Easy::dump(h5file, label + "order_" + std::to_string(o), col);
    }
}

int
main()
{
    constexpr auto max_order = 20;

    auto fpath = fs::path("data/boys_reference.h5");

    std::cout << "Path to reference data " << fpath << std::endl;

    auto ref = H5Easy::File(fpath.string(), H5Easy::File::ReadOnly);

    auto cpath = fs::path("data/boys_computed.h5");

    auto computed = H5Easy::File(cpath.string(), H5Easy::File::Overwrite);

    std::cout << "Path to computed data " << cpath << std::endl;

    std::string         lbl, in_lbl, out_lbl;
    std::vector<double> xs, ys;

    // loop over region: lo, mid, hi
    for (const auto& region : {"lo", "mid", "hi"})
    {
        lbl = std::string("/") + region + "_interval";

        in_lbl = lbl + "/xs";
        xs     = H5Easy::load<std::vector<double>>(ref, in_lbl);

        // "loop" over method: pade, mcmurchie_davidson
        // "loop" over backend: CPU, GPU

        {  // McMurchie-Davidson, CPU
            ys = mcmurchie_davidson::boys_function(max_order, xs);

            dump_to_h5(xs.size(), max_order, lbl + "/mcmurchie_davidson/cpu/ys/", ys, computed);
        }

        {  // Padé [5,6], CPU
            ys = pade::boys_function_5_6(max_order, xs);

            dump_to_h5(xs.size(), max_order, lbl + "/pade_5_6/cpu/ys/", ys, computed);
        }

        {  // Padé [9,10], CPU
            ys = pade::boys_function_9_10(max_order, xs);

            dump_to_h5(xs.size(), max_order, lbl + "/pade_9_10/cpu/ys/", ys, computed);
        }

        {  // Padé [15,16], CPU
            ys = pade::boys_function_15_16(max_order, xs);

            dump_to_h5(xs.size(), max_order, lbl + "/pade_15_16/cpu/ys/", ys, computed);
        }

        {  // Padé [25,26], CPU
            ys = pade::boys_function_25_26(max_order, xs);

            dump_to_h5(xs.size(), max_order, lbl + "/pade_25_26/cpu/ys/", ys, computed);
        }
    }

    return EXIT_SUCCESS;
}
