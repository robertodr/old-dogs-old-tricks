#!/usr/bin/env python3

import datetime
from pathlib import Path

import h5py
import typer
from sympy import Float, Integer, Pow, Range, lowergamma, symbols
from tqdm import trange

x = symbols("x", real=True, positive=True)
i, n = symbols("i n", integer=True, positive=True)


def interval(*, a, b, n):
    a = Float(a)
    b = Float(b)
    n = Integer(n)

    h = (b - a) / n
    return h, [Float(a + h * i) for i in Range(n)]


def boys(n, x):
    """Analytical implementation using the lower gamma function."""
    if x == 0:
        return Float(1 / (2 * n + 1))
    return lowergamma(n + 1 / 2, x) / (2 * Pow(x, (n + 1 / 2)))


def main(
    file_path: Path = typer.Argument(
        "boys_reference.h5",
        help="Filename for the reference dataset. Will be saved in the 'data' folder.",
    )
) -> None:
    # change suffix to ".h5"
    file_path = file_path.with_suffix(".h5")

    path = ("data" / file_path).resolve()

    print(f"Generating and saving reference values to {path}")

    intervals = {
        "lo": (0, 11.5, 10_000),
        "mid": (11.5, 13.5, 1_000),
        "hi": (13.5, 150, 100_000),
    }

    with h5py.File(path, "w") as fh:
        fh.attrs["created"] = f"{datetime.datetime.now().isoformat(timespec='minutes')}"

        for k, v in intervals.items():
            h, xs = interval(a=v[0], b=v[1], n=v[2])
            print(
                f"Generated {v[2]} points in [{v[0]}, {v[1]}) with uniform spacing {h:.3f}"
            )

            fh.create_dataset(
                f"{k}_interval/xs",
                data=[x.evalf(30) for x in xs],
                dtype=float,
                compression="gzip",
                compression_opts=9,
            )

            fh[f"{k}_interval/xs"].attrs["min"] = v[0]
            fh[f"{k}_interval/xs"].attrs["max"] = v[1]
            fh[f"{k}_interval/xs"].attrs["step"] = float(h)

            for o in trange(33):
                n = Integer(o)
                fh.create_dataset(
                    f"{k}_interval/ys/order_{o}",
                    data=[boys(n, x).evalf(30) for x in xs],
                    dtype=float,
                    compression="gzip",
                    compression_opts=9,
                )


if __name__ == "__main__":
    typer.run(main)
