#!/usr/bin/env python3

from pathlib import Path

import typer

import mcmurchie_davidson
import pade


def main(
    max_order: int = typer.Argument(..., help="Maximum order of the Boys' function"),
    folder: Path = typer.Argument(
        Path("src"), help="Where to store the generated sources."
    ),
    overwrite: bool = typer.Option(False, help="Overwrite existing generated files."),
) -> None:
    md = folder / "mcmurchie_davidson"
    md.mkdir(parents=True, exist_ok=overwrite)
    print(f"Generating McMurchie-Davidson table for orders 1 <= O <= {max_order}")
    mcmurchie_davidson.generate(max_order=max_order, folder=md)

    pd = folder / "pade"
    pd.mkdir(parents=True, exist_ok=overwrite)

    for P, Q in [(5, 6), (9, 10), (15, 16), (25, 26)]:
        print(f"Generating [{P}/{Q}] Padé approximant for orders 1 <= O <= {max_order}")
        pade.generate(max_order=max_order, P=P, Q=Q, folder=pd)


if __name__ == "__main__":
    typer.run(main)
