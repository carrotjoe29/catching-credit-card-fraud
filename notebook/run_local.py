"""Run the notebook outside Jupyter, on a laptop.

    DATA_ROOT="/path/to/competition/data" python notebook/run_local.py     (add FAST=1 for a quick try)

Charts are saved to notebook/out/fig_XX.png instead of shown.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    out = Path(os.environ.get("OUT_DIR", HERE / "out")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if "DATA_ROOT" in os.environ:
        os.environ["DATA_ROOT"] = str(Path(os.environ["DATA_ROOT"]).expanduser().resolve())
    os.chdir(out)
    nb = json.loads((HERE / "s6e11_showcase.ipynb").read_text())
    count = [0]

    def show(*a, **k):
        count[0] += 1
        plt.savefig(out / f"fig_{count[0]:02d}.png", bbox_inches="tight")
        plt.close("all")

    plt.show = show
    env = {"display": print, "__name__": "__main__"}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            print(f"--- cell {i} ---", flush=True)
            exec(compile("".join(cell["source"]), f"cell{i}", "exec"), env)
    print(f"done: {count[0]} figures and submission.csv in {out}")


if __name__ == "__main__":
    main()
