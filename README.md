# Vine CMS Analysis Stack

![vine-cms-analysis-stack banner: a VineReduceCoffea snippet, this stack's
HTCondor pool at Notre Dame, and the ADL trijet benchmark this repo
ships](docs/banner.svg)

A reference stack for running CMS analysis
workflows with [Coffea](https://github.com/scikit-hep/coffea) on top of
[TaskVine](https://cctools.readthedocs.io/en/stable/taskvine), orchestrated
by [VineReduce](https://github.com/cooperative-computing-lab/vine-reduce).
Nothing here depends on Notre Dame resources, so it should work anywhere
there's a Python environment and either local cores or a batch cluster
(HTCondor, SLURM, SGE, ...) to point workers at.

This README covers setting up the environment and running an example
analysis. For everything else — why VineReduce exists, how the
distributor/executor split works, packaging environments for remote
workers, and the full example index — see **[DOC.md](DOC.md)**.

## Installation

Clone this repo — it carries its own `environment.yml`/`pyproject.toml`,
so it's the only clone needed to start running or adapting the examples.
Python 3.13+ is required. `ndcctools` (which provides TaskVine) is a
conda-forge-only package, so both options below go through a conda-forge
channel rather than plain PyPI.

```bash
git clone https://github.com/cooperative-computing-lab/vine-cms-analysis-stack.git
cd vine-cms-analysis-stack
```

### Option A: conda

```bash
conda env create -f environment.yml
conda activate vine-cms-analysis-stack
```

### Option B: pixi

[pixi](https://pixi.sh) reproduces the same conda-forge environment from
this repo's own `pyproject.toml`/`pixi.lock`, pinned and managed
automatically; it also installs VineReduce from PyPI on its own.

```bash
curl -fsSL https://pixi.sh/install.sh | bash   # if you don't already have pixi

pixi install          # runtime environment
pixi install -e dev   # optional: adds pytest, black, flake8, pyright
```

Run everything through `pixi run` (e.g. `pixi run python analysis_script.py`)
so it picks up the managed environment, or drop into `pixi shell` for the
rest of the session.

Need to ship this environment to remote TaskVine workers instead of
running locally? See ["Packaging the environment for TaskVine
workers"](DOC.md#packaging-the-environment-for-taskvine-workers) in DOC.md.

## Example: running an analysis

[`examples/trijet/vr_trijet_taskvine.py`](examples/trijet/vr_trijet_taskvine.py)
is the introductory template for adapting VineReduce's `VineReduceCoffea`
to your own Coffea analysis. It's a complete, working analysis — ADL
benchmark Q6 (for events with at least three jets, plot the pT of the
trijet four-momentum whose invariant mass is closest to the top quark
mass, and the max b-tag discriminant among that trijet's jets) — run over
synthetic NanoAOD-like data, so it works standalone with no real CMS
dataset or grid proxy needed.
[`examples/trijet/vr_trijet_iterative.py`](examples/trijet/vr_trijet_iterative.py)
is the same example run over `LocalDistributor` instead of TaskVine — no
cluster or `vine_factory`/`vine_worker` needed, useful for a quick local
check.

Run it from inside this repo:

```bash
cd examples/trijet

# conda
conda activate vine-cms-analysis-stack
python vr_trijet_taskvine.py

# pixi
pixi run python vr_trijet_taskvine.py
```

Its source is reproduced below, with the commented-out default
parameters trimmed for space — see the real file, linked above, for the
fully annotated version showing every `VineReduceCoffea` parameter:

```python
"""
To adapt this into your own analysis, replace the pieces marked
`REPLACE` in the file: `datasets=...` (point it at your own files),
`trijet_processor()` (your per-chunk analysis logic), and the
`processors={...}` entry in `main()`.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys

import awkward as ak
import hist
import ndcctools.taskvine as vine
import numpy as np

from vine_reduce import serialization
from vine_reduce.coffea import VineReduceCoffea
from vine_reduce.taskvine_distributor import TaskVineDistributor

# -----------------------------------------------------------------------
# REPLACE - this is your analysis. Everything else in this file just wires
# whatever function(s) you put here into VineReduceCoffea.
# -----------------------------------------------------------------------


def trijet_processor(events):
    """Runs remotely, once per chunk of NanoEvents. Swap this
    body (and the function's name, and the "trijet" key in main()'s
    `processors={...}` dict) for your own per-chunk analysis logic. The
    signature stays the same for any analysis: one `events` NanoEvents
    array in, any picklable object out (here, a dict of two Hists) -
    VineReduceCoffea's default_reducer already knows how to sum plain
    Hists and dicts of them across chunks, so no custom reducer is needed
    here (contrast with ../cortado/vr_cortado.py, whose skimmer returns
    awkward arrays instead and needs one)."""
    jets = ak.zip(
        {k: getattr(events.Jet, k) for k in ["x", "y", "z", "t", "btag"]},
        with_name="LorentzVector",
        behavior=events.Jet.behavior,
    )
    trijet = ak.combinations(jets, 3, fields=["j1", "j2", "j3"])
    trijet["p4"] = trijet.j1 + trijet.j2 + trijet.j3
    trijet = ak.flatten(
        trijet[ak.singletons(ak.argmin(abs(trijet.p4.mass - 172.5), axis=1))]
    )
    maxBtag = np.maximum(
        trijet.j1.btag,
        np.maximum(trijet.j2.btag, trijet.j3.btag),
    )
    return {
        "trijetpt": hist.Hist.new.Reg(
            100, 0, 200, name="pt3j", label="Trijet $p_{T}$ [GeV]"
        )
        .Double()
        .fill(trijet.p4.pt),
        "maxbtag": hist.Hist.new.Reg(
            100, 0, 1, name="btag", label="Max jet b-tag score"
        )
        .Double()
        .fill(maxBtag),
    }


def load_result(results_dir, dataset_name, processor_name):
    """Final results land under results_dir/<dataset_name>/<processor_name>/
    as a single compressed, pickled file (name includes a random uuid, hence
    the glob). serialization.load reverses what the reducer wrote."""
    pattern = os.path.join(results_dir, dataset_name, processor_name, "*.pkl.zst")
    (result_file,) = glob.glob(pattern)
    return serialization.load(result_file)


# -----------------------------------------------------------------------
# BOILERPLATE below, except the `processors={...}` dict passed to
# VineReduceCoffea (REPLACE its "trijet": trijet_processor entry with your
# own name(s)/function(s) - add more entries the same way
# ../ADL/vr_adl_benchmarks.py does for its eight queries) and the results
# section at the bottom, which is specific to what trijet_processor
# returns and will need adapting to whatever your own processor(s) return.
# -----------------------------------------------------------------------


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(here, "results")

    # REPLACE: Point this to your own dataset information.
    # Currently it expects it in the format of coffea.preprocess
    datasets = ...

    # vine.Factory setup as with a local worker process,
    # no cluster or separate vine_worker needed to run
    # this example standalone. (could also use batch_type="condor"|"slurm"|"sge")
    # environment=None here means tasks run in whatever Python env the
    # worker was started with. Pass a poncho package/tarball path (see
    # ndcctools.taskvine.Manager.declare_poncho) to ship a self-contained
    # env instead, e.g. when workers run on nodes without this project
    # pre-installed.
    distributor = TaskVineDistributor(
        port=0,
        environment=None,
        resources_processor={"cores": 1},
        resources_reducer={"cores": 1}
    )

    vr = VineReduceCoffea(
        # {name: processor_fn} - one Pipeline per (processor, dataset) pair
        processors={"trijet": trijet_processor},

        # coffea-shaped dataset dict (or a json path) - see build_datasets()
        input=datasets,

        # where processor/reducer calls actually run - a TaskVineDistributor here
        distributor=distributor,

        # where each dataset/processor's final result lands (see load_result)
        results_dir=results_dir,
    )

    workers = vine.Factory(manager_host_port=f"localhost:{distributor.port}")
    workers.ssl = True
    workers.cores = 2
    workers.min_workers = 1
    workers.max_workers = 1

    with distributor, workers:
        vr.compute()

    result = load_result(results_dir, "ttbar_like", "trijet")

    trijetpt_entries = result["trijetpt"].sum(flow=True)
    maxbtag_entries = result["maxbtag"].sum(flow=True)
    print(f"trijet pT histogram: {trijetpt_entries:.0f} entries")
    print(f"max b-tag histogram: {maxbtag_entries:.0f} entries")

if __name__ == "__main__":
    main()
```

See [`examples/README.md`](examples/README.md) for the full index of
runnable examples — all eight ADL benchmarks, plus two that have
actually been run in production at Notre Dame, beyond the synthetic
data shown on this page:
[`examples/cortado`](examples/cortado/vr_cortado.py), a skim with a
custom reducer that's also been run at scale (real-data results to
follow), and, in
[`examples/ttBar`](examples/ttBar/run_processor_with_vr.py), the
[`TopEFT/ttbarEFT`](https://github.com/TopEFT/ttbarEFT) production
integration: this is how that CMS top-quark EFT search has actually
run its analysis stage through this stack, distributed over an
HTCondor pool — see DOC.md's ["Production use:
ttbarEFT"](DOC.md#production-use-ttbareft) for how that integration is
wired up — and [DOC.md](DOC.md) for everything else.

## Further reading

- [DOC.md](DOC.md) — full design rationale, VineReduce concepts,
  packaging environments for TaskVine workers, and more
- [examples/README.md](examples/README.md) — full index of runnable examples
- [`examples/ttBar/run_processor_with_vr.py`](examples/ttBar/run_processor_with_vr.py)
  — the real `ttbarEFT` production integration this stack runs at Notre
  Dame (see [DOC.md](DOC.md#production-use-ttbareft) for context)
- [`examples/cortado/vr_cortado.py`](examples/cortado/vr_cortado.py)
  — a skim with a custom reducer, also run at scale at Notre Dame
  (real-data results to follow)
- [VineReduce repository](https://github.com/cooperative-computing-lab/vine-reduce)
- [TaskVine documentation](https://cctools.readthedocs.io/en/stable/taskvine)
- [TopEFT/ttbarEFT](https://github.com/TopEFT/ttbarEFT) — the CMS
  top-quark EFT search using this stack in production

## License

This project is licensed under the GPL 2.0 - see the LICENSE file for details.
