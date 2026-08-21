"""START HERE: the introductory example for adapting vine_reduce's
VineReduceCoffea to your own coffea analysis, via the TaskVine executor.

This file is a complete, working analysis - ADL benchmark Q6Processor from
https://github.com/CoffeaTeam/coffea-benchmarks/blob/master/coffea-adl-benchmarks.py
(for events with at least three jets, plot the pT of the trijet
four-momentum whose invariant mass is closest to 172.5 GeV, the top quark
mass, and the maximum b-tag discriminant among that trijet's three jets;
the same query also appears as `q6` in ../ADL/processors.py) - run over
synthetic NanoAOD-like data so it works standalone, with no real CMS
dataset or grid proxy needed.

To turn this into *your* analysis, replace the three pieces marked
"REPLACE" below, in this order:

1. build_datasets() - point it at your own files instead of synthetic ones
   (see its docstring for the exact shape VineReduceCoffea expects). The
   "BOILERPLATE" section above it (ensure_datasets() and the
   ../write_test_data.py subprocess call) is synthetic-data plumbing you
   can delete entirely once you have real data.
2. trijet_processor() - swap in your own per-chunk analysis logic. Keep
   the signature (one `events` NanoEvents array in, any picklable object
   out) - a coffea ProcessorABC.process(self, events) body drops in almost
   unchanged. See ../ADL/vr_adl_benchmarks.py for eight more examples of
   this translation, and ../cortado/vr_cortado.py for a case that also
   needs a custom reducer (the default one just sums Hists, like this
   file's processor returns).
3. main()'s `processors={"trijet": trijet_processor}` - one entry per
   (name, processor function) pair you want run; add more the same way
   ../ADL/vr_adl_benchmarks.py does for its eight queries. Everything else
   in main() (the TaskVine workers, VineReduceCoffea call, results
   loading) is boilerplate that works unchanged for most analyses.
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

JET_MEAN = 6.0  # per-event jet count, Poisson mean - high enough that most
# events have the >=3 jets a trijet combination needs.

# -----------------------------------------------------------------------
# BOILERPLATE - synthetic-data plumbing only. Delete ensure_datasets() and
# the build_datasets() call below once you have real files to point at;
# nothing else in this file depends on how the data was produced.
# -----------------------------------------------------------------------


def ensure_datasets(data_dir, *script_args):
    """Generates data_dir's ROOT files + datasets.json manifest by running
    ../write_test_data.py as a subprocess - but only the first time; once
    datasets.json exists, later runs reuse the same synthetic data instead
    of paying the generation cost again. Returns the loaded manifest,
    already shaped as the `input` dict VineReduceCoffea expects (see that
    script's module docstring)."""
    write_test_data = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "write_test_data.py"
    )
    manifest_path = os.path.join(data_dir, "datasets.json")
    if not os.path.exists(manifest_path):
        subprocess.run(
            [sys.executable, write_test_data, "--data-dir", data_dir, *script_args],
            check=True,
        )
    with open(manifest_path) as f:
        return json.load(f)


# -----------------------------------------------------------------------
# REPLACE - this is your analysis. Everything else in this file just wires
# whatever function(s) you put here into VineReduceCoffea.
# -----------------------------------------------------------------------


def trijet_processor(events):
    """Runs remotely, once per Chunk of NanoEvents - Q6Processor.process's
    body, unchanged, as a plain function (see module docstring). Swap this
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
# REPLACE - point this at your own dataset(s) once you're past synthetic
# data; keep the return shape described below, since that's what
# VineReduceCoffea (and coffea.dataset_tools.preprocess(), which produces
# the same shape for real files) expects.
# -----------------------------------------------------------------------


def build_datasets(data_dir):
    """Builds the `input` dict VineReduceCoffea expects: one entry per
    dataset name, each a dict with a "metadata" dict and a "files" dict
    mapping each file's path to {"object_path": ..., "num_entries": ...}
    - coffea's own preprocessed-dataset shape, exactly what
    coffea.dataset_tools.preprocess() produces for real files (see
    coffea_input_to_datasets, VineReduceCoffea's default input_to_datasets,
    for how it's read). Here it's generated (or reused) via
    ../write_test_data.py - see that script's module docstring for how the
    manifest it writes matches this shape."""
    return ensure_datasets(
        data_dir,
        "--dataset-names", "ttbar_like",
        "--num-files", "3",
        "--jet-mean", str(JET_MEAN),
    )


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
    data_dir = os.path.join(here, "data")
    results_dir = os.path.join(here, "results")
    checkpoint_dir = os.path.join(here, "checkpoints")

    # Fresh data every run, so stale results/checkpoints from a previous run
    # do not clutter previous test run.
    shutil.rmtree(results_dir, ignore_errors=True)
    shutil.rmtree(checkpoint_dir, ignore_errors=True)

    # This is only needed because we are generating synthetic data.
    datasets = build_datasets(data_dir)

    # Same TaskVineDistributor + vine.Factory setup as ../cortado: one
    # local worker process, no cluster or separate vine_worker needed to run
    # this example standalone.
    distributor = TaskVineDistributor(
        port=0, resources_processor={"cores": 1}, resources_reducer={"cores": 1}
    )
    workers = vine.Factory(manager_host_port=f"localhost:{distributor.port}")
    workers.cores = 2
    workers.min_workers = 1
    workers.max_workers = 1

    with workers:
        # reducer defaults to VineReduceCoffea's own default_reducer, which
        # already knows how to sum a dict of Hist objects (see module
        # docstring) - no custom reducer needed here, unlike cortado.
        vr = VineReduceCoffea(
            processors={"trijet": trijet_processor},
            input=datasets,
            chunksize=10000,
            results_dir=results_dir,
            checkpoint_dir=checkpoint_dir,
            distributor=distributor,
        )
        vr.compute()
    distributor.shutdown()

    result = load_result(results_dir, "ttbar_like", "trijet")
    # flow=True: our uncorrelated, random-direction synthetic jets often
    # give the trijet system a pT above the [0, 200) GeV plotted below, so
    # plain .sum() (which excludes the overflow bin) would undercount how
    # many events actually got a trijet - flow=True counts every fill,
    # in-range or not.
    trijetpt_entries = result["trijetpt"].sum(flow=True)
    maxbtag_entries = result["maxbtag"].sum(flow=True)
    print(f"trijet pT histogram: {trijetpt_entries:.0f} entries")
    print(f"max b-tag histogram: {maxbtag_entries:.0f} entries")

    # Sanity check: with JET_MEAN=6.0, the vast majority of synthetic events
    # have >=3 jets, so both histograms should be filled for most events
    # across all files.
    total_events = sum(
        file_info["num_entries"] for file_info in datasets["ttbar_like"]["files"].values()
    )
    assert trijetpt_entries == maxbtag_entries, "both histograms should fill once per event"
    assert trijetpt_entries > 0.5 * total_events, (
        "expected most synthetic events to have >=3 jets and fill the trijet histograms"
    )
    print("OK: trijet histograms filled for the large majority of events")


if __name__ == "__main__":
    main()
