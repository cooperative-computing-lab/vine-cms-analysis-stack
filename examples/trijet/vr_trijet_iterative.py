"""The same introductory example as vr_trijet_taskvine.py, run through
vine_reduce's LocalDistributor instead of TaskVine - a plain
ProcessPoolExecutor-backed distributor that needs no cluster, no
vine_factory/vine_worker, and no environment packaging. Useful for a quick
local check of an analysis before scaling it out to a real TaskVine setup;
see vr_trijet_taskvine.py for that version, and its module docstring for
the full "REPLACE"-by-"REPLACE" walkthrough of adapting this template to
your own analysis (identical here, just swap "TaskVine" for "iterative"
below).

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
   in main() (the LocalDistributor, VineReduceCoffea call, results
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
import numpy as np

from vine_reduce import serialization
from vine_reduce.coffea import VineReduceCoffea
from vine_reduce.local_distributor import LocalDistributor

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
    """Runs in a local worker subprocess, once per chunk of NanoEvents -
    Q6Processor.process's body, unchanged, as a plain function (see module
    docstring). Swap this body (and the function's name, and the "trijet"
    key in main()'s `processors={...}` dict) for your own per-chunk
    analysis logic. The signature stays the same for any analysis: one
    `events` NanoEvents array in, any picklable object out (here, a dict of
    two Hists) - VineReduceCoffea's default_reducer already knows how to
    sum plain Hists and dicts of them across chunks, so no custom reducer
    is needed here (contrast with ../cortado/vr_cortado.py, whose skimmer
    returns awkward arrays instead and needs one)."""
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

    # per-event jet count, Poisson mean - high enough that most
    # events have the >=3 jets a trijet combination needs.
    JET_MEAN = 6.0

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

    # LocalDistributor runs each processor/reducer call in its own local
    # subprocess via concurrent.futures.ProcessPoolExecutor - no manager,
    # no vine_factory/vine_worker, no environment packaging, since worker
    # subprocesses already share this process's filesystem and Python env.
    # See vine_reduce/local_distributor.py's module docstring for the
    # tradeoffs versus TaskVineDistributor (see vr_trijet_taskvine.py).
    distributor = LocalDistributor(max_workers=2, checkpoint_dir=checkpoint_dir)

    # Every VineReduceCoffea/VineReduce parameter, spelled out explicitly
    # so this template shows the full surface area in one place. Commented
    # parameters show defaults.
    vr = VineReduceCoffea(
        # --------- what to run, over what ---------
        # {name: processor_fn} - one Pipeline per (processor, dataset) pair
        processors={"trijet": trijet_processor},

        # coffea-shaped dataset dict (or a json path) - see build_datasets()
        input=datasets,

        # extra kwargs passed to each processor call, beyond `events`
        # processor_args=None,

        # extra local files shipped to workers beyond the task itself
        # extra_files=[],

        # extra environment variables set for worker tasks
        # environment_variables={},

        # --- execution backend ---
        # where processor/reducer calls actually run - a LocalDistributor here
        distributor=distributor,

        # --- final results ---
        # where each dataset/processor's final result lands (see load_result)
        results_dir=results_dir,

        # by default, an accumulation counts as "final" once it covers every event of its
        # dataset. Pass a function(num_events, total_time_s, total_memory_mb) -> bool
        # instead to emit results in parts, e.g. every N events, every T seconds,
        # or once M MB have accumulated.
        # is_result=None,

        # no transform applied to a final result before it's written out, akin to lambda x: x
        # result_postprocess=None,


        # --------- combining results ---------
        # default reducer is akin to a += b, folds two chunks'/groups' results together,
        # which already sums Hists and dicts of them.
        # reducer=None

        # how many results get folded together per reduction step
        # reduction_size=10,

        # --------- reading events out of a chunk ------------
        # schema used to interpret each ROOT file's branches
        # schema=coffea.nanoevents.NanoAODSchema,

        # NanoEvents factory laziness mode - "virtual" arrays materialize on first use
        # mode="virtual",

        # TTree name read from each file
        # object_path="Events",

        # extra kwargs forwarded to uproot when opening each file
        # uproot_options=None,

        # ------ chunking / scheduling ------
        # events per chunk; None (the default) -> one chunk per file
        # chunksize=None,

        # cap on chunks in flight (processing + reducing) at once
        # max_chunks_active=1000,

        # cap on new chunks submitted per scheduling-loop iteration
        # max_chunks_cycle=100,

        # ------ checkpointing / restart ------
        # non-final checkpoints themselves are the distributor's concern
        # (see LocalDistributor's checkpoint_dir above), not VineReduce's

        # whether each accumulation should be checkpointed
        # checkpoint_accumulations=False,

        # time-based (runtime seconds) checkpoint trigger
        # checkpoint_time=None,

        # distance-based (accumulations since last checkpoint) checkpoint trigger
        # checkpoint_distance=None,

        # results_dir/vine_reduce.db; sqlite db tracking what's already been computed
        # db_path=None,
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
