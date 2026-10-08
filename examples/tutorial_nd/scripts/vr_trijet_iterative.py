"""Module 2's script - the ADL benchmark Q6 trijet analysis
(https://github.com/CoffeaTeam/coffea-benchmarks/blob/master/coffea-adl-benchmarks.py:
for events with at least three jets, plot the pT of the trijet four-momentum
whose invariant mass is closest to 172.5 GeV, the top quark mass, and the
maximum b-tag discriminant among that trijet's three jets) run through
vine_reduce's LocalDistributor: a plain ProcessPoolExecutor-backed
distributor that needs no cluster, no vine_factory/vine_worker, and no
environment packaging, since worker subprocesses already share this
process's filesystem and Python env. Module 3 runs the exact same
trijet_processor through a real TaskVine factory/worker instead - only the
`distributor=` argument changes.

Adapted from examples/trijet/vr_trijet_iterative.py: that version generates
its own synthetic data via write_test_data.py; this one reads real CMS Open
Data, the 15 files of data/datasets_vast.json (3.7M events in 5 datasets, on
/project01). Run it from examples/tutorial_nd/, where scripts/ and data/ live.
"""

from __future__ import annotations

import glob
import json
import os
import shutil

import awkward as ak
import hist
import numpy as np

from vine_reduce import serialization
from vine_reduce.coffea import VineReduceCoffea
from vine_reduce.local_distributor import LocalDistributor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def trijet_processor(events):
    """Runs in a local worker subprocess, once per chunk of NanoEvents - one
    `events` NanoEvents array in, any picklable object out (here, a dict of
    two Hists). VineReduceCoffea's default reducer already knows how to sum
    plain Hists and dicts of them across chunks, so no custom reducer is
    needed here (contrast with Module 7's result_postprocess)."""
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


def main():
    datasets_path = os.path.join(REPO_ROOT, "data", "datasets_vast.json")
    results_dir = os.path.join(REPO_ROOT, "results")
    checkpoint_dir = os.path.join(REPO_ROOT, "checkpoints")

    # Fresh run every time, so stale results/checkpoints don't linger.
    shutil.rmtree(results_dir, ignore_errors=True)
    shutil.rmtree(checkpoint_dir, ignore_errors=True)

    # The datasets: each a name plus its file list, already coffea's own
    # preprocessed shape - VineReduceCoffea accepts a path to a json file
    # holding them directly, no need to load them into a dict first.
    distributor = LocalDistributor(max_workers=2)

    vr = VineReduceCoffea(
        # one Pipeline per (processor, dataset) pair - 1 processor x 5 datasets
        processors={"trijet": trijet_processor},
        input=datasets_path,
        # where processor/reducer calls actually run
        distributor=distributor,
        results_dir=results_dir,
        checkpoint_dir=checkpoint_dir,
        # events per processor call - ~90 chunks over today's ~3.7M events
        chunksize=50_000,
    )

    with distributor:
        vr.compute()

    with open(datasets_path) as f:
        dataset_names = list(json.load(f))

    # one line per dataset, then the total over all of them
    total_pt = total_btag = 0
    for name in dataset_names:
        result = load_result(results_dir, name, "trijet")
        # flow=True counts every fill, in-range or not: a trijet pT above the
        # [0, 200) GeV histogram range lands in the overflow bin, which plain
        # .sum() would leave out
        pt = result["trijetpt"].sum(flow=True)
        btag = result["maxbtag"].sum(flow=True)
        print(f"{name}: {pt:.0f} entries (trijet pT), {btag:.0f} entries (max b-tag)")
        total_pt += pt
        total_btag += btag
    print(f"trijet pT histogram, all datasets: {total_pt:.0f} entries")
    print(f"max b-tag histogram, all datasets: {total_btag:.0f} entries")


if __name__ == "__main__":
    main()
