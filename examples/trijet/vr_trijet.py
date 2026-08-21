"""ADL benchmark Q6, translated from coffea's own ProcessorABC into
vine_reduce.VineReduceCoffea over synthetic NanoAOD-like data, via the
TaskVine executor.

Q6Processor comes from
https://github.com/CoffeaTeam/coffea-benchmarks/blob/master/coffea-adl-benchmarks.py:
for events with at least three jets, plot the pT of the trijet
four-momentum whose invariant mass is closest to 172.5 GeV (the top quark
mass) in each event, and the maximum b-tag discriminant among that
trijet's three jets. The same query also appears as `q6` in
../ADL/processors.py
"""

from __future__ import annotations

import glob
import os
import shutil

import awkward as ak
import hist
import ndcctools.taskvine as vine
import numpy as np

import write_test_data
from vine_reduce import serialization
from vine_reduce.coffea import VineReduceCoffea
from vine_reduce.taskvine_distributor import TaskVineDistributor

JET_MEAN = 6.0  # per-event jet count, Poisson mean - high enough that most
# events have the >=3 jets a trijet combination needs.


def trijet_processor(events):
    """Runs remotely, once per Chunk of NanoEvents - Q6Processor.process's
    body, unchanged, as a plain function (see module docstring)."""
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


def build_datasets(data_dir):
    """Builds the `input` dict VineReduceCoffea expects - see
    ../cortado/vr_cortado.py's build_datasets for the full shape
    explanation."""
    rng = np.random.default_rng()
    files = write_test_data.generate_dataset_files(
        data_dir,
        "ttbar_like",
        3,    # files per dataset
        6.0,  # jet mean
        rng
    )
    return {
        "ttbar_like": {
            "metadata": {},
            "files": {
                path: {"object_path": "Events", "num_entries": num_events}
                for path, num_events in files.items()
            },
        }
    }


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
