"""Module 3's script - the same ADL benchmark Q6 trijet analysis as
scripts/vr_trijet_iterative.py (Module 2), over the same
data/datasets_vast.json, but run through a real TaskVine manager
(TaskVineDistributor) and one local TaskVine worker (vine.Factory) instead
of LocalDistributor's subprocesses. trijet_processor and load_result are
unchanged; only the distributor (and the worker factory it needs) differ.

Adapted from examples/trijet/vr_trijet_taskvine.py.
"""

from __future__ import annotations

import getpass
import glob
import json
import os
import shutil

import awkward as ak
import hist
import ndcctools.taskvine as vine
import numpy as np

from vine_reduce import serialization
from vine_reduce.coffea import VineReduceCoffea
from vine_reduce.taskvine_distributor import TaskVineDistributor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MANAGER_NAME = f"{getpass.getuser()}-trijet-tutorial"


def trijet_processor(events):
    """Unchanged from Module 2 - now it runs inside a TaskVine worker,
    once per chunk."""
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
    """Unchanged from Module 2."""
    pattern = os.path.join(results_dir, dataset_name, processor_name, "*.pkl.zst")
    (result_file,) = glob.glob(pattern)
    return serialization.load(result_file)


def main():
    datasets_path = os.path.join(REPO_ROOT, "data", "datasets_vast.json")
    results_dir = os.path.join(REPO_ROOT, "results")
    checkpoint_dir = os.path.join(REPO_ROOT, "checkpoints")

    shutil.rmtree(results_dir, ignore_errors=True)
    shutil.rmtree(checkpoint_dir, ignore_errors=True)

    # A TaskVine manager, listening on a free port (port=0) - it places
    # every processor/reducer call on whatever worker connects to it.
    distributor = TaskVineDistributor(
        name=MANAGER_NAME,
        port=0,
        resources_processor={"cores": 1},
        resources_reducer={"cores": 1},
    )
    print(f"TaskVine manager {MANAGER_NAME!r} listening on port {distributor.port}")

    vr = VineReduceCoffea(
        processors={"trijet": trijet_processor},
        input=datasets_path,
        distributor=distributor,
        results_dir=results_dir,
        checkpoint_dir=checkpoint_dir,
        chunksize=50_000,
    )

    # One TaskVine worker, started as a local process on this machine and
    # pointed straight at the manager's port - no catalog, no cluster yet.
    workers = vine.Factory(manager_host_port=f"localhost:{distributor.port}")
    workers.ssl = True
    workers.cores = 2
    workers.min_workers = 1
    workers.max_workers = 1
    # the worker exits as soon as it disconnects from this manager, instead
    # of lingering to look for more work
    workers.extra_options = "--single-shot"

    with distributor, workers:
        vr.compute()

    with open(datasets_path) as f:
        dataset_names = list(json.load(f))

    # one line per dataset, then the total over all of them
    total_pt = total_btag = 0
    for name in dataset_names:
        result = load_result(results_dir, name, "trijet")
        pt = result["trijetpt"].sum(flow=True)
        btag = result["maxbtag"].sum(flow=True)
        print(f"{name}: {pt:.0f} entries (trijet pT), {btag:.0f} entries (max b-tag)")
        total_pt += pt
        total_btag += btag
    print(f"trijet pT histogram, all datasets: {total_pt:.0f} entries")
    print(f"max b-tag histogram, all datasets: {total_btag:.0f} entries")


if __name__ == "__main__":
    main()
