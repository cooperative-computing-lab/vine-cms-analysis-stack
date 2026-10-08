"""Module 8's script - Module 4's scripts/vr_trijet_condor.py, but with a
small family of processors instead of one: the same trijet analysis, keeping
only events whose best trijet mass lies within +/-10, +/-25, or +/-50 GeV of
172.5 GeV (the top quark mass).

One thing changes from Module 4: processors={...} gets three entries, built
by make_trijet_processor(window). VineReduce builds one pipeline per
(processor, dataset) pair - 3 processors x 1 dataset = 3 pipelines here - and
runs them all at once over the same manager and the same workers. Nothing
else needs wiring.

Workers still come from your vine_factory in a second terminal (Module 4).
"""

from __future__ import annotations

import getpass
import glob
import json
import os
import shutil

import awkward as ak
import hist
import numpy as np

from vine_reduce import get_environment, serialization
from vine_reduce.coffea import VineReduceCoffea
from vine_reduce.taskvine_distributor import TaskVineDistributor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MANAGER_NAME = f"{getpass.getuser()}-trijet-tutorial"

TOP_MASS = 172.5

# processor name -> half-width of the mass window, in GeV
MASS_WINDOWS = {
    "trijet_pm10": 10,
    "trijet_pm25": 25,
    "trijet_pm50": 50,
}


def make_trijet_processor(window):
    """Returns Module 4's trijet_processor, restricted to events whose best
    trijet mass is within `window` GeV of TOP_MASS. Each call builds a new
    function with its own `window` baked in."""

    def trijet_processor(events):
        jets = ak.zip(
            {k: getattr(events.Jet, k) for k in ["x", "y", "z", "t", "btag"]},
            with_name="LorentzVector",
            behavior=events.Jet.behavior,
        )
        trijet = ak.combinations(jets, 3, fields=["j1", "j2", "j3"])
        trijet["p4"] = trijet.j1 + trijet.j2 + trijet.j3
        trijet = ak.flatten(
            trijet[ak.singletons(ak.argmin(abs(trijet.p4.mass - TOP_MASS), axis=1))]
        )
        # the only new line: keep trijets inside this processor's window
        trijet = trijet[abs(trijet.p4.mass - TOP_MASS) < window]
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

    return trijet_processor


def load_result(results_dir, dataset_name, processor_name):
    """Unchanged from Modules 2-4."""
    pattern = os.path.join(results_dir, dataset_name, processor_name, "*.pkl.zst")
    (result_file,) = glob.glob(pattern)
    return serialization.load(result_file)


def main():
    datasets_path = os.path.join(REPO_ROOT, "data", "datasets_xrootd.json")
    results_dir = os.path.join(REPO_ROOT, "results")
    checkpoint_dir = os.path.join(REPO_ROOT, "checkpoints")

    shutil.rmtree(results_dir, ignore_errors=True)
    shutil.rmtree(checkpoint_dir, ignore_errors=True)

    environment = get_environment()

    distributor = TaskVineDistributor(
        name=MANAGER_NAME,
        port=0,
        environment=environment,
        resources_processor={"cores": 1, "memory_mb": 2000},
        resources_reducer={"cores": 1, "memory_mb": 2000},
    )
    print(f"TaskVine manager {MANAGER_NAME!r} listening on port {distributor.port}")
    print("waiting for workers from your vine_factory...")

    vr = VineReduceCoffea(
        processors={
            name: make_trijet_processor(window) for name, window in MASS_WINDOWS.items()
        },
        input=datasets_path,
        distributor=distributor,
        results_dir=results_dir,
        checkpoint_dir=checkpoint_dir,
        chunksize=100_000,
    )

    with distributor:
        vr.compute()

    with open(datasets_path) as f:
        dataset_names = list(json.load(f))

    # one line per processor, summed over the 5 datasets
    for name, window in MASS_WINDOWS.items():
        entries = 0
        for dataset_name in dataset_names:
            result = load_result(results_dir, dataset_name, name)
            entries += result["trijetpt"].sum(flow=True)
        print(f"{name} (|m - {TOP_MASS}| < {window} GeV): {entries:.0f} entries")


if __name__ == "__main__":
    main()
