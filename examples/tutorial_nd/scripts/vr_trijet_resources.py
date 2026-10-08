"""Module 5's script - Module 4's scripts/vr_trijet_condor.py with the
three knobs this module is about pulled out as command-line arguments:

    python scripts/vr_trijet_resources.py CHUNKSIZE MEMORY_MB [REDUCTION_SIZE]

- CHUNKSIZE: events per processor call. VineReduce halves it automatically
  when a processor call is killed for using more memory than it was allowed,
  for chunks it hasn't cut yet.
- MEMORY_MB: the memory each processor call is allowed (resources_processor).
  Only memory is declared for processors here, not cores: a task that
  declares cores=1 is given at least that core's share of the worker's
  memory (8000 MB / 4 cores here), whatever smaller memory_mb it asks for.
- REDUCTION_SIZE: how many partial results one reducer call folds together
  (default 10). A fan-in count, not a resource.

Workers still come from your vine_factory in a second terminal (Module 4).
"""

from __future__ import annotations

import getpass
import glob
import json
import os
import shutil
import sys

import awkward as ak
import hist
import numpy as np

from vine_reduce import get_environment, serialization
from vine_reduce.coffea import VineReduceCoffea
from vine_reduce.taskvine_distributor import TaskVineDistributor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MANAGER_NAME = f"{getpass.getuser()}-trijet-tutorial"

DATASET = "SingleMu_Run2012B_194790-195398"


def trijet_processor(events):
    """Unchanged from Modules 2-4."""
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
    """Unchanged from Modules 2-4."""
    pattern = os.path.join(results_dir, dataset_name, processor_name, "*.pkl.zst")
    (result_file,) = glob.glob(pattern)
    return serialization.load(result_file)


def main():
    chunksize = int(sys.argv[1])
    memory_mb = int(sys.argv[2])
    reduction_size = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    print(f"chunksize={chunksize} memory_mb={memory_mb} reduction_size={reduction_size}")

    # Just one file of the real data: the first of dataset DATASET (430,713
    # events), so that one chunk can hold all of it.
    with open(os.path.join(REPO_ROOT, "data", "datasets_xrootd.json")) as f:
        all_datasets = json.load(f)
    files = all_datasets[DATASET]["files"]
    first_file = next(iter(files))
    datasets = {
        DATASET: {
            "metadata": all_datasets[DATASET]["metadata"],
            "files": {first_file: files[first_file]},
        }
    }
    results_dir = os.path.join(REPO_ROOT, "results")
    checkpoint_dir = os.path.join(REPO_ROOT, "checkpoints")

    shutil.rmtree(results_dir, ignore_errors=True)
    shutil.rmtree(checkpoint_dir, ignore_errors=True)

    environment = get_environment()

    distributor = TaskVineDistributor(
        name=MANAGER_NAME,
        port=0,
        environment=environment,
        resources_processor={"memory_mb": memory_mb},
        resources_reducer={"cores": 1, "memory_mb": 2000},
    )
    print(f"TaskVine manager {MANAGER_NAME!r} listening on port {distributor.port}")
    print("waiting for workers from your vine_factory...")

    vr = VineReduceCoffea(
        processors={"trijet": trijet_processor},
        input=datasets,
        distributor=distributor,
        results_dir=results_dir,
        checkpoint_dir=checkpoint_dir,
        chunksize=chunksize,
        reduction_size=reduction_size,
    )

    with distributor:
        vr.compute()

    result = load_result(results_dir, DATASET, "trijet")
    trijetpt_entries = result["trijetpt"].sum(flow=True)
    maxbtag_entries = result["maxbtag"].sum(flow=True)
    print(f"trijet pT histogram: {trijetpt_entries:.0f} entries")
    print(f"max b-tag histogram: {maxbtag_entries:.0f} entries")


if __name__ == "__main__":
    main()
