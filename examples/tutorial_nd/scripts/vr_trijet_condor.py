"""Module 4's script - the same ADL benchmark Q6 trijet analysis, now over
real CMS Open Data (data/datasets_xrootd.json: 15 files of Run2012B_SingleMu
in 5 datasets, read over xrootd from cmsxrootd.crc.nd.edu), with its
processor/reducer calls run on TaskVine workers in the HTCondor pool instead
of on this machine.

Three things change from Module 3's scripts/vr_trijet_taskvine.py; the
processor, reducer, and result loading do not:

1. The manager gets a name, <your username>-trijet-tutorial. It advertises
   that name to the catalog server (catalog.cse.nd.edu), which is how
   workers on pool nodes find it: they have no idea what host or port this
   script is on.
2. No vine.Factory here. Workers come from a separate vine_factory
   process, started in a second terminal, that submits them as HTCondor
   jobs asking for managers with that name.
3. environment=get_environment(): pool nodes don't have this Python
   environment installed, so every task ships a packed copy of it (built
   before class, so this call is a cache hit).
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


def trijet_processor(events):
    """Unchanged from Modules 2 and 3."""
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
    """Unchanged from Modules 2 and 3."""
    pattern = os.path.join(results_dir, dataset_name, processor_name, "*.pkl.zst")
    (result_file,) = glob.glob(pattern)
    return serialization.load(result_file)


def main():
    datasets_path = os.path.join(REPO_ROOT, "data", "datasets_xrootd.json")
    results_dir = os.path.join(REPO_ROOT, "results")
    checkpoint_dir = os.path.join(REPO_ROOT, "checkpoints")

    shutil.rmtree(results_dir, ignore_errors=True)
    shutil.rmtree(checkpoint_dir, ignore_errors=True)

    # Packs this conda/pixi environment into a tarball TaskVine ships to
    # every worker; reuses the cached tarball when nothing changed.
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
        processors={"trijet": trijet_processor},
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
