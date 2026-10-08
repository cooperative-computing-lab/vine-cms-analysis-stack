"""Module 9's script - Module 2's scripts/vr_trijet_iterative.py over the
same data/datasets_vast.json, plus one file that doesn't exist in each of
the 5 datasets, to show failure_proportion.

Each bad entry is a path next to its dataset's real files that isn't there.
Every chunk of it fails to open; once one chunk has used up its `attempts`
(3), the whole file is given up on for good and logged to failed_files.log
in the current directory. What happens next is failure_proportion's call,
per dataset:

    permanently failed files / files in the dataset > failure_proportion
        -> the run aborts

Here that's 1 / 4 = 0.25 for every dataset, so:

    python scripts/vr_trijet_failures.py 0      # aborts
    python scripts/vr_trijet_failures.py 0.3    # finishes, with a warning
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import sys

import awkward as ak
import hist
import numpy as np

from vine_reduce import serialization
from vine_reduce.coffea import VineReduceCoffea
from vine_reduce.local_distributor import LocalDistributor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BAD_FILE_NAME = "DOES_NOT_EXIST.root"


def trijet_processor(events):
    """Unchanged from Module 2."""
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
    failure_proportion = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0

    results_dir = os.path.join(REPO_ROOT, "results")
    checkpoint_dir = os.path.join(REPO_ROOT, "checkpoints")

    shutil.rmtree(results_dir, ignore_errors=True)
    shutil.rmtree(checkpoint_dir, ignore_errors=True)

    # Module 2's datasets, plus one bad file in each. num_entries is made up:
    # chunks are cut from it without opening the file, so nothing notices
    # the file is missing until a processor call tries to read it.
    with open(os.path.join(REPO_ROOT, "data", "datasets_vast.json")) as f:
        datasets = json.load(f)
    for dataset in datasets.values():
        first_file = next(iter(dataset["files"]))
        bad_file = first_file.rsplit("/", 1)[0] + "/" + BAD_FILE_NAME
        dataset["files"][bad_file] = {"object_path": "Events", "num_entries": 100_000}
    n_files = len(next(iter(datasets.values()))["files"])
    print(
        f"failure_proportion={failure_proportion}, {len(datasets)} datasets of "
        f"{n_files} files, 1 file in each of them missing"
    )

    distributor = LocalDistributor(max_workers=2)

    vr = VineReduceCoffea(
        processors={"trijet": trijet_processor},
        input=datasets,
        distributor=distributor,
        results_dir=results_dir,
        checkpoint_dir=checkpoint_dir,
        chunksize=50_000,
        failure_proportion=failure_proportion,
    )

    with distributor:
        vr.compute()

    # one line per dataset, then the total over all of them
    total_pt = total_btag = 0
    for name in datasets:
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
