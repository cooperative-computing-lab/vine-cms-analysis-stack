"""Module 7's script - a skim instead of histograms, over the same real data
as Module 4 (data/datasets_xrootd.json: 5 datasets), on the HTCondor pool.

The processor keeps every event whose best trijet (the one with mass
closest to 172.5 GeV) has a mass between 150 and 195 GeV, and returns those
events' jets plus the trijet's pT and mass - an awkward array, not a
histogram. Skims can be large, so this script never brings them back to
the manager:

- reducer=accumulate_skims: two partial skims are concatenated.
- result_postprocess: runs on the worker, once per final result, writes the
  skim straight to parquet under SKIM_DIR (/project01, shared by the pool
  nodes and the login node), and returns only a small locator,
  {"path": ..., "events": ...}.
- result_postprocess_offload=True: that locator is what gets written into
  results/ - as a *.pointer.json file - instead of a *.pkl.zst pickle.
- is_result=lambda n, t, m: n >= 50_000: any group of partial results
  covering at least 50,000 events becomes a final result right away, so
  one run writes several parquet files per dataset instead of one.

Workers still come from your vine_factory in a second terminal (Module 4),
with condor-requirements added so they land on nodes that mount /project01.
"""

from __future__ import annotations

import getpass
import glob
import json
import os
import shutil
import uuid

import awkward as ak

from vine_reduce import get_environment
from vine_reduce.coffea import VineReduceCoffea
from vine_reduce.taskvine_distributor import TaskVineDistributor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MANAGER_NAME = f"{getpass.getuser()}-trijet-tutorial"

SKIM_DIR = f"/project01/ndcms/users/{getpass.getuser()}/trijet-skims"

MASS_LOW, MASS_HIGH = 150, 195


def trijet_skimmer(events):
    """Runs on a worker, once per chunk. Same best-trijet selection as
    Modules 2-4, but returns the selected events instead of filling
    histograms."""
    jets = ak.zip(
        {k: getattr(events.Jet, k) for k in ["x", "y", "z", "t", "btag"]},
        with_name="LorentzVector",
        behavior=events.Jet.behavior,
    )
    trijet = ak.combinations(jets, 3, fields=["j1", "j2", "j3"])
    trijet["p4"] = trijet.j1 + trijet.j2 + trijet.j3
    best = ak.firsts(trijet[ak.singletons(ak.argmin(abs(trijet.p4.mass - 172.5), axis=1))])

    # events with fewer than 3 jets have no trijet at all (None)
    in_window = ak.fill_none((best.p4.mass > MASS_LOW) & (best.p4.mass < MASS_HIGH), False)

    selected = events[in_window]
    best = ak.drop_none(best[in_window], axis=0)
    return ak.zip(
        {
            "jet_pt": selected.Jet.pt,
            "jet_eta": selected.Jet.eta,
            "jet_phi": selected.Jet.phi,
            "jet_mass": selected.Jet.mass,
            "jet_btag": selected.Jet.btag,
            "trijet_pt": best.p4.pt,
            "trijet_mass": best.p4.mass,
            # result_postprocess only sees the skim, not which dataset it
            # came from - carry the name along with the events
            "dataset": ak.Array([events.metadata["dataset"]] * len(selected)),
        },
        depth_limit=1,
    )


def accumulate_skims(a, b):
    """Reducer: two partial skims are just concatenated. Runs on a worker."""
    return ak.concatenate([a, b], axis=0)


def make_result_postprocess(skim_dir):
    """Builds the result_postprocess callback. It runs on a worker, once per
    final result, writes that result's events to parquet under skim_dir,
    and returns a small locator instead of the events themselves."""

    def result_postprocess(skim):
        if len(skim) == 0:
            return {"path": None, "events": 0}
        dataset_name = str(skim["dataset"][0])
        os.makedirs(skim_dir, exist_ok=True)
        # several final results per run can finish at the same time: each
        # gets its own file instead of racing to overwrite a shared one
        out_path = os.path.join(skim_dir, f"{dataset_name}.{uuid.uuid4().hex}.parquet")
        ak.to_parquet(skim, out_path)
        return {"path": out_path, "events": len(skim)}

    return result_postprocess


def read_pointers(results_dir, dataset_name, processor_name):
    """Reads every *.pointer.json vine_reduce wrote for this pipeline. Each
    is {"offloaded": ..., "processor_name": ..., "dataset_name": ...,
    "locator": <what result_postprocess returned>}."""
    pattern = os.path.join(results_dir, dataset_name, processor_name, "*.pointer.json")
    pointers = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            pointers.append(json.load(f))
    return pointers


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
    print(f"skims will be written under {SKIM_DIR}")
    print("waiting for workers from your vine_factory...")

    vr = VineReduceCoffea(
        processors={"skim": trijet_skimmer},
        input=datasets_path,
        distributor=distributor,
        results_dir=results_dir,
        checkpoint_dir=checkpoint_dir,
        chunksize=50_000,
        reducer=accumulate_skims,
        is_result=lambda n, t, m: n >= 50_000,
        result_postprocess=make_result_postprocess(SKIM_DIR),
        result_postprocess_offload=True,
    )

    with distributor:
        vr.compute()

    with open(datasets_path) as f:
        dataset_names = list(json.load(f))

    total = 0
    n_results = 0
    for dataset_name in dataset_names:
        for pointer in read_pointers(results_dir, dataset_name, "skim"):
            locator = pointer["locator"]
            total += locator["events"]
            n_results += 1
            print(f"{locator['events']:>6} events -> {locator['path']}")
    print(f"{n_results} final results, {total} events in the skim")


if __name__ == "__main__":
    main()
