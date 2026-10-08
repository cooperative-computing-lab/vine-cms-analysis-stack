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

RDataFrame flavor: the twin of scripts/vr_trijet_skim.py, with the same
dataset, the same VineReduce knobs, and the same skim. The processor selects
events with RDataFrame and hands the selected columns to awkward
(ak.from_rdataframe), so the reducer and result_postprocess are identical to
the coffea flavor's: a skim is an awkward array either way. Plain VineReduce
replaces VineReduceCoffea, so chunk_to_args is spelled out here (see
scripts/vr_trijet_iterative_rdf.py).
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
from vine_reduce.coffea import coffea_input_to_datasets
from vine_reduce.engine import VineReduce
from vine_reduce.taskvine_distributor import TaskVineDistributor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MANAGER_NAME = f"{getpass.getuser()}-trijet-tutorial"

SKIM_DIR = f"/project01/ndcms/users/{getpass.getuser()}/trijet-skims"

MASS_LOW, MASS_HIGH = 150, 195


# The one piece of the analysis that is C++: for one event's jets, try every
# 3-jet combination, and return {pT, max b-tag, mass} of the one whose
# invariant mass is closest to 172.5 GeV. RDataFrame compiles it with ROOT's
# interpreter. The #ifndef guard lets one process declare it many times,
# which a worker process will, since it runs many chunks.
TRIJET_CPP = """
#ifndef VR_TRIJET_H
#define VR_TRIJET_H
#include "Math/Vector4D.h"
#include "ROOT/RVec.hxx"

ROOT::RVecD BestTrijet(const ROOT::RVecF &pt, const ROOT::RVecF &eta,
                       const ROOT::RVecF &phi, const ROOT::RVecF &mass,
                       const ROOT::RVecF &btag) {
  ROOT::RVecD best{0., 0., 0.};
  double best_diff = 1e30;
  const auto n = pt.size();
  for (size_t i = 0; i < n; ++i)
    for (size_t j = i + 1; j < n; ++j)
      for (size_t k = j + 1; k < n; ++k) {
        ROOT::Math::PtEtaPhiMVector p4(pt[i], eta[i], phi[i], mass[i]);
        p4 += ROOT::Math::PtEtaPhiMVector(pt[j], eta[j], phi[j], mass[j]);
        p4 += ROOT::Math::PtEtaPhiMVector(pt[k], eta[k], phi[k], mass[k]);
        const double diff = std::abs(p4.M() - 172.5);
        if (diff < best_diff) {
          best_diff = diff;
          best = {p4.Pt(), std::max({btag[i], btag[j], btag[k]}), p4.M()};
        }
      }
  return best;
}
#endif
"""


def chunk_to_args(chunk, dataset_metadata, distributor_metadata=None):
    """Runs on a worker, once per chunk, before the processor: an
    RDataFrame over just this chunk's entries [start, stop), plus the
    dataset's name - result_postprocess only sees the skim, not which
    dataset it came from, so the name travels along with the events."""
    import ROOT

    df = ROOT.RDataFrame("Events", chunk.url).Range(chunk.start, chunk.stop)
    return df, dataset_metadata["dataset"]


def trijet_skimmer(args):
    """Runs on a worker, once per chunk. Same best-trijet selection as
    Modules 2-4, but returns the selected events instead of filling
    histograms."""
    import ROOT

    df, dataset_name = args
    ROOT.gInterpreter.Declare(TRIJET_CPP)

    # events with fewer than 3 jets have no trijet at all, so the first
    # Filter drops them before BestTrijet looks for one
    selected = (
        df.Filter("nJet >= 3")
        .Define("best", "BestTrijet(Jet_pt, Jet_eta, Jet_phi, Jet_mass, Jet_btag)")
        .Filter(f"best[2] > {MASS_LOW} && best[2] < {MASS_HIGH}")
        .Define("jet_pt", "Jet_pt")
        .Define("jet_eta", "Jet_eta")
        .Define("jet_phi", "Jet_phi")
        .Define("jet_mass", "Jet_mass")
        .Define("jet_btag", "Jet_btag")
        .Define("trijet_pt", "best[0]")
        .Define("trijet_mass", "best[2]")
    )
    # ak.from_rdataframe runs the event loop and hands back the selected
    # columns as one awkward array - jagged jet columns and all.
    skim = ak.from_rdataframe(
        selected,
        columns=(
            "jet_pt",
            "jet_eta",
            "jet_phi",
            "jet_mass",
            "jet_btag",
            "trijet_pt",
            "trijet_mass",
        ),
    )
    skim["dataset"] = ak.Array([dataset_name] * len(skim))
    return skim


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

    vr = VineReduce(
        processors={"skim": trijet_skimmer},
        input=datasets_path,
        # reads the coffea-shaped datasets json, as VineReduceCoffea does
        input_to_datasets=coffea_input_to_datasets,
        # the piece VineReduceCoffea would have supplied
        chunk_to_args=chunk_to_args,
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
