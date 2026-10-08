"""Module 5's script - Module 4's scripts/vr_trijet_condor_rdf.py with the
three knobs this module is about pulled out as command-line arguments:

    python scripts/vr_trijet_resources_rdf.py CHUNKSIZE MEMORY_MB [REDUCTION_SIZE]

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

RDataFrame flavor: the twin of scripts/vr_trijet_resources.py, with the same
datasets, the same VineReduce knobs, and the same numbers. The processor is
RDataFrame code, and plain VineReduce replaces VineReduceCoffea, so
chunk_to_args is spelled out here; the processor hands back hist.Hist
objects, as the coffea flavor does (see scripts/vr_trijet_iterative_rdf.py
for what each piece does).
"""

from __future__ import annotations

import getpass
import glob
import json
import os
import shutil
import sys

import hist

from vine_reduce import get_environment, serialization
from vine_reduce.coffea import coffea_input_to_datasets, coffea_reducer
from vine_reduce.engine import VineReduce
from vine_reduce.taskvine_distributor import TaskVineDistributor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MANAGER_NAME = f"{getpass.getuser()}-trijet-tutorial"

DATASET = "SingleMu_Run2012B_194790-195398"


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
    """Runs in a local worker subprocess, once per chunk, before the
    processor. VineReduceCoffea would hand the processor NanoEvents here; we
    hand it an RDataFrame over just this chunk's entries [start, stop)."""
    import ROOT

    return ROOT.RDataFrame("Events", chunk.url).Range(chunk.start, chunk.stop)


def trijet_processor(df):
    """Runs in a local worker subprocess, once per chunk - one RDataFrame in,
    any picklable object out (here, a dict of two hist.Hists)."""
    import ROOT

    ROOT.gInterpreter.Declare(TRIJET_CPP)

    best = (
        df.Filter("nJet >= 3")
        .Define("best", "BestTrijet(Jet_pt, Jet_eta, Jet_phi, Jet_mass, Jet_btag)")
        .Define("trijet_pt", "best[0]")
        .Define("max_btag", "best[1]")
    )
    trijetpt = best.Histo1D(
        ("trijetpt", "Trijet p_{T};Trijet p_{T} [GeV];Events", 100, 0, 200), "trijet_pt"
    )
    maxbtag = best.Histo1D(
        ("maxbtag", "Max jet b-tag;Max jet b-tag score;Events", 100, 0, 1), "max_btag"
    )
    # RDataFrame is lazy: nothing has been read yet. The first GetValue()
    # runs one pass over the chunk that fills both histograms.
    return {
        "trijetpt": to_hist(trijetpt.GetValue()),
        "maxbtag": to_hist(maxbtag.GetValue()),
    }


def to_hist(th1):
    """Copies a ROOT TH1 into a hist.Hist, underflow and overflow bins
    included. A ROOT histogram that has been unpickled in a fresh process
    cannot be pickled again until that process has made a ROOT histogram of
    its own, and every reducer call is exactly such a process; plain
    hist.Hist objects have no such trouble, and the coffea reducer already
    knows how to add them."""
    nbins = th1.GetNbinsX()
    axis = th1.GetXaxis()
    out = hist.Hist.new.Reg(
        nbins, axis.GetXmin(), axis.GetXmax(), name=th1.GetName(), label=axis.GetTitle()
    ).Double()
    out.view(flow=True)[:] = [th1.GetBinContent(i) for i in range(nbins + 2)]
    return out


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

    vr = VineReduce(
        processors={"trijet": trijet_processor},
        input=datasets,
        # reads the coffea-shaped datasets json, as VineReduceCoffea does
        input_to_datasets=coffea_input_to_datasets,
        # the piece VineReduceCoffea would have supplied
        chunk_to_args=chunk_to_args,
        # adds up dicts of hist.Hist, as VineReduceCoffea does
        reducer=coffea_reducer,
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
