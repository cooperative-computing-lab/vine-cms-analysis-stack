"""Module 9's script - Module 2's scripts/vr_trijet_iterative_rdf.py over the
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

    python scripts/vr_trijet_failures_rdf.py 0      # aborts
    python scripts/vr_trijet_failures_rdf.py 0.3    # finishes, with a warning

RDataFrame flavor: the twin of scripts/vr_trijet_failures.py, with the same
datasets, the same VineReduce knobs, and the same numbers. The processor is
RDataFrame code, and plain VineReduce replaces VineReduceCoffea, so
chunk_to_args is spelled out here; the processor hands back hist.Hist
objects, as the coffea flavor does (see scripts/vr_trijet_iterative_rdf.py
for what each piece does).
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import sys

import hist

from vine_reduce import serialization
from vine_reduce.coffea import coffea_input_to_datasets, coffea_reducer
from vine_reduce.engine import VineReduce
from vine_reduce.local_distributor import LocalDistributor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BAD_FILE_NAME = "DOES_NOT_EXIST.root"


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
