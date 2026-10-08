"""Module 2's script, RDataFrame flavor - the ADL benchmark Q6 trijet
analysis (https://github.com/CoffeaTeam/coffea-benchmarks/blob/master/coffea-adl-benchmarks.py:
for events with at least three jets, plot the pT of the trijet four-momentum
whose invariant mass is closest to 172.5 GeV, the top quark mass, and the
maximum b-tag discriminant among that trijet's three jets), written with
ROOT's RDataFrame instead of coffea, and run through vine_reduce's
LocalDistributor: a plain ProcessPoolExecutor-backed distributor that needs
no cluster, no vine_factory/vine_worker, and no environment packaging,
since worker subprocesses already share this process's filesystem and
Python env.

It is the twin of scripts/vr_trijet_iterative.py (the coffea flavor), over
the same data/datasets_vast.json, and prints the same entry counts.
Plain VineReduce replaces VineReduceCoffea, which means one piece that
VineReduceCoffea supplies for you is written out here:

- chunk_to_args: turns a chunk (a file plus an entry range) into an
  RDataFrame over exactly those entries.

The processor then hands back plain hist.Hist objects, like the coffea
flavor does, so the coffea reducer, the result files, and load_result are
the same too (see to_hist for why the ROOT histograms are converted).

Module 3 runs the exact same trijet_processor through a real TaskVine
factory/worker instead - only the `distributor=` argument changes.
"""

from __future__ import annotations

import glob
import json
import os
import shutil

import hist

from vine_reduce import serialization
from vine_reduce.coffea import coffea_input_to_datasets, coffea_reducer
from vine_reduce.engine import VineReduce
from vine_reduce.local_distributor import LocalDistributor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    """Final results land under results_dir/<dataset_name>/<processor_name>/
    as a single compressed, pickled file (name includes a random uuid, hence
    the glob). serialization.load reverses what the reducer wrote."""
    pattern = os.path.join(results_dir, dataset_name, processor_name, "*.pkl.zst")
    (result_file,) = glob.glob(pattern)
    return serialization.load(result_file)


def main():
    datasets_path = os.path.join(REPO_ROOT, "data", "datasets_vast.json")
    results_dir = os.path.join(REPO_ROOT, "results")
    checkpoint_dir = os.path.join(REPO_ROOT, "checkpoints")

    # Fresh run every time, so stale results/checkpoints don't linger.
    shutil.rmtree(results_dir, ignore_errors=True)
    shutil.rmtree(checkpoint_dir, ignore_errors=True)

    # The datasets: each a name plus its file list - VineReduce accepts a path
    # to a json file holding them directly, no need to load them into a dict
    # first.
    distributor = LocalDistributor(max_workers=2)

    vr = VineReduce(
        # one Pipeline per (processor, dataset) pair - 1 processor x 5 datasets
        processors={"trijet": trijet_processor},
        input=datasets_path,
        # where processor/reducer calls actually run
        distributor=distributor,
        results_dir=results_dir,
        checkpoint_dir=checkpoint_dir,
        # events per processor call - ~90 chunks over today's ~3.7M events
        chunksize=50_000,
        # reads the coffea-shaped datasets json, as VineReduceCoffea does
        input_to_datasets=coffea_input_to_datasets,
        # the piece VineReduceCoffea would have supplied
        chunk_to_args=chunk_to_args,
        # adds up dicts of hist.Hist, as VineReduceCoffea does
        reducer=coffea_reducer,
    )

    with distributor:
        vr.compute()

    with open(datasets_path) as f:
        dataset_names = list(json.load(f))

    # one line per dataset, then the total over all of them
    total_pt = total_btag = 0
    for name in dataset_names:
        result = load_result(results_dir, name, "trijet")
        # flow=True counts every fill, in-range or not: a trijet pT above the
        # [0, 200) GeV histogram range lands in the overflow bin, which plain
        # .sum() would leave out
        pt = result["trijetpt"].sum(flow=True)
        btag = result["maxbtag"].sum(flow=True)
        print(f"{name}: {pt:.0f} entries (trijet pT), {btag:.0f} entries (max b-tag)")
        total_pt += pt
        total_btag += btag
    print(f"trijet pT histogram, all datasets: {total_pt:.0f} entries")
    print(f"max b-tag histogram, all datasets: {total_btag:.0f} entries")


if __name__ == "__main__":
    main()
