"""Module 4's script - the same ADL benchmark Q6 trijet analysis, now over
real CMS Open Data (data/datasets_xrootd.json: 15 files of Run2012B_SingleMu
in 5 datasets, read over xrootd from cmsxrootd.crc.nd.edu), with its
processor/reducer calls run on TaskVine workers in the HTCondor pool instead
of on this machine.

Three things change from Module 3's scripts/vr_trijet_taskvine_rdf.py; the
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

RDataFrame flavor: the twin of scripts/vr_trijet_condor.py, with the same
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

import hist

from vine_reduce import get_environment, serialization
from vine_reduce.coffea import coffea_input_to_datasets, coffea_reducer
from vine_reduce.engine import VineReduce
from vine_reduce.taskvine_distributor import TaskVineDistributor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MANAGER_NAME = f"{getpass.getuser()}-trijet-tutorial"


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

    vr = VineReduce(
        processors={"trijet": trijet_processor},
        input=datasets_path,
        # reads the coffea-shaped datasets json, as VineReduceCoffea does
        input_to_datasets=coffea_input_to_datasets,
        # the piece VineReduceCoffea would have supplied
        chunk_to_args=chunk_to_args,
        # adds up dicts of hist.Hist, as VineReduceCoffea does
        reducer=coffea_reducer,
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
