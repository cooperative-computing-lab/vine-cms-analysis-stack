"""The eight ADL benchmark queries (Q1-Q8), translated from coffea's own
Q*Processor classes into vine_reduce processor functions.

Source: coffea's own
https://github.com/CoffeaTeam/coffea-benchmarks/blob/master/coffea-adl-benchmarks.py
(the "IRIS-HEP ADL benchmarks", a standard set of representative HEP
analysis queries used to compare column-analysis frameworks/backends).

Every Q*Processor there is a coffea.processor.ProcessorABC subclass whose
only real content is a `process(self, events)` method (`postprocess` is
always a no-op passthrough). VineReduceCoffea's own executor (see
vine_reduce/coffea.py's _make_executor) calls a processor as plain
`processor(events, **processor_args)` - no `self`, no ABC - so each
`process` body below drops in completely unchanged as a plain function;
only the class wrapper and the vestigial `postprocess` are gone. Nothing
in any of these bodies was rewritten for vine_reduce.

The other piece the originals relied on - merging every chunk's histogram
output back together - also needs no new code here:
VineReduceCoffea's default_reducer (ported from coffea's own accumulate()
helper) already knows how to sum two `Hist` objects, or two dicts of them
(as Q6 returns), with `+`. See ../trijet-turned-Q6's note in
vr_adl_benchmarks.py, or vine_reduce/coffea.py's default_reducer, for the
detail - no processor here needs a custom reducer.

q2_kin2d/q2_kin3d correspond to the original's Q2Kin2DProcessor/
Q2Kin3DProcessor (extra jet-kinematics histograms benchmarking 2D/3D fills,
not "Q2 part 2/3" physics queries).
"""

from __future__ import annotations

import awkward as ak
import hist
import numpy as np


def q1(events):
    """Plot the MET of all events."""
    return (
        hist.Hist.new.Reg(100, 0, 200, name="met", label="$E_{T}^{miss}$ [GeV]")
        .Double()
        .fill(events.MET.pt)
    )


def q2(events):
    """Plot the pT of all jets."""
    return (
        hist.Hist.new.Reg(100, 0, 200, name="ptj", label="Jet $p_{T}$ [GeV]")
        .Double()
        .fill(ak.flatten(events.Jet.pt))
    )


def q2_kin2d(events):
    """Plot the pT and eta of all jets."""
    return (
        hist.Hist.new.Reg(100, 0, 200, name="ptj", label="Jet $p_{T}$ [GeV]")
        .Reg(100, -5, 5, name="etaj", label=r"Jet $\eta$")
        .Double()
        .fill(ak.flatten(events.Jet.pt), ak.flatten(events.Jet.eta))
    )


def q2_kin3d(events):
    """Plot the pT, eta, and phi of all jets."""
    return (
        hist.Hist.new.Reg(100, 0, 200, name="ptj", label="Jet $p_{T}$ [GeV]")
        .Reg(100, -5, 5, name="etaj", label=r"Jet $\eta$")
        .Reg(100, -np.pi, np.pi, name="phij", label=r"Jet $\phi$")
        .Double()
        .fill(
            ak.flatten(events.Jet.pt),
            ak.flatten(events.Jet.eta),
            ak.flatten(events.Jet.phi),
        )
    )


def q3(events):
    """Plot the pT of jets with |eta| < 1."""
    return (
        hist.Hist.new.Reg(100, 0, 200, name="ptj", label="Jet $p_{T}$ [GeV]")
        .Double()
        .fill(ak.flatten(events.Jet[abs(events.Jet.eta) < 1].pt))
    )


def q4(events):
    """Plot the MET of events that have at least two jets with pT > 40 GeV."""
    has2jets = ak.sum(events.Jet.pt > 40, axis=1) >= 2
    return (
        hist.Hist.new.Reg(100, 0, 200, name="met", label="$E_{T}^{miss}$ [GeV]")
        .Double()
        .fill(events[has2jets].MET.pt)
    )


def q5(events):
    """Plot the MET of events that have an opposite-charge muon pair with an
    invariant mass between 60 and 120 GeV."""
    # The original benchmark accesses the unnamed pair fields as
    # `mupair.slot0`/`mupair.slot1`, an awkward-array convenience no longer
    # present in current awkward (2.x) - naming the fields explicitly via
    # `fields=` below is the same combinations() call, just addressed the
    # way current awkward requires.
    mupair = ak.combinations(events.Muon, 2, fields=["mu1", "mu2"])
    with np.errstate(invalid="ignore"):
        pairmass = (mupair.mu1 + mupair.mu2).mass
    goodevent = ak.any(
        (pairmass > 60)
        & (pairmass < 120)
        & (mupair.mu1.charge == -mupair.mu2.charge),
        axis=1,
    )
    return (
        hist.Hist.new.Reg(100, 0, 200, name="met", label="$E_{T}^{miss}$ [GeV]")
        .Double()
        .fill(events[goodevent].MET.pt)
    )


def q6(events):
    """For events with at least three jets, plot the pT of the trijet
    four-momentum that has the invariant mass closest to 172.5 GeV in each
    event and plot the maximum b-tagging discriminant value among the jets
    in this trijet."""
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


def q7(events):
    """Plot the scalar sum in each event of the pT of jets with pT > 30 GeV
    that are not within 0.4 in deltaR of any light lepton with pT > 10 GeV."""
    cleanjets = events.Jet[
        ak.all(events.Jet.metric_table(events.Muon[events.Muon.pt > 10]) >= 0.4, axis=2)
        & ak.all(
            events.Jet.metric_table(events.Electron[events.Electron.pt > 10]) >= 0.4,
            axis=2,
        )
        & (events.Jet.pt > 30)
    ]
    return (
        hist.Hist.new.Reg(100, 0, 200, name="sumjetpt", label=r"Jet $\sum p_{T}$ [GeV]")
        .Double()
        .fill(ak.sum(cleanjets.pt, axis=1))
    )


def q8(events):
    """For events with at least three light leptons and a same-flavor
    opposite-charge light lepton pair, find such a pair that has the
    invariant mass closest to 91.2 GeV in each event and plot the
    transverse mass of the system consisting of the missing transverse
    momentum and the highest-pT light lepton not in this pair."""
    events["Electron", "pdgId"] = -11 * events.Electron.charge
    events["Muon", "pdgId"] = -13 * events.Muon.charge
    # The original benchmark concatenates events.Electron and events.Muon
    # directly. In current awkward/vector, concatenating two NanoAOD
    # collections whose record types differ (Electron and Muon carry
    # different fields/behavior even after the pdgId assignments above)
    # produces a union array that vector's delta_phi (used below) can't
    # handle - zipping both down to one common {pt, eta, phi, mass, charge,
    # pdgId} record/behavior first keeps the same physics, addressed the
    # way current awkward requires.
    def _as_lepton(collection):
        return ak.zip(
            {field: getattr(collection, field) for field in ("pt", "eta", "phi", "mass", "charge", "pdgId")},
            with_name="PtEtaPhiMLorentzVector",
            behavior=collection.behavior,
        )

    events["leptons"] = ak.concatenate(
        [_as_lepton(events.Electron), _as_lepton(events.Muon)], axis=1
    )
    events = events[ak.num(events.leptons) >= 3]

    pair = ak.argcombinations(events.leptons, 2, fields=["l1", "l2"])
    pair = pair[(events.leptons[pair.l1].pdgId == -events.leptons[pair.l2].pdgId)]
    with np.errstate(invalid="ignore"):
        pair = pair[
            ak.singletons(
                ak.argmin(
                    abs((events.leptons[pair.l1] + events.leptons[pair.l2]).mass - 91.2),
                    axis=1,
                )
            )
        ]
    events = events[ak.num(pair) > 0]
    pair = pair[ak.num(pair) > 0][:, 0]

    l3 = ak.local_index(events.leptons)
    l3 = l3[(l3 != pair.l1) & (l3 != pair.l2)]
    l3 = l3[ak.argmax(events.leptons[l3].pt, axis=1, keepdims=True)]
    l3 = events.leptons[l3][:, 0]

    mt = np.sqrt(2 * l3.pt * events.MET.pt * (1 - np.cos(events.MET.delta_phi(l3))))
    return (
        hist.Hist.new.Reg(
            100, 0, 200, name="mt", label=r"$\ell$-MET transverse mass [GeV]"
        )
        .Double()
        .fill(mt)
    )
