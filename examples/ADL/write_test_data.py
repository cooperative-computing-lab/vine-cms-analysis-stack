"""Generates synthetic NanoAOD-like ROOT files for the ADL benchmark
examples (see processors.py).

Same "counter-branch" NanoAOD layout as ../cortado/write_test_data.py (see
that file for the full explanation of why uproot's mktree/counter_name/
field_name hooks are used): one file gets everything all eight Q*
processors, combined, read off of - jagged Jet/Muon/Electron collections
plus a flat, per-event MET - so one synthetic dataset exercises every
query in processors.py, rather than one dataset per query.

The kinematics below are deliberately *uncorrelated* random draws, not a
physically realistic event generator (no momentum conservation, no actual
recoil between MET and jets, no true resonance decays for the muon/electron
pairs Q5/Q8 look for). That's enough to exercise every code path with
non-trivial results - including selections as tight as Q5/Q8's dimuon or
dilepton mass window - without pulling in an actual physics event
generator; see vr_adl_benchmarks.py's docstring for how each query's
selection rate was sanity-checked against this data before picking these
ranges/means.
"""

from __future__ import annotations

import os

import awkward as ak
import numpy as np
import uproot

# Muon mass (GeV); reused for electrons too since Q5/Q8 only care about the
# leptons' pt/eta/phi/charge, and the mass difference is negligible for a
# synthetic demo.
LEPTON_MASS = 0.10566


def _field_name(outer: str, inner: str) -> str:
    return inner if outer == "" else f"{outer}_{inner}"


def _counter_name(counted: str) -> str:
    return f"n{counted}"


def _make_jets(num_events: int, jet_mean: float, rng: np.random.Generator) -> ak.Array:
    """A jagged {pt, eta, phi, mass, btag} collection, `jet_mean` jets per
    event on average (Poisson), each with plausible-looking kinematics and
    a uniform[0, 1) b-tag discriminant."""
    counts = rng.poisson(jet_mean, size=num_events)
    total = int(counts.sum())
    flat = ak.Array(
        {
            "pt": rng.uniform(20.0, 200.0, size=total).astype(np.float32),
            "eta": rng.uniform(-2.5, 2.5, size=total).astype(np.float32),
            "phi": rng.uniform(-np.pi, np.pi, size=total).astype(np.float32),
            "mass": rng.uniform(0.0, 20.0, size=total).astype(np.float32),
            "btag": rng.uniform(0.0, 1.0, size=total).astype(np.float32),
        }
    )
    return ak.unflatten(flat, counts)


def _make_leptons(num_events: int, lepton_mean: float, rng: np.random.Generator) -> ak.Array:
    """A jagged {pt, eta, phi, mass, charge} collection, `lepton_mean`
    leptons per event on average (Poisson) - used for both Muon and
    Electron."""
    counts = rng.poisson(lepton_mean, size=num_events)
    total = int(counts.sum())
    flat = ak.Array(
        {
            "pt": rng.uniform(10.0, 100.0, size=total).astype(np.float32),
            "eta": rng.uniform(-2.5, 2.5, size=total).astype(np.float32),
            "phi": rng.uniform(-np.pi, np.pi, size=total).astype(np.float32),
            "mass": np.full(total, LEPTON_MASS, dtype=np.float32),
            "charge": rng.choice(np.array([-1, 1], dtype=np.int32), size=total),
        }
    )
    return ak.unflatten(flat, counts)


def write_root_file(path: str, num_events: int, rng: np.random.Generator) -> None:
    """Writes one NanoAOD-shaped ROOT file with num_events events: a Jet
    collection (mean 6/event, enough for Q4/Q6's >=2/>=3-jet cuts to fire
    often), a Muon and an Electron collection (mean 3 and 2/event - enough
    for Q5's dimuon and Q8's >=3-lepton cuts to fire often), and a flat MET
    - under an "Events" tree."""
    # NanoAODSchema requires these three run-identification branches even
    # though nothing downstream in this example actually reads them.
    events = ak.Array(
        {
            "run": np.ones(num_events, dtype=np.uint32),
            "luminosityBlock": np.ones(num_events, dtype=np.uint32),
            "event": np.arange(num_events, dtype=np.int64),
            "Jet": _make_jets(num_events, 6.0, rng),
            "Muon": _make_leptons(num_events, 3.0, rng),
            "Electron": _make_leptons(num_events, 2.0, rng),
            "MET": ak.Array(
                {
                    "pt": rng.uniform(0.0, 150.0, size=num_events).astype(np.float32),
                    "phi": rng.uniform(-np.pi, np.pi, size=num_events).astype(np.float32),
                }
            ),
        }
    )
    with uproot.recreate(path) as f:
        branch_types = {name: events[name].type for name in events.fields}
        f.mktree("Events", branch_types, counter_name=_counter_name, field_name=_field_name)
        f["Events"].extend({name: events[name] for name in events.fields})


def generate_dataset_files(
    data_dir: str, dataset_name: str, num_files: int, rng: np.random.Generator
) -> dict[str, int]:
    """Writes num_files ROOT files for dataset_name under data_dir, each with
    a random event count in [300, 500). Returns {absolute_path:
    num_events} - build_datasets in vr_adl_benchmarks.py wraps this into the
    "files" shape a coffea-preprocessed dataset expects."""
    dataset_dir = os.path.join(data_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    files = {}
    for i in range(num_files):
        num_events = int(rng.integers(300, 500))
        path = os.path.abspath(os.path.join(dataset_dir, f"file_{i}.root"))
        write_root_file(path, num_events, rng)
        files[path] = num_events
    return files
