"""Generates synthetic NanoAOD-like ROOT files for the trijet example.

Same "counter-branch" NanoAOD layout as ../cortado/write_test_data.py (see
that file for the full explanation of why uproot's mktree/counter_name/
field_name hooks are used), but with a single jagged "Jet" collection
{pt, eta, phi, mass, btag} instead of Electron/Muon - all Q6Processor
(see vr_trijet.py) reads off of `events.Jet`. `btag` is not a standard
NanoAOD branch name (real NanoAOD ships `Jet_btagCSVV2`/`Jet_btagDeepB`);
it matches the flat opendata-derived file the original coffea-benchmarks
Q6Processor was written against, which exposes the discriminant directly
as `Jet_btag`.
"""

from __future__ import annotations

import os

import awkward as ak
import numpy as np
import uproot


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


def write_root_file(
    path: str, num_events: int, jet_mean: float, rng: np.random.Generator
) -> None:
    """Writes one NanoAOD-shaped ROOT file with num_events events and a Jet
    collection (see _make_jets), under an "Events" tree."""
    # NanoAODSchema requires these three run-identification branches even
    # though nothing downstream in this example actually reads them.
    events = ak.Array(
        {
            "run": np.ones(num_events, dtype=np.uint32),
            "luminosityBlock": np.ones(num_events, dtype=np.uint32),
            "event": np.arange(num_events, dtype=np.int64),
            "Jet": _make_jets(num_events, jet_mean, rng),
        }
    )
    with uproot.recreate(path) as f:
        branch_types = {name: events[name].type for name in events.fields}
        f.mktree("Events", branch_types, counter_name=_counter_name, field_name=_field_name)
        f["Events"].extend({name: events[name] for name in events.fields})


def generate_dataset_files(
    data_dir: str,
    dataset_name: str,
    num_files: int,
    jet_mean: float,
    rng: np.random.Generator,
) -> dict[str, int]:
    """Writes num_files ROOT files for dataset_name under data_dir, each with
    a random event count in [300, 500). Returns {absolute_path:
    num_events} - build_datasets in vr_trijet.py wraps this into the
    "files" shape a coffea-preprocessed dataset expects."""
    dataset_dir = os.path.join(data_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    files = {}
    for i in range(num_files):
        num_events = int(rng.integers(300, 500))
        path = os.path.abspath(os.path.join(dataset_dir, f"file_{i}.root"))
        write_root_file(path, num_events, jet_mean, rng)
        files[path] = num_events
    return files
