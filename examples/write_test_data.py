"""Generates synthetic NanoAOD-like ROOT files for this repo's examples
(cortado, ADL, trijet), plus a JSON manifest describing them.

This is a script, not a module: nothing here is meant to be imported.
Each example's own vr_*.py calls it as a subprocess, and only when its
data/ directory doesn't already have a manifest (see ensure_datasets() in
each example) - so a data/ directory, once generated, is reused by later
runs instead of being rebuilt from scratch every time. Delete an example's
data/ directory (or the datasets.json inside it) to force regeneration.

Shared by all three examples since their own write_test_data.py files
used to be near-identical copies of each other. Every generated file gets
the union of fields any of the three examples read: a jagged Jet
collection (pt/eta/phi/mass/btag), jagged Muon/Electron collections
(pt/eta/phi/mass/charge), and a flat MET (pt/phi) - the "counter-branch"
NanoAOD layout coffea's NanoAODSchema expects, built via uproot's
mktree/counter_name/field_name hooks. An example that only reads one of
these collections (trijet: Jet; cortado: Muon/Electron) simply ignores
the rest.

--num-datasets datasets are generated, named dataset_0 .. dataset_{n-1}
unless --dataset-names overrides that (cortado uses this to keep its
"signal"/"background" names). --jet-mean/--muon-mean/--electron-mean each
take either one value (applied to every dataset) or one value per
dataset - cortado uses two different --muon-mean/--electron-mean values
to give "signal" a higher per-event lepton count than "background", which
is the whole point of that example's skim.

Writes {data-dir}/datasets.json, shaped exactly like the "input" dict
VineReduceCoffea expects (coffea's own preprocessed-dataset shape: name ->
metadata + files, each file -> object_path/num_entries) - callers load it
directly, with no Python-side reshaping needed.
"""

from __future__ import annotations

import argparse
import json
import os

import awkward as ak
import numpy as np
import uproot

LEPTON_MASS = 0.10566  # muon mass (GeV); reused for electrons too - the
# selections in these examples only care about pt/eta/phi/charge.


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
    Electron, each with their own mean."""
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


def write_root_file(
    path: str,
    num_events: int,
    jet_mean: float,
    muon_mean: float,
    electron_mean: float,
    rng: np.random.Generator,
) -> None:
    """Writes one NanoAOD-shaped ROOT file with num_events events - a Jet,
    a Muon, an Electron collection, and a flat MET - under an "Events"
    tree, the layout coffea's NanoAODSchema (and VineReduceCoffea) expect."""
    # NanoAODSchema requires these three run-identification branches even
    # though nothing downstream in these examples actually reads them.
    events = ak.Array(
        {
            "run": np.ones(num_events, dtype=np.uint32),
            "luminosityBlock": np.ones(num_events, dtype=np.uint32),
            "event": np.arange(num_events, dtype=np.int64),
            "Jet": _make_jets(num_events, jet_mean, rng),
            "Muon": _make_leptons(num_events, muon_mean, rng),
            "Electron": _make_leptons(num_events, electron_mean, rng),
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


def _generate_dataset_files(
    dataset_dir: str,
    num_files: int,
    jet_mean: float,
    muon_mean: float,
    electron_mean: float,
    rng: np.random.Generator,
) -> dict[str, int]:
    """Writes num_files ROOT files under dataset_dir, each with a random
    event count in [300, 500). Returns {absolute_path: num_events}."""
    files = {}
    for i in range(num_files):
        num_events = int(rng.integers(300, 500))
        path = os.path.abspath(os.path.join(dataset_dir, f"file_{i}.root"))
        write_root_file(path, num_events, jet_mean, muon_mean, electron_mean, rng)
        files[path] = num_events
    return files


def _broadcast(values: list[float], num_datasets: int, flag_name: str) -> list[float]:
    if len(values) == 1:
        return list(values) * num_datasets
    if len(values) == num_datasets:
        return list(values)
    raise ValueError(
        f"--{flag_name} must be given once (applied to every dataset) or once "
        f"per dataset ({num_datasets} values), got {len(values)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--num-datasets", type=int, default=1)
    parser.add_argument(
        "--dataset-names",
        nargs="+",
        default=None,
        help="One name per dataset; defaults to dataset_0 .. dataset_{n-1}.",
    )
    parser.add_argument("--num-files", type=int, default=3, help="ROOT files per dataset.")
    parser.add_argument("--jet-mean", type=float, nargs="+", default=[6.0])
    parser.add_argument("--muon-mean", type=float, nargs="+", default=[3.0])
    parser.add_argument("--electron-mean", type=float, nargs="+", default=[2.0])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.dataset_names is not None:
        if len(args.dataset_names) != args.num_datasets:
            parser.error(
                f"--dataset-names must give exactly --num-datasets "
                f"({args.num_datasets}) names, got {len(args.dataset_names)}"
            )
        dataset_names = args.dataset_names
    else:
        dataset_names = [f"dataset_{i}" for i in range(args.num_datasets)]

    jet_means = _broadcast(args.jet_mean, args.num_datasets, "jet-mean")
    muon_means = _broadcast(args.muon_mean, args.num_datasets, "muon-mean")
    electron_means = _broadcast(args.electron_mean, args.num_datasets, "electron-mean")

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.data_dir, exist_ok=True)

    manifest = {}
    for name, jet_mean, muon_mean, electron_mean in zip(
        dataset_names, jet_means, muon_means, electron_means
    ):
        dataset_dir = os.path.join(args.data_dir, name)
        os.makedirs(dataset_dir, exist_ok=True)
        files = _generate_dataset_files(
            dataset_dir, args.num_files, jet_mean, muon_mean, electron_mean, rng
        )
        manifest[name] = {
            "metadata": {},
            "files": {
                path: {"object_path": "Events", "num_entries": num_events}
                for path, num_events in files.items()
            },
        }

    manifest_path = os.path.join(args.data_dir, "datasets.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
