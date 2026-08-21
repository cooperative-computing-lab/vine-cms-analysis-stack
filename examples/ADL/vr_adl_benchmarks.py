"""Runs all eight ADL benchmark queries (see processors.py) as one
VineReduceCoffea computation - one dataset, eight processors - over
synthetic NanoAOD-like data, via the TaskVine executor.

Running eight processors over one dataset in a single VineReduceCoffea
call, rather than one call per query, mirrors how
../ttBar/run_processor_with_vr.py drives its three lepton channels
(ee_chan/mm_chan/em_chan) through one `processors={...}` dict: each
(processor, dataset) pair gets its own independent Pipeline (chunking,
reduction, checkpointing) under the hood, so the eight queries below don't
interact with each other - they just happen to share one TaskVine run.

Every query gets its own checkpointed result under
results/adl_demo/<query>/, and default_reducer (see processors.py's module
docstring) merges each into either a single Hist (q1-q5, q7, q8) or a
dict of two Hists (q6) across chunks, with no custom reducer needed for
any of them.

The synthetic data (see write_test_data.py) is uncorrelated random
kinematics, not a physical event generator, so entry counts below are
sanity checks that each query's selection actually fires on a sizeable
fraction of events - not physics results. The probabilities were checked
against this data's ranges/means before picking EXPECTED_MIN_FRACTION
per query (e.g. Q5's dimuon mass window alone passes ~1/3 of random muon
pairs from this data's kinematics - see write_test_data.py).
"""

from __future__ import annotations

import glob
import os
import shutil

import ndcctools.taskvine as vine
import numpy as np

import processors
import write_test_data
from vine_reduce import serialization
from vine_reduce.coffea import VineReduceCoffea
from vine_reduce.taskvine_distributor import TaskVineDistributor

FILES_PER_DATASET = 3
CHUNKSIZE = 150
DATASET_NAME = "adl_demo"

PROCESSORS = {
    "q1": processors.q1,
    "q2": processors.q2,
    "q2_kin2d": processors.q2_kin2d,
    "q2_kin3d": processors.q2_kin3d,
    "q3": processors.q3,
    "q4": processors.q4,
    "q5": processors.q5,
    "q6": processors.q6,
    "q7": processors.q7,
    "q8": processors.q8,
}

# Least-selective fraction of total events each query is expected to fill
# at least one histogram entry for, given write_test_data.py's kinematics -
# checked empirically (see that file's module docstring), then rounded well
# below the observed rate so this isn't a flaky check. q1/q2/q2_kin2d/
# q2_kin3d/q2/q3 fill (at least) once per event unconditionally, so they're
# omitted here (checked separately, exactly).
EXPECTED_MIN_FRACTION = {
    "q4": 0.5,  # >=2 jets with pT>40, jet pT is uniform up to 200 - common.
    "q5": 0.1,  # >=2 muons (~80%) with an OS pair in [60, 120] GeV (~33%).
    "q6": 0.5,  # >=3 jets, mean 6/event - common.
    "q7": 0.1,  # some pT>30 jet surviving the lepton-cleaning cut - common.
    "q8": 0.3,  # >=3 leptons (mean 5/event) with a same-flavor OS pair.
}


def total_entries(result) -> float:
    """A query's result is either one Hist (most queries) or a dict of two
    (q6's trijetpt/maxbtag) - this returns the smaller of the two for q6, so
    the same EXPECTED_MIN_FRACTION check works for every query uniformly.
    flow=True: some histograms' fill values (e.g. q6's trijet pT, from
    uncorrelated synthetic jets) land outside the plotted range more often
    than in a real analysis - flow=True counts every fill, in-range or not.
    """
    if isinstance(result, dict):
        return min(h.sum(flow=True) for h in result.values())
    return result.sum(flow=True)


def load_result(results_dir, dataset_name, processor_name):
    """Final results land under results_dir/<dataset_name>/<processor_name>/
    as a single compressed, pickled file (name includes a random uuid, hence
    the glob). serialization.load reverses what the reducer wrote."""
    pattern = os.path.join(results_dir, dataset_name, processor_name, "*.pkl.zst")
    (result_file,) = glob.glob(pattern)
    return serialization.load(result_file)


def build_datasets(data_dir):
    """Builds the `input` dict VineReduceCoffea expects - see
    ../cortado/vr_cortado.py's build_datasets for the full shape
    explanation."""
    rng = np.random.default_rng()
    files = write_test_data.generate_dataset_files(data_dir, DATASET_NAME, FILES_PER_DATASET, rng)
    return {
        DATASET_NAME: {
            "metadata": {},
            "files": {
                path: {"object_path": "Events", "num_entries": num_events}
                for path, num_events in files.items()
            },
        }
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "data")
    results_dir = os.path.join(here, "results")
    checkpoint_dir = os.path.join(here, "checkpoints")

    # Fresh data every run, so stale results/checkpoints from a previous run
    # (over different random files) must not linger either.
    shutil.rmtree(results_dir, ignore_errors=True)
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    datasets = build_datasets(data_dir)
    total_events = sum(
        file_info["num_entries"] for file_info in datasets[DATASET_NAME]["files"].values()
    )

    # Same TaskVineDistributor + vine.Factory setup as ../cortado: one
    # local worker process, no cluster or separate vine_worker needed to run
    # this example standalone.
    distributor = TaskVineDistributor(
        port=0, resources_processor={"cores": 1}, resources_reducer={"cores": 1}
    )
    workers = vine.Factory(manager_host_port=f"localhost:{distributor.port}")
    workers.cores = 2
    workers.min_workers = 1
    workers.max_workers = 1

    with workers:
        # reducer defaults to VineReduceCoffea's own default_reducer, which
        # already knows how to sum Hists (and dicts of them, for q6) - see
        # processors.py's module docstring.
        vr = VineReduceCoffea(
            processors=PROCESSORS,
            input=datasets,
            chunksize=CHUNKSIZE,
            results_dir=results_dir,
            checkpoint_dir=checkpoint_dir,
            distributor=distributor,
            # q1..q8 are plain module-level functions in processors.py, not
            # closures/lambdas - cloudpickle serializes those *by reference*
            # (just the module + qualname), so the worker needs its own copy
            # of that file alongside the task to import them from.
            extra_files=[os.path.join(here, "processors.py")],
        )
        vr.compute()
    distributor.shutdown()

    results = {name: load_result(results_dir, DATASET_NAME, name) for name in PROCESSORS}
    for name, result in results.items():
        print(f"{name}: {total_entries(result):.0f} entries")

    # q1 fills events.MET.pt unconditionally, once per event: an exact
    # sanity check that every event actually made it through the pipeline.
    assert total_entries(results["q1"]) == total_events, (
        "expected q1 to fill exactly once per event"
    )
    for name, min_fraction in EXPECTED_MIN_FRACTION.items():
        entries = total_entries(results[name])
        assert entries >= min_fraction * total_events, (
            f"{name}: expected at least {min_fraction:.0%} of {total_events} events "
            f"to pass, got {entries:.0f}"
        )
    print(f"OK: all {len(PROCESSORS)} ADL benchmark queries ran over {total_events} events")


if __name__ == "__main__":
    main()
