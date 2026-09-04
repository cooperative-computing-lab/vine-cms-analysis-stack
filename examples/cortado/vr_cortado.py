"""HEP skim tutorial: vine_reduce.VineReduceCoffea over synthetic NanoAOD-like
data, via the TaskVine executor.

Adapted from the cortado example in
https://github.com/cooperative-computing-lab/cortado: a coffea
processor skims events down to the ones with at least four leptons, and a
custom reducer concatenates the surviving events from every chunk into one
growing awkward array per dataset, exactly like cortado's own
"accumulator" hook. Dropped relative to the original: ROOT output (this
example writes plain parquet, see make_result_postprocess below), on-site
condor/xrootd
config (samples here are local synthetic files, not a real CMS dataset), and
periodic checkpoint-triggered writes to disk (VineReduce already checkpoints
intermediate reduce results under checkpoint_dir for restart, so nothing
extra is needed for that).

VineReduceCoffea (src/vine_reduce/coffea.py) is a VineReduce specialization
for coffea: given a dataset in coffea's own preprocessed-file shape (name ->
metadata + files, each file carrying object_path/num_entries - see
build_datasets), it takes care of reading a Chunk as NanoEvents and
materializing a processor's awkward-array output before it's sent back over
the wire. Chunking, checkpointing, and restart are otherwise inherited
unchanged from VineReduce (see vine_reduce's own
examples/quick_start/quick_start.py for a line-by-line tour of those
mechanics with plain Python types instead of awkward arrays).

Concretely, in this example:

- ../write_test_data.py (shared with the other examples) generates two
  datasets ("signal" and "background") of three NanoAOD-shaped ROOT files
  each, under examples/cortado/data/, the first time this runs (see
  ensure_datasets() below - later runs reuse the same files). "signal"
  files average more leptons per event than "background" ones (see
  DATASET_LEPTON_MEANS), so the skim below should keep a noticeably larger
  fraction of "signal" events.
- skimmer (the map step) keeps only events with >=4 reconstructed leptons
  (electrons + muons combined) - a placeholder for a real analysis'
  selection - and tags the survivors with their dataset name so
  result_postprocess below knows where to write them.
- accumulate_skims (the reduce step) concatenates the surviving events from
  two chunks/groups into one awkward array, replacing VineReduce's default
  `a += b` reducer (which doesn't know how to concatenate awkward arrays).
- result_postprocess (see make_result_postprocess) runs remotely, once per
  final reduction group, and writes that group's surviving events straight
  to parquet under results_dir/skim_4lep/<dataset_name>.<uuid>.parquet -
  the form a downstream analysis step would actually want. It returns just
  the row count, which is what the framework pickles to
  results_dir/<dataset_name>/skim_4lep/*.pkl.zst, so main() can report and
  check the signal-vs-background asymmetry the input data was built to
  produce without ever reopening the parquet fragments.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import uuid

import awkward as ak
import ndcctools.taskvine as vine

from vine_reduce.coffea import VineReduceCoffea
from vine_reduce.serialization import load as load_result
from vine_reduce.taskvine_distributor import TaskVineDistributor

CHUNKSIZE = 150
FILES_PER_DATASET = 10
DATASET_LEPTON_MEANS = {"signal": 3.0, "background": 1.0}


def ensure_datasets(data_dir, *script_args):
    """Generates data_dir's ROOT files + datasets.json manifest by running
    ../write_test_data.py as a subprocess - but only the first time; once
    datasets.json exists, later runs reuse the same synthetic data instead
    of paying the generation cost again. Returns the loaded manifest,
    already shaped as the `input` dict VineReduceCoffea expects (see that
    script's module docstring)."""
    write_test_data = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "write_test_data.py"
    )
    manifest_path = os.path.join(data_dir, "datasets.json")
    if not os.path.exists(manifest_path):
        subprocess.run(
            [sys.executable, write_test_data, "--data-dir", data_dir, *script_args],
            check=True,
        )
    with open(manifest_path) as f:
        return json.load(f)


def skimmer(events):
    """Runs remotely, once per Chunk of NanoEvents (VineReduceCoffea's
    chunk_to_args + executor take care of turning a Chunk into `events`
    and materializing this function's return value). Placeholder ">=4
    leptons" selection, echoing a typical multi-lepton search skim.

    Tags every surviving event with its dataset name (from
    events.metadata, set by chunk_to_args from the dataset's "dataset"
    metadata key) - result_postprocess below only ever sees the
    accumulated array itself, not which dataset produced it, and needs
    this to know where to write its parquet output."""
    num_leptons = ak.num(events.Electron) + ak.num(events.Muon)
    skim = events[num_leptons >= 4]
    return ak.with_field(skim, events.metadata["dataset"], "_dataset")


def accumulate_skims(a, b):
    """Reducer for the skimmer processor: two chunks' (or groups')
    surviving events are just concatenated into one, larger, awkward
    array. Runs remotely, like the base reducer it replaces."""
    return ak.concatenate([a, b], axis=0)


def make_result_postprocess(results_dir):
    """Builds the result_postprocess callback: runs remotely, once per
    final reduction group, and writes that group's surviving events
    straight to parquet under
    results_dir/skim_4lep/<dataset_name>.<uuid>.parquet, dropping the
    _dataset tag skimmer added (only needed to route this write).

    The uuid suffix (rather than a plain <dataset_name>.parquet) is
    needed because more than one final result for the same dataset can
    complete concurrently - so each writes its own fragment instead of
    racing to overwrite a shared file. main() sums row counts across a
    dataset's fragments when reporting.

    Returns just the row count rather than the skim itself: whatever this
    returns gets pickled by the framework to its own per-group results
    file (results_dir/<dataset_name>/skim_4lep/*.pkl.zst) - that's the
    count count_skim_rows below reads back, so main() never has to reopen
    the parquet fragments just to learn how many rows they hold."""
    def result_postprocess(skim):
        if len(skim) == 0:
            return 0
        dataset_name = str(skim["_dataset"][0])
        out_dir = os.path.join(results_dir, "skim_4lep")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{dataset_name}.{uuid.uuid4().hex}.parquet")
        fields = [f for f in skim.fields if f != "_dataset"]
        ak.to_parquet(skim[fields], out_path)
        return len(skim)
    return result_postprocess


def count_skim_rows(results_dir, dataset_name):
    """Sums result_postprocess's returned row counts (see
    make_result_postprocess) across every final result vine_reduce wrote
    for this dataset - those pickles already are the counts, so nothing
    here needs to touch the parquet fragments."""
    pattern = os.path.join(results_dir, dataset_name, "skim_4lep", "*.pkl.zst")
    return sum(load_result(f) for f in glob.glob(pattern))


def build_datasets(data_dir):
    """Builds the `input` dict VineReduceCoffea expects: coffea's own
    preprocessed-dataset shape, one entry per dataset, each with a
    "metadata" dict and a "files" dict mapping each file's path to
    {"object_path": ..., "num_entries": ...} - what
    coffea.dataset_tools.preprocess() itself produces, and what
    coffea_input_to_datasets (VineReduceCoffea's default input_to_datasets)
    knows how to read. Generated (or reused) via ../write_test_data.py -
    DATASET_LEPTON_MEANS' values are passed as both --muon-mean and
    --electron-mean, one per dataset, in the same order as its keys."""
    dataset_names = list(DATASET_LEPTON_MEANS.keys())
    lepton_means = [str(mean) for mean in DATASET_LEPTON_MEANS.values()]
    return ensure_datasets(
        data_dir,
        "--dataset-names", *dataset_names,
        "--num-datasets", str(len(dataset_names)),
        "--num-files", str(FILES_PER_DATASET),
        "--min-events", "1000",
        "--max-events", "5000",
        "--muon-mean", *lepton_means,
        "--electron-mean", *lepton_means,
    )


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

    # Same TaskVineDistributor + vine.Factory setup as quick_start.py: one
    # local worker process, no cluster or separate vine_worker needed to run
    # this example standalone.
    distributor = TaskVineDistributor(
        port=0,
        resources_processor={"cores": 1},
        resources_reducer={"cores": 1},
        checkpoint_dir=checkpoint_dir,
    )
    workers = vine.Factory(manager_host_port=f"localhost:{distributor.port}")
    workers.cores = 2
    workers.min_workers = 1
    workers.max_workers = 1

    with workers:
        vr = VineReduceCoffea(
            processors={"skim_4lep": skimmer},
            input=datasets,
            reducer=accumulate_skims,
            chunksize=CHUNKSIZE,
            results_dir=results_dir,
            distributor=distributor,
            result_postprocess=make_result_postprocess(results_dir),
        )
        vr.compute()
    distributor.shutdown()

    # Every dataset's parquet skim was already written by
    # result_postprocess during compute(); just report how many events
    # survived, straight from the row counts it also returned.
    counts = {}
    for dataset_name in datasets:
        counts[dataset_name] = count_skim_rows(results_dir, dataset_name)
        print(f"{dataset_name}: {counts[dataset_name]} events pass the >=4-lepton skim")

    # Sanity check tied to how the input data was generated (see
    # DATASET_LEPTON_MEANS): "signal" has a higher mean lepton count than
    # "background", so it should pass the skim more often.
    assert (
        counts["signal"] > counts["background"]
    ), "expected signal to pass the skim more often than background"
    print("OK: signal passes the >=4-lepton skim more often than background")


if __name__ == "__main__":
    main()
