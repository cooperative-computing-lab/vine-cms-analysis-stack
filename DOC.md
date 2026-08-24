# Vine CMS Analysis Stack

A reference stack for running CMS analysis
workflows with [Coffea](https://github.com/scikit-hep/coffea) on top of
[TaskVine](https://cctools.readthedocs.io/en/stable/taskvine), orchestrated
by [`vine_reduce`](https://github.com/cooperative-computing-lab/vine_reduce).
Nothing here depends on Notre Dame resources, so it should work anywhere
there's a Python environment and either local cores or a batch cluster
(HTCondor, SLURM, SGE, ...) to point workers at.

## Shape of the stack

The stack always has the same two-sided shape: physics code, `vine_reduce`,
and a distributor run once, locally; the executor is what actually runs
remotely, once per chunk, on whatever worker picks it up:

```text
       local (x1)
+---------------------+
|     physics code    |
+---------------------+
|     vine_reduce     |
+---------------------+
|     distributor     |
+---------------------+
           |
           v
      remote (xN)
+---------------------+
|       executor      |
+---------------------+
```

This repo's examples all use TaskVine as the distributor; the executor
slot is what varies:

```text
       local (x1)                  local (x1)
+---------------------+     +---------------------+
|     physics code    |     |     physics code    |
+---------------------+     +---------------------+
|     vine_reduce     |     |     vine_reduce     |
+---------------------+     +---------------------+
|       TaskVine      |     |       TaskVine      |
+---------------------+     +---------------------+
           |                           |
           v                           v
      remote (xN)                 remote (xN)
+---------------------+     +---------------------+
|        Coffea       |     |   ROOT RDataFrame   |
| + virtual arrays or |     +---------------------+
|        + dask       |
+---------------------+
```

At Notre Dame the executors run as individual tasks, one per chunk, inside
persistent TaskVine workers on HTCondor.

## Why vine_reduce, alongside Coffea

Most Coffea CMS analyses write a `coffea.processor` and hand it to one of
Coffea's built-in executors, and that's a great default for a lot of
analyses. `vine_reduce` is aimed at a specific point that comes up once
an analysis needs to scale across a cluster: Coffea's executors couple
two concerns that don't have to travel together:

- **Parallelism**: how a processor's work over one dataset is split into
  chunks, mapped, and reduced back together.
- **Distribution**: how those chunks actually get scheduled onto,
  transferred to, and executed on a cluster of remote machines.

`vine_reduce` factors these apart, so the chunking/reduction logic and
the cluster backend it runs on can each change independently of the
other. That's a complement to Coffea, not a replacement for it — see
"Relation to coffea-workflow" below for how the two are meant to fit
together.

## How `vine_reduce` fits in

`vine_reduce` is an orchestrator, not a scheduler or a runtime. It knows
nothing about how to run code on a remote machine. What it knows is how to
turn a HEP analysis into a dynamic MapReduce computation:

- A **dataset** is a named, orthogonal collection of files — no result
  from one dataset is ever combined with another's.
- A **processing function** ("processor") is applied independently to
  chunks of events. Two chunks can be processed in any order, or at the
  same time, without affecting each other's result.
- A **reducer** that folds two processor outputs (or two partial
  reductions) into one is assumed to be commutative, associative, and
  distributive over the dataset's chunks — so partial results can be
  combined in any order, in any grouping, as they become available,
  rather than waiting for a fixed reduction tree.

Because of those properties, `vine_reduce` can submit chunk and reduce
tasks opportunistically, checkpoint intermediate reductions as they
complete, and resume an interrupted run without redoing finished work —
all without knowing anything about *how* a task actually executes. That
part is delegated to a **distributor**.

### Distributor vs. executor

`vine_reduce` draws a sharp line between two roles that Coffea's
executors usually merge:

- **Distributor** — manages the computation across the cluster: submits
  tasks, tracks their outcome, reports resource usage, and tells
  `vine_reduce` when it's safe to free intermediate results. This stack
  uses `TaskVineDistributor`, backed by
  [TaskVine](https://cctools.readthedocs.io/en/stable/taskvine).
- **Executor** — runs a single processor call at the execution site, once
  a distributor has placed it on a worker. `vine_reduce` ships a plain
  in-process executor, a `cloudpickle`-based one that isolates a call in
  its own subprocess, and a `dask_executor` for processors that return a
  dask-delayed object or array.

Splitting these lets a `TaskVineDistributor` place work on a cluster
while the executor running *inside* each task hands off to Dask, or
consumes Coffea's virtual/lazy awkward arrays, without `vine_reduce`
itself needing to know about either.

The concepts above (developed under an earlier prototype's code, but the
same design) are walked through in more depth in
[this presentation](https://docs.google.com/presentation/d/1C1e9BFT1-jZIi08ZGsBuaIqSFFA-ulvQ2SVqGH-D80k/edit?slide=id.g345a2bdd640_4_10#slide=id.g345a2bdd640_4_10)
and in [`vine_reduce`'s own `PLAN.md`](https://github.com/cooperative-computing-lab/vine_reduce/blob/main/PLAN.md).

### Relation to coffea-workflow

[`coffea-workflow`](https://github.com/CoffeaTeam/coffea-workflow) targets
the whole HEP pipeline: sample bookkeeping, preprocessing, analysis, and
beyond. `vine_reduce` is narrower by design — it only covers the analysis
(map/reduce over chunks) step. Conceptually, `vine_reduce` could be
plugged in as one stage of a `coffea-workflow` pipeline rather than
replacing it.

## Advantages

A few consequences follow from the design above:

- **No end-of-run accumulation.** Datasets are orthogonal, so nothing
  waits on every dataset to finish before it can produce a result: each
  (processor, dataset) pair reaches its own final result independently,
  as soon as its own chunks are done, rather than all datasets piling up
  into one accumulation step at the end. Further, several final results
  per (processor, dataset) can be defined according to number of events
  processed.
- **Earlier datasets finish first.** Datasets run concurrently, but when
  submission capacity is limited, they're fed chunks in the order they're
  listed, so datasets declared earlier claim that capacity first and tend
  to reach their own final result sooner. If a run is interrupted, the
  datasets it got to first are more likely to have already checkpointed a
  complete result.
- **Dynamic accumulation.** Because reducers are commutative, associative,
  and distributive, partial results can be folded together in whatever
  order and grouping happens to be ready, instead of following a fixed
  reduction tree laid out ahead of time.
- **Checkpointing.** Intermediate reductions are checkpointed as they're
  produced, not only once a (processor, dataset) pair is entirely done.
- **Restart from disk.** Checkpoints, and a record of what's already been
  processed, live on disk, so an interrupted run resumes from where it
  left off instead of recomputing chunks it already finished.
- **Results never open at the manager.** `vine_reduce`'s own process
  handles results only as opaque tokens or byte streams — it never
  deserializes one, even when writing a final result to disk. The
  memory-heavy work (materializing chunks, reducing them) happens
  entirely at the worker that produced the result, not at the machine
  running `vine_reduce`.
- **No global task graph.** Chunk and reduce tasks are submitted
  opportunistically as work becomes available, rather than built as one
  graph spanning the whole dataset up front. When an executor itself uses
  something like Dask, the graph it builds stays scoped to that one
  chunk — a "minigraph" — instead of growing to cover the entire dataset.
- **Dynamic chunking.** Chunk size adapts to what the distributor reports
  back — it's reduced when a chunk fails from resource exhaustion, rather
  than staying fixed for the whole run.
- **Modular stages.** Because the distributor and the executor are
  separate, no stage of the pipeline has to commit the whole workflow to
  one library's execution model. A single `TaskVineDistributor` run can
  pair a Coffea-with-Dask executor for one processor with a plain ROOT
  RDataFrame call for another, rather than everything having to be "all
  Coffea" or "all Dask" or "all TaskVine."

## Installation

Clone this repo - it carries its own `environment.yml`/`pyproject.toml`,
so it's the only clone needed to start running or adapting the examples.
Its dependencies are Coffea, TaskVine/`ndcctools`, awkward, uproot, ... and
`vine_reduce` itself. Python 3.13+ is required either way. `ndcctools`
(which provides TaskVine) is a conda-forge-only package, so both options
below go through a conda-forge channel rather than plain PyPI.

```bash
git clone https://github.com/cooperative-computing-lab/vine-cms-analysis-stack.git
cd vine-cms-analysis-stack
```

`vine_reduce` isn't published on PyPI yet (pending an organization
authorization), so for now it has to be installed once, up front, from
its own repository, into whichever environment is set up below:

```bash
git clone https://github.com/cooperative-computing-lab/vine_reduce.git /tmp/vine_reduce
```

Once it's on PyPI, that step goes away and `pip install .` below picks
`vine_reduce` up on its own, like any other dependency.


### Option A: conda

```bash
conda env create -f environment.yml
conda activate vine-cms-analysis-stack

# TODO: remove this step
(cd /tmp/vine_reduce && pip install .)
```

### Option B: pixi

[pixi](https://pixi.sh) reproduces the same conda-forge environment from
this repo's own `pyproject.toml`/`pixi.lock`, so it's an alternative to
Option A for when the exact dependency versions should be pinned and
managed automatically. `pixi` clones and installs `vine_reduce` straight
from its repository too (see `[tool.pixi.pypi-dependencies]` in
`pyproject.toml`), so there's no separate step for it here.

```bash
# Install pixi, if you don't already have it
curl -fsSL https://pixi.sh/install.sh | bash

pixi install          # runtime environment
pixi install -e dev   # optional: adds pytest, black, flake8, pyright

# TODO: remove this step
pixi run pip install .
```

Run everything through `pixi run` (e.g. `pixi run python analysis_script.py`)
so it picks up the managed environment, or drop into `pixi shell` for the
rest of the session.

## Packaging the environment for TaskVine workers

Nothing requires a worker node to have Coffea, awkward, or `vine_reduce`
itself pre-installed. `TaskVineDistributor` accepts an `environment=`
argument — a path to a packed, relocatable environment tarball — which it
declares as a [poncho package](https://cctools.readthedocs.io/en/stable/poncho)
(`manager.declare_poncho`) and attaches to every task it submits
(`task.add_environment`). TaskVine then ships that tarball to each worker
alongside the task and unpacks/activates it there, so the worker's host
system needs nothing beyond TaskVine itself.

`vine_reduce.get_environment()` builds that tarball for you, via
`poncho_package_create` — no manual `conda-pack`/`pixi` bookkeeping
needed. It packs `vine_reduce` itself by default; pass this repo's own
checkout through `extra_pip` to include it too (a plain, non-editable pip
install, so — unlike `pip install -e .` — the packed tarball never points
back at a path that only exists on the machine that built it):

```python
from pathlib import Path
from vine_reduce import TaskVineDistributor, get_environment

repo_root = Path(__file__).resolve().parent  # this repo's own checkout

environment = get_environment(
    extra_pip=[str(repo_root)],
    # optional: rebuild automatically whenever this repo has uncommitted
    # changes, the same way it already does for vine_reduce by default -
    # see "Installation" for why this repo is normally installed editable.
    pip_local_to_watch={"vine-cms-analysis-stack": ["examples", "pyproject.toml"]},
)

distributor = TaskVineDistributor(
    port=0,
    resources_processor={"cores": 1},
    resources_reducer={"cores": 1},
    environment=environment,
)
```

Every processor/reducer task submitted through this `distributor` now
runs inside that packed environment at the worker, regardless of what
Python (if any) is installed on that machine. Builds are cached on disk
(keyed by the resolved package spec) and reused across runs; `force=True`
rebuilds unconditionally, and `unstaged="fail"` raises `UnstagedChanges`
instead of silently rebuilding when a watched, editable checkout (this
repo, `vine_reduce`, or anything else named in `pip_local_to_watch`) has
uncommitted changes. Building requires `poncho_package_create` and `conda`
on `PATH` — the same `ndcctools`/`conda` dependency TaskVine itself needs
(see "Installation" above).

See `vine_reduce`'s own README, ["Packaging an environment for remote
workers"](https://github.com/cooperative-computing-lab/vine_reduce#packaging-an-environment-for-remote-workers),
for the full `get_environment()` API, or reach for `poncho_package_create`
directly if a build needs more control than a conda+pip spec allows.

## Quickstart: cortado on synthetic data

`vine_reduce` ships a runnable example built around
[`VineReduceCoffea`](https://github.com/cooperative-computing-lab/vine_reduce/blob/main/src/vine_reduce/coffea.py),
the Coffea specialization of `VineReduce`. It's adapted from the
["cortado" example](https://github.com/cooperative-computing-lab/dynamic_data_reduction/tree/main/examples/cortado)
in `dynamic_data_reduction`, the project `vine_reduce`'s dynamic
map-reduce loop descends from.

It generates synthetic NanoAOD-like ROOT files for two datasets
(`signal` and `background`), skims each down to events with at least
four leptons, and merges the surviving events per dataset with a reducer
that concatenates awkward arrays — no real CMS data or `xrootd` access
needed, and no separate `vine_worker` process to start by hand:

```bash
cd examples/cortado

# conda
conda activate vine-cms-analysis-stack
python vr_cortado.py

# pixi
pixi run python vr_cortado.py
```

The full, heavily-commented source is at
[`examples/cortado/vr_cortado.py`](examples/cortado/vr_cortado.py).
The shape of the call it makes is:

```python
from vine_reduce.coffea import VineReduceCoffea
from vine_reduce.taskvine_distributor import TaskVineDistributor
import awkward as ak
import ndcctools.taskvine as vine

def skimmer(events):
    num_leptons = ak.num(events.Electron) + ak.num(events.Muon)
    return events[num_leptons >= 4]

def accumulate_skims(a, b):
    return ak.concatenate([a, b], axis=0)

# datasets: {name: {"metadata": {...}, "files": {path: {"object_path": "Events", "num_entries": N}}}}
# see build_datasets() in vr_cortado.py for how the synthetic files above are described this way.

distributor = TaskVineDistributor(
    port=0, resources_processor={"cores": 1}, resources_reducer={"cores": 1}
)
workers = vine.Factory(manager_host_port=f"localhost:{distributor.port}")
workers.cores = 2
workers.min_workers = workers.max_workers = 1  # local worker for this quickstart

with workers:
    vr = VineReduceCoffea(
        processors={"skim_4lep": skimmer},
        input=datasets,
        reducer=accumulate_skims,
        chunksize=150,
        results_dir="results",
        checkpoint_dir="checkpoints",
        distributor=distributor,
    )
    vr.compute()
distributor.shutdown()
```

For a version without any Coffea/awkward machinery — plain Python values
chunked out of binary files — see
[`examples/quick_start/quick_start.py`](https://github.com/cooperative-computing-lab/vine_reduce/blob/main/examples/quick_start/quick_start.py)
instead; it's the fastest way to see `chunk_to_args`, `processors`, and
`reducer` wired together without any physics-specific types in the way.

### Running cortado at scale

TODO(btovar): fill in with the actual large-scale cortado run.

stubs...
```bash
cd examples/cortado

# conda
conda activate vine-cms-analysis-stack
python vr_cortado.py

# pixi
pixi run python vr_cortado.py
```

## Production use: ttbarEFT

[`TopEFT/ttbarEFT`](https://github.com/TopEFT/ttbarEFT) is a CMS
top-quark EFT search that runs its analysis stage through `vine_reduce`
on top of TaskVine, distributing histogram-filling processors over an
HTCondor pool.
[`examples/ttBar/run_processor_with_vr.py`](examples/ttBar/run_processor_with_vr.py)
shows how that integration looked in practice: driving a `ttbarEFT`
`AnalysisProcessor` per lepton channel through `vine_reduce`. It predates
the current `VineReduceCoffea`/`TaskVineDistributor` API described above
(it was written against an earlier `vine_reduce` release), so treat it as
a reference for how a full physics analysis wires up channels,
Wilson-coefficient/histogram selection, and X509 proxy handling around
`vine_reduce`, not as a runnable script against the current API. See
[`examples/README.md`](examples/README.md) for a full index of runnable
examples.

## Further reading

- [`vine_reduce` design concepts (presentation)](https://docs.google.com/presentation/d/1C1e9BFT1-jZIi08ZGsBuaIqSFFA-ulvQ2SVqGH-D80k/edit?slide=id.g345a2bdd640_4_10#slide=id.g345a2bdd640_4_10)
- [`vine_reduce` repository](https://github.com/cooperative-computing-lab/vine_reduce)
- [TaskVine documentation](https://cctools.readthedocs.io/en/stable/taskvine)
- [Coffea](https://github.com/scikit-hep/coffea)
- [coffea-workflow](https://github.com/CoffeaTeam/coffea-workflow)
- [ttbarEFT](https://github.com/TopEFT/ttbarEFT)

## License

This project is licensed under the GPL 2.0 - see the LICENSE file for details.
