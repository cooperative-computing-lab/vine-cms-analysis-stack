# Tutorial examples (Notre Dame)

The scripts and data manifests used by the TaskVine / VineReduce tutorial
([vine-cms-analysis-stack-tutorials](https://github.com/cooperative-computing-lab/vine-cms-analysis-stack-tutorials)).
They run the ADL benchmark Q6 trijet analysis over 15 files of real CMS Open
Data (`Run2012B_SingleMu`, 3,736,416 events in 5 datasets), in two flavors:
coffea (`vr_trijet_*.py`) and RDataFrame (`vr_trijet_*_rdf.py`).

Run everything from this directory, as `results/`, `checkpoints/`, and the
TaskVine logs are created in the current directory:

```bash
cd examples/tutorial_nd
pixi run python scripts/vr_trijet_iterative.py
```

| script | what it adds over the previous one |
|---|---|
| `vr_trijet_iterative` | `LocalDistributor`, over `/project01` |
| `vr_trijet_taskvine` | a TaskVine manager and one local worker |
| `vr_trijet_condor` | HTCondor workers, xrootd data, packed environment |
| `vr_trijet_resources` | chunksize / memory / reduction size on the command line |
| `vr_trijet_checkpoint` | checkpoints and restart |
| `vr_trijet_skim` | `result_postprocess`: a skim written from the worker |
| `vr_trijet_windows` | several processors in one run |
| `vr_trijet_failures` | `failure_proportion` |

**These examples are specific to the Notre Dame CRC.** The manifests point at
files on ND's `/project01` filesystem and at `cmsxrootd.crc.nd.edu`, and
`vr_trijet_condor*` and later expect ND's HTCondor pool and catalog server, so
they will not run elsewhere without changing those. For examples that run
anywhere, see `../trijet/` (synthetic data).

`data/datasets_vast.json` lists the files on `/project01` (login nodes);
`data/datasets_xrootd.json` lists the same files over xrootd (pool nodes).

## Spec

### Problem

The tutorial teaches TaskVine / VineReduce through one analysis that grows
module by module. Students should not have to download scripts or data
separately from the repo they already cloned to build the environment, and
each module should be readable as a small diff against the previous script
instead of as an edit-it-yourself exercise.

### Scope

- In: the 16 scripts in `scripts/` (8 modules x coffea / RDataFrame) and the
  two manifests in `data/`.
- In: the layout contract below.
- Out: the tutorial text itself, which lives in
  [vine-cms-analysis-stack-tutorials](https://github.com/cooperative-computing-lab/vine-cms-analysis-stack-tutorials).
- Out: the certificate wrapper `vr_trijet_condor*` and later modules start
  workers through (`with_oasis_certs`); it is not in this repo and is provided
  separately.

### Behavior

- **Working directory.** Every command is run from `examples/tutorial_nd/`.
  The scripts find `data/` and write `results/` and `checkpoints/` relative
  to their own location (the parent of `scripts/`), so those always land here.
  `vine-run-info/`, `failed_files.log`, `factory.json`, and `plots/` are
  relative to the current directory instead, so the tutorial's commands only
  line up if you run from this directory.
- **Scripts are not edited by students.** Each is a complete, runnable
  program. The tutorial shows what differs from the previous module's script
  (see the table above); the differences are real: they can be reproduced with
  `diff` between the two files.
- **Pairs.** Every script has an `_rdf` twin with the same behavior and the
  same printed totals (3,736,416 events; 1,697,939 entries per histogram, up
  to the edge-of-window rounding noted in Module 8).
- **Manifests.** `datasets_vast.json` and `datasets_xrootd.json` list the same
  15 files in 5 datasets by absolute path; the first for login nodes, the
  second for pool nodes. Paths are absolute because TaskVine tasks run in a
  worker sandbox where a relative path would not resolve.
- **Failure semantics.** A missing `/project01` or an unreachable xrootd host
  makes the first chunk fail; VineReduce retries it and then gives up on the
  file (see Module 9). Nothing here retries or falls back on its own.

### Non-goals

- Running outside Notre Dame. The data, the pool, the catalog server, and the
  certificate wrapper are ND's.
- A stable API. These scripts track the tutorial, not the library.
- Automatic checking that the tutorial text matches the scripts.

### Open questions

- How students obtain the certificate wrapper (Module 4 carries a visible
  TODO until this is decided).
- Whether to add a check (for example, regenerating each module's diff from
  this directory and comparing with the tutorial docs) so edits here cannot
  silently invalidate them. Today it is a manual check: if you change a
  script in `scripts/`, check the matching tutorial page.
