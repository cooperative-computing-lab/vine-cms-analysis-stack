# Examples

The real physics examples live here, alongside this repo's Coffea/TaskVine
setup; `vine_reduce` only keeps its own minimal, non-physics quickstart:

- [`trijet/vr_trijet_taskvine.py`](trijet/vr_trijet_taskvine.py) —
  **start here.** The introductory template for adapting
  `VineReduceCoffea` to your own analysis: ADL benchmark Q6 (trijet pT /
  max b-tag histograms) from
  [`coffea-benchmarks`](https://github.com/CoffeaTeam/coffea-benchmarks/blob/master/coffea-adl-benchmarks.py),
  translated from a `coffea.processor.ProcessorABC` into a single, plain
  `VineReduceCoffea` processor function over synthetic NanoAOD-like ROOT
  files, no real CMS data needed - the smallest, most self-contained
  example of the translation. Its comments mark the pieces meant to be
  copied as-is ("BOILERPLATE") versus the ones to swap out for your own
  analysis ("REPLACE": your data in `build_datasets()`, your per-chunk
  logic in `trijet_processor()`, and your processor(s) in `main()`'s
  `processors={...}` dict). Runs over `TaskVineDistributor`;
  [`trijet/vr_trijet_iterative.py`](trijet/vr_trijet_iterative.py) is the
  same example run over `LocalDistributor` instead - no cluster or
  `vine_factory`/`vine_worker` needed, useful for a quick local check.
- [`ADL/vr_adl_benchmarks.py`](ADL/vr_adl_benchmarks.py) — all eight
  [IRIS-HEP ADL benchmark](https://github.com/CoffeaTeam/coffea-benchmarks/blob/master/coffea-adl-benchmarks.py)
  queries (Q1-Q8; `processors.py` holds the query bodies, including the
  same `q6` as `trijet` above), translated the same way and run together
  over one synthetic NanoAOD-like dataset. Shows that a coffea
  `ProcessorABC`'s `process(self, events)` body drops in almost unchanged,
  and that `VineReduceCoffea`'s `default_reducer` already knows how to sum
  `Hist` histograms (or, for Q6, a dict of two) across chunks with no
  custom reducer required (contrast with `cortado`'s `accumulate_skims`).
- [`cortado/vr_cortado.py`](cortado/vr_cortado.py) — a HEP skim over
  synthetic NanoAOD-like ROOT files using `VineReduceCoffea`, no real CMS
  data or `xrootd` access needed. See the ["Quickstart: cortado on
  synthetic data"](../DOC.md#quickstart-cortado-on-synthetic-data)
  section of DOC.md for the call shape and how it maps onto this stack.
- [`ttBar/run_processor_with_vr.py`](ttBar/run_processor_with_vr.py) — the
  `ttbarEFT` production integration. Predates the current
  `VineReduceCoffea`/`TaskVineDistributor` API, so it's a reference for
  how a full physics analysis wires up channels, histogram selection, and
  X509 proxy handling — not a runnable script against the current API.
  See DOC.md's ["Production use:
  ttbarEFT"](../DOC.md#production-use-ttbareft) section for context.

## Running the examples

Set up the environment once (see [README.md's Installation
section](../README.md#installation)), then run `trijet`, `ADL`, `cortado`,
or `ttBar` from inside this repo, like:

```bash
cd examples/trijet   # or ADL, cortado, ttBar

# conda
conda activate vine-cms-analysis-stack
python vr_trijet_taskvine.py

# pixi
pixi run python vr_trijet_taskvine.py
```

`trijet`, `ADL`, and `cortado` all generate their synthetic NanoAOD-like
ROOT files the same way, via the shared
[`write_test_data.py`](write_test_data.py) script - each example calls it
as a subprocess the first time it runs (see `ensure_datasets()` in its
`vr_*.py`), and reuses the same files (and the `datasets.json` manifest it
writes alongside them) on later runs. Delete an example's `data/`
directory to force fresh synthetic data.
