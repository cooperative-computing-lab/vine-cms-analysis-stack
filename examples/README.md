# Examples

Runnable examples for this stack live in
[`vine_reduce`](https://github.com/cooperative-computing-lab/vine_reduce),
not in this repo, so there's one canonical copy that stays in sync with
the `VineReduce`/`VineReduceCoffea` API it's written against:

- [`examples/quick_start/quick_start.py`](https://github.com/cooperative-computing-lab/vine_reduce/blob/main/examples/quick_start/quick_start.py) —
  plain Python values chunked out of binary files, no Coffea/awkward
  machinery. The fastest way to see `chunk_to_args`, `processors`, and
  `reducer` wired together.
- [`examples/cortado/vr_cortado.py`](https://github.com/cooperative-computing-lab/vine_reduce/blob/main/examples/cortado/vr_cortado.py) —
  a HEP skim over synthetic NanoAOD-like ROOT files using
  `VineReduceCoffea`, no real CMS data or `xrootd` access needed. See
  the ["Quickstart: cortado on synthetic
  data"](../README.md#quickstart-cortado-on-synthetic-data) section of
  this repo's README for the call shape and how it maps onto this
  stack.
- [`examples/ttBar/run_processor_with_vr.py`](https://github.com/cooperative-computing-lab/vine_reduce/blob/main/examples/ttBar/run_processor_with_vr.py) —
  the `ttbarEFT` production integration. Predates the current
  `VineReduceCoffea`/`TaskVineDistributor` API, so it's a reference for
  how a full physics analysis wires up channels, histogram selection,
  and X509 proxy handling — not a runnable script against the current
  API. See this repo's ["Production use:
  ttbarEFT"](../README.md#production-use-ttbareft) section for context.

Run any of them with `pixi run python <script>.py` from inside
`vine_reduce`, or `python <script>.py` inside an activated `cms-stack`
conda environment (see [Installation](../README.md#installation)).
