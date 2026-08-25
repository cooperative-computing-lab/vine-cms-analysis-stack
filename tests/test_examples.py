"""End-to-end test that actually runs examples/cortado/vr_cortado.py, exactly
`pixi run python vr_cortado.py`, the same way a user would - against a real
local TaskVine worker. Slower and heavier than a unit test (starts a
manager, a vine_factory, and a worker process, and waits for real tasks to
run): this is an integration smoke test for the example itself, not a unit
test of vine_reduce's internals.

Moved here from vine_reduce's tests/test_examples.py when the cortado
example moved to this repo. Not yet wired into this repo's test tooling
(no pyproject.toml/pytest setup here) - moved as-is to fix later if needed.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("vine_factory") is None, reason="vine_factory not on PATH"
)

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
TIMEOUT = 600  # generous: covers worker connect latency plus real task execution


def _run_example(
    tmp_path: Path, example_dir_name: str, script_name: str
) -> subprocess.CompletedProcess:
    """Copies examples/<example_dir_name>'s .py files into an isolated tmp
    directory and runs script_name there, exactly as a user would from
    within examples/<example_dir_name>. Isolating into tmp_path keeps
    generated data/results/checkpoints out of the source tree.

    Also copies examples/write_test_data.py to tmp_path itself (a sibling
    of run_dir, not inside it) - each example's vr_*.py locates it via
    "../write_test_data.py" relative to its own directory, so it needs to
    land one level up from run_dir, exactly mirroring the real examples/
    layout.
    """
    src_dir = EXAMPLES_DIR / example_dir_name
    run_dir = tmp_path / example_dir_name
    run_dir.mkdir()
    for py_file in src_dir.glob("*.py"):
        shutil.copy(py_file, run_dir / py_file.name)
    shutil.copy(EXAMPLES_DIR / "write_test_data.py", tmp_path / "write_test_data.py")

    return subprocess.run(
        [sys.executable, script_name],
        cwd=run_dir,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )


def test_trijet_example(tmp_path):
    result = _run_example(tmp_path, "trijet", "vr_trijet_taskvine.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK: trijet histograms filled for the large majority of events" in result.stdout


def test_trijet_iterative_example(tmp_path):
    result = _run_example(tmp_path, "trijet", "vr_trijet_iterative.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK: trijet histograms filled for the large majority of events" in result.stdout


def test_cortado_example(tmp_path):
    result = _run_example(tmp_path, "cortado", "vr_cortado.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK: signal passes the >=4-lepton skim more often than background" in result.stdout


def test_adl_example(tmp_path):
    result = _run_example(tmp_path, "ADL", "vr_adl_benchmarks.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK: all 10 ADL benchmark queries ran" in result.stdout
