"""Fast smoke coverage for the high-count gateway stress harness."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import subprocess
import tempfile
from pathlib import Path
import pytest


_MODULE_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "stress_gateway.py"
_SPEC = importlib.util.spec_from_file_location("stress_gateway_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
stress = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = stress
_SPEC.loader.exec_module(stress)


def test_small_gateway_matrix_reconciles() -> None:
    args = stress.build_parser().parse_args(
        ["--requests", "30", "--concurrency", "1,8", "--sample-every", "1,7"]
    )
    report = asyncio.run(stress.async_main(args))
    assert report["passed"] is True
    assert report["aggregate"]["attempts"] == 120
    assert report["aggregate"]["request_failures"] == 0
    assert report["aggregate"]["invariant_failures"] == 0
    for cell in report["cells"]:
        expected = cell["config"]["requests"] // cell["config"]["sample_every"]
        assert cell["counts"]["records"] == expected
        assert cell["counts"]["unique_validation_ids"] == expected


def test_direct_script_cli_without_pythonpath() -> None:
    result = subprocess.run(
        [sys.executable, str(_MODULE_PATH), "--requests", "2", "--concurrency", "1", "--sample-every", "1"],
        cwd=_MODULE_PATH.parents[1], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_matrix_bounds_only_limit_concurrency_values() -> None:
    assert stress.parse_positive_csv("257", "--sample-every") == [257]
    with pytest.raises(stress.argparse.ArgumentTypeError):
        stress.parse_positive_csv("257", "--concurrency")
    with pytest.raises(stress.argparse.ArgumentTypeError):
        stress.parse_positive_csv(",".join(["1"] * 17), "--sample-every")
    args = stress.build_parser().parse_args(
        ["--requests", "1", "--concurrency", "1,2,3,4,5,6,7,8,9", "--sample-every", "1,2,3,4,5,6,7,8"]
    )
    with pytest.raises(stress.argparse.ArgumentTypeError, match="64 cells"):
        asyncio.run(stress.async_main(args))


def test_cell_failure_returns_safe_report_and_always_cleans_up(monkeypatch) -> None:
    calls: list[str] = []

    async def fail_start(self) -> None:
        calls.append("start")
        raise RuntimeError("primary failure must not be rendered")

    async def fail_stop(self) -> None:
        calls.append("stop")
        raise RuntimeError("cleanup failure must not mask primary")

    monkeypatch.setattr(stress.GatewayService, "start", fail_start)
    monkeypatch.setattr(stress.GatewayService, "stop", fail_stop)
    with tempfile.TemporaryDirectory() as directory:
        report = asyncio.run(stress.run_cell(3, 1, 1, Path(directory)))
    assert calls == ["start", "stop"]
    assert report["invariants_passed"] is False
    assert report["invariant_failure_categories"] == ["cell_exception"]
    assert report["exception_categories"] == ["RuntimeError"]
    assert report["latency_seconds"]["max"] is None


def test_cleanup_failure_cannot_be_reported_as_a_passing_cell(monkeypatch) -> None:
    async def fail_stop(self) -> None:
        raise RuntimeError("cleanup failure must remain content-free")

    monkeypatch.setattr(stress.GatewayService, "stop", fail_stop)
    with tempfile.TemporaryDirectory() as directory:
        report = asyncio.run(stress.run_cell(2, 1, 1, Path(directory)))
    assert report["invariants_passed"] is False
    assert report["exception_categories"] == ["RuntimeError"]
