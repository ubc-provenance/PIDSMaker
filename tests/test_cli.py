"""The `pidsmaker` command runs the pipeline with the environment of reproducible runs, in the
foreground or in the background (pidsmaker/cli.py)."""

import os
import sys

import pytest

import pidsmaker.cli as cli

pytestmark = pytest.mark.no_database


@pytest.fixture
def calls(monkeypatch):
    """Records the restarts, background processes and pipeline runs instead of running them."""
    calls = {"execve": [], "popen": [], "pipeline": []}

    def execve(path, args, env):
        calls["execve"].append((path, args, env))

    class Popen:
        pid = 1234

        def __init__(self, command, **kwargs):
            calls["popen"].append((command, kwargs))

    import pidsmaker.main

    monkeypatch.setattr(cli.os, "execve", execve)
    monkeypatch.setattr(cli.subprocess, "Popen", Popen)
    monkeypatch.setattr(pidsmaker.main, "cli", lambda argv: calls["pipeline"].append(argv))
    return calls


def test_restarts_with_reproducible_env(calls, monkeypatch):
    for var in cli.REPRODUCIBLE_ENV:
        monkeypatch.delenv(var, raising=False)
    cli.main(["orthrus", "CADETS_E3", "--wandb"])
    ((path, args, env),) = calls["execve"]
    assert path == sys.executable
    assert args == [sys.executable, "-m", "pidsmaker.cli", "orthrus", "CADETS_E3", "--wandb"]
    assert env["PYTHONHASHSEED"] == "0" and env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_runs_when_env_is_set(calls, monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    # Values set by the user are kept
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    cli.main(["orthrus", "CADETS_E3"])
    assert calls["execve"] == []
    assert calls["pipeline"] == [["orthrus", "CADETS_E3"]]


def test_background(calls, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    for _ in range(2):
        assert cli.main(["kairos", "THEIA_E3", "--background", "--wandb"]) == 0
    assert calls["execve"] == [] and calls["pipeline"] == []
    (command, kwargs), _ = calls["popen"]
    assert command == [sys.executable, "-m", "pidsmaker.cli", "kairos", "THEIA_E3", "--wandb"]
    assert kwargs["start_new_session"]
    assert kwargs["env"]["PYTHONHASHSEED"] == "0"
    assert kwargs["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    # One log file per run, in the working directory
    logs = sorted(os.listdir(tmp_path))
    assert len(logs) == 2 and all(f.startswith("pidsmaker_kairos_THEIA_E3_") for f in logs)
