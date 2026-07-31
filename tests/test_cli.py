from types import SimpleNamespace

import pytest

import driftaware_sialt.cli as cli
import driftaware_sialt.input_data as input_data


def _resolved_config(tmp_path, stage="stacking"):
    return {
        "stage": stage,
        "logging": str(tmp_path),
    }


def test_syncdata_command_only_synchronizes_inputs(tmp_path, monkeypatch):
    config = _resolved_config(tmp_path)
    synchronized = []
    processed = []

    monkeypatch.setattr(cli, "load_config", lambda path: config.copy())
    monkeypatch.setattr(cli, "init_logger", lambda resolved: None)
    monkeypatch.setattr(
        input_data,
        "sync_required_input_data",
        lambda resolved: (
            synchronized.append(resolved)
            or SimpleNamespace(downloaded=[])
        ),
    )
    monkeypatch.setattr(
        cli, "run_stage", lambda resolved: processed.append(resolved))

    result = cli.main(["syncdata", "stacking.yaml"])

    assert result == 0
    assert len(synchronized) == 1
    assert processed == []


def test_stacking_command_does_not_synchronize_inputs(tmp_path, monkeypatch):
    config = _resolved_config(tmp_path)
    synchronized = []
    processed = []

    monkeypatch.setattr(cli, "load_config", lambda path: config.copy())
    monkeypatch.setattr(cli, "init_logger", lambda resolved: None)
    monkeypatch.setattr(
        input_data,
        "sync_required_input_data",
        lambda resolved: synchronized.append(resolved),
    )
    monkeypatch.setattr(
        cli, "run_stage", lambda resolved: processed.append(resolved))

    result = cli.main(["stacking", "stacking.yaml"])

    assert result == 0
    assert synchronized == []
    assert len(processed) == 1


def test_stage_command_rejects_mismatched_configuration(
    tmp_path,
    monkeypatch,
):
    config = _resolved_config(tmp_path, stage="gridding")
    monkeypatch.setattr(cli, "load_config", lambda path: config.copy())

    with pytest.raises(
        ValueError,
        match=r"stacking command requires.*stage: gridding",
    ):
        cli.main(["stacking", "gridding.yaml"])


@pytest.mark.parametrize(
    "arguments",
    [
        ["sync", "stacking.yaml"],
        ["run", "stacking.yaml"],
        ["stacking.yaml"],
    ],
)
def test_removed_compatibility_commands_are_rejected(arguments):
    with pytest.raises(SystemExit, match="2"):
        cli.main(arguments)
