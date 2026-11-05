import json
from pathlib import Path

import pytest

from varipath.cli import main
from varipath.workflow import load_config


@pytest.mark.parametrize("value", [None, [], "ALDP", 12])
def test_non_object_config_exits_cleanly_before_creating_output(tmp_path, capsys, value):
    path = tmp_path/"config.json"
    path.write_text(json.dumps(value))
    output = tmp_path/"output"
    with pytest.raises(SystemExit) as error:
        main(["prepare", str(path), "--output", str(output)])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "顶层必须是 JSON 对象" in message
    assert "Traceback" not in message
    assert not output.exists()


@pytest.mark.parametrize("section,value", [("runtime", None), ("sampling", []), ("training", "invalid")])
def test_non_object_sections_have_explicit_errors(section, value):
    with pytest.raises(ValueError, match=section):
        load_config({"molecule":"ALDP", section:value})


@pytest.mark.parametrize("threads", [0, -1, 1.7, True, "2", None])
def test_thread_count_is_not_silently_coerced(threads):
    with pytest.raises(ValueError, match="runtime.threads"):
        load_config({"molecule":"ALDP", "runtime":{"threads":threads}})


def test_existing_examples_and_integer_threads_remain_valid():
    root = Path(__file__).resolve().parents[1]
    for path in (root/"configs").glob("*.json"):
        assert load_config(path) == json.loads(path.read_text())
    assert load_config({"molecule":"ALDP", "runtime":{"threads":2}})["runtime"]["threads"] == 2
