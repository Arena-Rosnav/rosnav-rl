import pytest

from rosnav_rl.utils.yaml_io import dump_yaml, dumps_yaml, load_yaml, loads_yaml

CONFIG = {"lr": 3e-05, "eps": 1e-08, "steps": 600, "normalize": True, "name": "agent", "ranges": [0.0, 0.5]}


def test_a_dumped_mapping_loads_back_with_the_same_values(tmp_path):
    dump_yaml(CONFIG, tmp_path / "config.yaml")

    assert load_yaml(tmp_path / "config.yaml") == CONFIG


def test_exponent_floats_without_a_decimal_point_read_as_floats():
    assert loads_yaml("lr: 3e-05\neps: 1e-08\n") == {"lr": pytest.approx(3e-05), "eps": pytest.approx(1e-08)}


def test_dumps_keeps_the_insertion_order_in_block_style():
    assert dumps_yaml({"b": 1, "a": {"c": 2}}) == "b: 1\na:\n  c: 2\n"
