"""Regression tests for ModelFactory lazy backend dispatch (P4.1/P4.3, audit 2026-07-04).

RL_Agent.from_agent_dir used to hardcode an if/else on
SupportedRLFrameworks.DREAMER_V3 and eagerly import both
StableBaselinesModel and DreamerV3Model at module import time. Now
ModelFactory.get_model_class() resolves + imports the requested backend
only, and each RL_Model subclass declares its own inference construction
kwargs / load sequence via inference_construction_kwargs()/load_for_inference().
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def test_get_model_class_resolves_stable_baselines3():
    from rosnav_rl.model.model_factory import ModelFactory
    from rosnav_rl.model.stable_baselines3 import StableBaselinesModel
    from rosnav_rl.utils.type_aliases import SupportedRLFrameworks

    assert (
        ModelFactory.get_model_class(SupportedRLFrameworks.STABLE_BASELINES3)
        is StableBaselinesModel
    )


def test_get_model_class_resolves_dreamer_v3():
    from rosnav_rl.model.dreamerv3.dreamerv3_model import DreamerV3Model
    from rosnav_rl.model.model_factory import ModelFactory
    from rosnav_rl.utils.type_aliases import SupportedRLFrameworks

    assert (
        ModelFactory.get_model_class(SupportedRLFrameworks.DREAMER_V3)
        is DreamerV3Model
    )


def test_get_model_class_unsupported_framework_raises():
    from rosnav_rl.model.model_factory import ModelFactory

    with pytest.raises(ValueError):
        ModelFactory.get_model_class("bogus_framework")


def test_default_inference_construction_kwargs_is_empty():
    from rosnav_rl.model.model import RL_Model

    assert RL_Model.inference_construction_kwargs(Path("/fake")) == {}


def test_dreamer_inference_construction_kwargs():
    from rosnav_rl.model.dreamerv3.dreamerv3_model import DreamerV3Model

    assert DreamerV3Model.inference_construction_kwargs(Path("/fake")) == {
        "inference_only": True
    }


def test_default_load_for_inference_loads_best_model_when_uninitialized():
    from rosnav_rl.model.model import RL_Model

    mock_self = MagicMock()
    mock_self.is_model_initialized = False
    RL_Model.load_for_inference(mock_self, Path("/fake/dir"))
    mock_self.load.assert_called_once_with(path=Path("/fake/dir") / "best_model.zip")


def test_default_load_for_inference_skips_load_when_already_initialized():
    from rosnav_rl.model.model import RL_Model

    mock_self = MagicMock()
    mock_self.is_model_initialized = True
    RL_Model.load_for_inference(mock_self, Path("/fake/dir"))
    mock_self.load.assert_not_called()


def test_dreamer_load_for_inference_calls_setup_then_load():
    from rosnav_rl.model.dreamerv3.dreamerv3_model import DreamerV3Model

    mock_self = MagicMock()
    DreamerV3Model.load_for_inference(mock_self, Path("/fake/dir"))
    mock_self.setup_model.assert_called_once_with()
    mock_self.load.assert_called_once_with("latest")


def test_stable_baselines_dispatch_does_not_import_dreamer():
    # Run in a fresh subprocess: within the same pytest session other test
    # modules will have already imported dreamerv3, making an in-process
    # sys.modules check meaningless.
    code = (
        "import sys\n"
        "from rosnav_rl.model.model_factory import ModelFactory\n"
        "from rosnav_rl.utils.type_aliases import SupportedRLFrameworks\n"
        "ModelFactory.get_model_class(SupportedRLFrameworks.STABLE_BASELINES3)\n"
        "assert 'rosnav_rl.model.dreamerv3.dreamerv3_model' not in sys.modules, "
        "sorted(m for m in sys.modules if 'dreamerv3' in m)\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
