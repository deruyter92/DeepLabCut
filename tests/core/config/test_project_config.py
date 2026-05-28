#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Smoke tests for ProjectConfig (DLCVersionedConfig consumer)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from deeplabcut.core.config import ProjectConfig
from deeplabcut.core.config.versioning import CURRENT_CONFIG_VERSION


class TestProjectConfig:
    def test_defaults(self):
        cfg = ProjectConfig()
        assert cfg.config_version == CURRENT_CONFIG_VERSION
        assert cfg.engine == "pytorch"
        assert cfg.Task == ""
        assert isinstance(cfg.project_path, Path)

    def test_from_dict_minimal(self):
        cfg = ProjectConfig.from_dict(
            {
                "Task": "mytask",
                "scorer": "scorer",
                "date": "Jan01",
                "project_path": "/tmp/project",
            }
        )
        assert cfg.Task == "mytask"
        assert cfg.project_path == Path("/tmp/project")

    def test_from_dict_rejects_unknown_keys(self):
        with pytest.raises(ValidationError):
            ProjectConfig.from_dict({"Task": "t", "not_a_real_field": 1})

    def test_is_versioned_config(self):
        cfg = ProjectConfig()
        assert hasattr(cfg, "is_dirty")
        assert not cfg.is_dirty
