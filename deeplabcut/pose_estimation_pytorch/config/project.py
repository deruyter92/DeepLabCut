#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Project configuration classes for DeepLabCut pose estimation models."""

import warnings
from pydantic.dataclasses import dataclass
from pydantic import model_validator
from dataclasses import field, fields

from pydantic_core import ArgsKwargs

from deeplabcut.core.config import ConfigMixin
from deeplabcut.pose_estimation_pytorch.config.versioning import (
    migrate_config,
    CURRENT_CONFIG_VERSION,
)


@dataclass
class ProjectConfig(ConfigMixin):
    """Complete project configuration.

    Attributes:
        config_version: Configuration schema version (for migration tracking)
        project_path: Path to the DeepLabCut project
        pose_config_path: Path to the pose configuration file
        bodyparts: List of body parts
        unique_bodyparts: List of unique body parts
        individuals: List of individual animal identities
        with_identity: Whether identity tracking is enabled
        multianimalproject: Whether the project is a multi-animal project
        colormap: Colormap for visualization
        dotsize: Dot size for visualization
        alphavalue: Alpha value for visualization
    """  
    config_version: int = CURRENT_CONFIG_VERSION
    multianimalproject: bool = False
    project_path: str = ""
    pose_config_path: str = ""
    bodyparts: list[str] = field(default_factory=list)
    unique_bodyparts: list[str] = field(default_factory=list)
    individuals: list[str] = field(default_factory=list)
    with_identity: bool | None = None
    colormap: str = "rainbow"
    dotsize: int = 12
    alphavalue: float = 0.7

    @model_validator(mode="before")
    @classmethod
    def migrate_and_normalize_config(cls, data: dict | object) -> dict:
        """Migrate and normalize configuration to current version.
        
        This validator:
        1. Detects the config version (defaults to 0 for legacy/unversioned)
        2. Applies all necessary migrations to bring it to the current version
        3. Ensures the config is compatible with the current schema
        
        Args:
            data: Input data (dict for migration, or already an instance)
            
        Returns:
            Migrated and normalized dictionary compatible with current schema
        """
        # If data is already an instance or not a dict, pass through
        if isinstance(data, cls):
            return data

        # Convert to dictionary if ArgsKwargs is passed
        if isinstance(data, ArgsKwargs):
            names = [f.name for f in fields(cls)]
            data = dict(
                zip(names, data.args or []),
                **data.kwargs or {},
            )

        # Migrate config to current version
        # This handles all version-specific transformations
        migrated = migrate_config(data, target_version=CURRENT_CONFIG_VERSION)
        
        return migrated
