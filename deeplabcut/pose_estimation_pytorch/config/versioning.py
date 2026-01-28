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
"""Configuration migration system for handling version upgrades and downgrades.

This module provides a versioned migration system that allows configurations
to be upgraded from older versions to newer ones, or downgraded to older formats.
Upgrade migrations are chained together, so any version can be upgraded to the
latest by applying all intermediate migrations in sequence. Downgrade migrations
can be registered for specific version pairs when backward compatibility is needed.
"""

from typing import Callable, Dict
from functools import wraps


# Current configuration schema version
# Increment this when making breaking changes to the config structure
CURRENT_CONFIG_VERSION = 1


# Version registry: maps (from_version, to_version) -> migration function
_MIGRATIONS: Dict[tuple[int, int], Callable[[dict], dict]] = {}


def register_migration(from_version: int, to_version: int):
    """Decorator to register a migration function.
    
    Args:
        from_version: The source version number
        to_version: The target version number (must be from_version + 1)
    
    Example:
        @register_migration(1, 2)
        def migrate_v1_to_v2(config: dict) -> dict:
            # Transform config from version 1 to version 2
            return config
    """
    def decorator(func: Callable[[dict], dict]) -> Callable[[dict], dict]:
        _MIGRATIONS[(from_version, to_version)] = func
        
        @wraps(func)
        def wrapper(config: dict) -> dict:
            result = func(config.copy())  # Don't mutate input
            result["config_version"] = to_version
            return result
        
        return wrapper
    return decorator


def get_config_version(config: dict) -> int:
    """Extract the configuration version from a config dict.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Version number (0 for legacy/unversioned configs)
    """
    return config.get("config_version", 0)


def migrate_config(config: dict, target_version: int = CURRENT_CONFIG_VERSION) -> dict:
    """Migrate a configuration to the target version.
    
    Applies all necessary migrations in sequence to upgrade the config
    from its current version to the target version.
    
    Args:
        config: Configuration dictionary to migrate
        target_version: Target version to migrate to (default: current)
        
    Returns:
        Migrated configuration dictionary
        
    Raises:
        ValueError: If migration chain is incomplete or target version is invalid
    """
    current_version = get_config_version(config)
    
    if current_version == target_version:
        return config
    
    if target_version > CURRENT_CONFIG_VERSION:
        raise ValueError(
            f"Target version {target_version} exceeds current version {CURRENT_CONFIG_VERSION}"
        )
    
    # Apply migrations sequentially
    migrated = config.copy()
    migration_key = (current_version, target_version)
    if migration_key not in _MIGRATIONS:
        raise ValueError(
            f"Missing migration from version {current_version} to {target_version}. "
            f"Available migrations: {list(_MIGRATIONS.keys())}"
        )
    
    migration_func = _MIGRATIONS[migration_key]
    migrated = migration_func(migrated)
    
    return migrated


# ============================================================================
# Migration Definitions
# ============================================================================

@register_migration(0, 1)
def migrate_v0_to_v1(config: dict) -> dict:
    """Migrate from unversioned/legacy config (v0) to v1.
    
    This migration handles the initial typed config structure:
    - Normalizes legacy field names (uniquebodyparts -> unique_bodyparts)
    - Converts multianimalbodyparts to unified bodyparts field
    - Handles "MULTI!" marker in bodyparts
    - Normalizes multianimalproject flag
    - Converts identity -> with_identity
    """
    normalized = config.copy()
    
    # Normalize legacy field names
    if "uniquebodyparts" in normalized and "unique_bodyparts" not in normalized:
        normalized["unique_bodyparts"] = normalized.pop("uniquebodyparts")
    
    # Handle multianimalbodyparts -> bodyparts
    if "multianimalbodyparts" in normalized:
        multianimal_bodyparts = normalized.pop("multianimalbodyparts")
        
        if "bodyparts" in normalized:
            existing_bodyparts = normalized["bodyparts"]
            if existing_bodyparts == "MULTI!":
                normalized["bodyparts"] = multianimal_bodyparts
            elif isinstance(existing_bodyparts, list):
                # Merge, avoiding duplicates while preserving order
                combined = list(dict.fromkeys(multianimal_bodyparts + existing_bodyparts))
                normalized["bodyparts"] = combined
            else:
                normalized["bodyparts"] = multianimal_bodyparts
        else:
            normalized["bodyparts"] = multianimal_bodyparts
    
    # Handle "MULTI!" marker
    if "bodyparts" in normalized and normalized["bodyparts"] == "MULTI!":
        normalized["bodyparts"] = []
    
    # Normalize multianimalproject
    # Check if multianimalbodyparts existed before we popped it
    had_multianimal_bodyparts = "multianimalbodyparts" in config
    if "multianimalproject" not in normalized or normalized.get("multianimalproject") in (None, ""):
        has_individuals = "individuals" in normalized and normalized.get("individuals")
        if has_individuals or had_multianimal_bodyparts:
            normalized["multianimalproject"] = True
        else:
            normalized["multianimalproject"] = False
    else:
        value = normalized["multianimalproject"]
        if isinstance(value, str):
            normalized["multianimalproject"] = value.lower() in ("true", "1", "yes")
        else:
            normalized["multianimalproject"] = bool(value)
    
    # Normalize identity -> with_identity
    if "identity" in normalized and "with_identity" not in normalized:
        normalized["with_identity"] = normalized.pop("identity")
    
    # Ensure lists are properly typed
    if "bodyparts" in normalized:
        if normalized["bodyparts"] == "MULTI!":
            normalized["bodyparts"] = []
        elif not isinstance(normalized["bodyparts"], list):
            normalized["bodyparts"] = list(normalized["bodyparts"]) if normalized["bodyparts"] else []
    
    if "unique_bodyparts" in normalized:
        if not isinstance(normalized["unique_bodyparts"], list):
            normalized["unique_bodyparts"] = list(normalized["unique_bodyparts"]) if normalized["unique_bodyparts"] else []
    
    return normalized


@register_migration(1, 0)
def migrate_v1_to_v0(config: dict) -> dict:
    """Migrate from v1 to v0 (legacy format).
    
    This migration converts the new typed config structure back to legacy format:
    - Converts unique_bodyparts -> uniquebodyparts
    - Converts bodyparts -> multianimalbodyparts (for multi-animal projects)
    - Sets bodyparts to "MULTI!" for multi-animal projects
    - Converts with_identity -> identity
    - Removes config_version (or sets to 0)
    """
    normalized = config.copy()
    
    # Convert unique_bodyparts -> uniquebodyparts
    if "unique_bodyparts" in normalized and "uniquebodyparts" not in normalized:
        normalized["uniquebodyparts"] = normalized.pop("unique_bodyparts")
    
    # Convert bodyparts -> multianimalbodyparts for multi-animal projects
    is_multi_animal = normalized.get("multianimalproject", False)
    if is_multi_animal and "bodyparts" in normalized:
        bodyparts = normalized.get("bodyparts", [])
        if isinstance(bodyparts, list) and bodyparts:
            # For multi-animal projects, bodyparts becomes multianimalbodyparts
            normalized["multianimalbodyparts"] = bodyparts
            normalized["bodyparts"] = "MULTI!"
        elif not bodyparts:
            # Empty bodyparts in multi-animal project
            normalized["multianimalbodyparts"] = []
            normalized["bodyparts"] = "MULTI!"
    
    # Convert with_identity -> identity
    if "with_identity" in normalized and "identity" not in normalized:
        normalized["identity"] = normalized.pop("with_identity")
    
    # Remove config_version (legacy format doesn't have it, or set to 0)
    if "config_version" in normalized:
        normalized["config_version"] = 0
    
    return normalized


@register_migration(1, 2)
def migrate_v1_to_v2(config: dict) -> dict:
    """Migrate from v1 to v2.
    
    Example migration - remove multianimalproject field as it can be inferred.
    This is just an example - adjust based on actual v2 changes.
    """
    # Example: In v2, we might remove multianimalproject and infer it
    # For now, this is a no-op migration to demonstrate the pattern
    normalized = config.copy()
    
    # Future v2 changes would go here
    # Example:
    # if "multianimalproject" in normalized:
    #     # Infer from other fields instead
    #     del normalized["multianimalproject"]
    
    return normalized


# ============================================================================
# Future migrations can be added here following the same pattern:
# ============================================================================
#
# @register_migration(2, 3)
# def migrate_v2_to_v3(config: dict) -> dict:
#     """Migrate from v2 to v3.
#     
#     Describe what changes in this version.
#     """
#     normalized = config.copy()
#     # Apply transformations...
#     return normalized
