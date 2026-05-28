from __future__ import annotations

import functools
import logging
import sys
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from pydantic import ArgsKwargs, BaseModel, ConfigDict, model_validator
from ruamel.yaml.comments import CommentedMap
from typing_extensions import Self

from deeplabcut.core.config.utils import (
    normalize_for_serialization,
    pretty_print,
    read_config_as_dict,
    resolve_aliases_in_dict,
    write_config,
)
from deeplabcut.core.config.versioning import CURRENT_CONFIG_VERSION, migrate_config

logger = logging.getLogger(__name__)


class DLCBaseConfig(BaseModel):
    """Base configuration class for all DeepLabCut configurations."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @classmethod
    @functools.cache
    def _alias_map(cls) -> dict[str, str]:
        """Build ``{alias: canonical_name}`` from ``json_schema_extra``."""
        mapping: dict[str, str] = {}
        for name, info in cls.model_fields.items():
            extra = info.json_schema_extra
            if not isinstance(extra, dict):
                continue
            for alias in extra.get("aliases", []):
                if alias in mapping:
                    raise ValueError(f"Duplicate alias '{alias}' for fields '{mapping[alias]}' and '{name}'")
                mapping[alias] = name
        return mapping

    def _resolve_alias(self, name: str) -> str | None:
        return type(self)._alias_map().get(name)

    @classmethod
    def _resolve_aliases_in_dict(cls, cfg_dict: dict) -> dict:
        return resolve_aliases_in_dict(cfg_dict, cls._alias_map())

    @classmethod
    def from_dict(cls, cfg_dict: dict) -> Self:
        cfg_dict = cls._resolve_aliases_in_dict(cfg_dict)
        return cls.model_validate(cfg_dict)

    @classmethod
    def from_any(
        cls,
        config: Self | dict | str | Path,
    ) -> Self:
        if isinstance(config, cls):
            return config
        elif isinstance(config, str | Path):
            return cls.from_yaml(config)
        elif isinstance(config, dict):
            return cls.from_dict(config)
        else:
            raise TypeError(
                "Failure to load configuration: Expected a config instance, "
                f"dictionary, string, or Path. Got {type(config)}"
            )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path, ignore_empty: bool = True) -> Self:
        yaml_dict = read_config_as_dict(yaml_path)
        if ignore_empty:
            yaml_dict = {k: v for k, v in yaml_dict.items() if v is not None}
        cfg = cls.from_dict(yaml_dict)
        cfg._post_yaml_load_updates(yaml_path=Path(yaml_path))
        return cfg

    def to_yaml(
        self,
        yaml_path: str | Path,
        *,
        overwrite: bool = True,
    ) -> None:
        dict_data = self.to_dict(normalize=True)
        data = CommentedMap(dict_data)
        for name, info in type(self).model_fields.items():
            extra = info.json_schema_extra
            if isinstance(extra, dict) and (comment := extra.get("comment")):
                data.yaml_set_comment_before_after_key(name, before=comment)
        write_config(yaml_path, data, overwrite=overwrite)

    def to_dict(self, *, normalize: bool = False) -> dict:
        if not normalize:
            return self.model_dump()
        return normalize_for_serialization(self.model_dump())

    def print(
        self,
        indent: int = 0,
        print_fn: Callable[[str], None] | None = None,
    ) -> None:
        pretty_print(config=self.to_dict(), indent=indent, print_fn=print_fn)

    def _post_yaml_load_updates(self, *, yaml_path: Path) -> None:
        pass

    def _warn_alias(self, alias: str, canonical: str, stacklevel: int = 3) -> None:
        warnings.warn(
            f"'{alias}' is deprecated, use '{canonical}' instead.",
            DeprecationWarning,
            stacklevel=stacklevel,
        )

    def __getattr__(self, name: str) -> Any:
        canonical = self._resolve_alias(name)
        if canonical is not None:
            self._warn_alias(name, canonical, stacklevel=2)
            return getattr(self, canonical)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __getitem__(self, key: str) -> Any:
        canonical = self._resolve_alias(key)
        if canonical is not None:
            self._warn_alias(key, canonical)
            return getattr(self, canonical)
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def __setitem__(self, key: str, value: Any) -> None:
        canonical = self._resolve_alias(key)
        if canonical is not None:
            self._warn_alias(key, canonical)
            key = canonical
        if key not in self._field_names():
            raise KeyError(f"'{type(self).__name__}' has no field '{key}'")
        setattr(self, key, value)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if key in self._field_names():
            return True
        return self._resolve_alias(key) is not None

    def __iter__(self) -> Iterator[str]:
        return iter(self._field_names())

    def __len__(self) -> int:
        return len(self._field_names())

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> list[str]:
        return self._field_names()

    def values(self) -> list[Any]:
        return [getattr(self, name) for name in self._field_names()]

    def items(self) -> list[tuple[str, Any]]:
        return [(name, getattr(self, name)) for name in self._field_names()]

    def select(self, path: str, default: Any = None) -> Any:
        obj: Any = self
        for part in path.split("."):
            if obj is None:
                return default
            try:
                obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
            except (KeyError, AttributeError, TypeError):
                return default
        return obj

    def _field_names(self) -> list[str]:
        cls = type(self)
        if not isinstance(self, BaseModel):
            raise TypeError(f"{cls.__name__} must inherit from pydantic.BaseModel")
        return list(cls.model_fields.keys())


class DLCVersionedConfig(DLCBaseConfig):
    """Configuration class for all DeepLabCut configurations with versioning."""

    @model_validator(mode="before")
    @classmethod
    def migrate_before_validate(cls, data: Any) -> Any:
        """Migrate raw input data to the current config version.

        Converts ``ArgsKwargs`` (positional / keyword constructor args) to a
        plain dict, then runs the migration chain.  Already-constructed
        instances and non-dict data are returned unchanged.
        """
        if isinstance(data, ArgsKwargs):
            names = list(cls.model_fields.keys())
            data = dict(
                zip(names, data.args or [], strict=False),
                **(data.kwargs or {}),
            )
        if isinstance(data, dict):
            data = migrate_config(data, target_version=CURRENT_CONFIG_VERSION)
        return data

    def to_yaml(
        self,
        yaml_path: str | Path,
        *,
        overwrite: bool = True,
        log_changes: bool = True,
        mark_clean: bool = True,
    ) -> None:
        super().to_yaml(yaml_path, overwrite=overwrite)
        if log_changes:
            self.log_changes()
        if mark_clean:
            self.mark_clean()

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._init_change_tracking()

    def _init_change_tracking(self) -> None:
        if getattr(self, "_change_tracking_initialized", False):
            return
        object.__setattr__(self, "_change_tracking_initialized", True)

        cls = type(self)
        if not getattr(cls, "_change_tracking_installed", False):
            original_setattr = cls.__setattr__

            def __setattr__(self, name: str, value: Any) -> None:
                if name in (
                    "_dirty_fields",
                    "_change_notes",
                    "_change_tracking_initialized",
                ):
                    object.__setattr__(self, name, value)
                    return
                field_names = list(type(self).model_fields.keys())
                dirty_fields = getattr(self, "_dirty_fields", None)
                if dirty_fields is not None and name in field_names:
                    old = getattr(self, name)
                    original_setattr(self, name, value)
                    if old != value:
                        dirty_fields.add(name)
                else:
                    original_setattr(self, name, value)

            cls.__setattr__ = __setattr__
            cls._change_tracking_installed = True

        object.__setattr__(self, "_dirty_fields", set())
        object.__setattr__(self, "_change_notes", {})

    @property
    def is_dirty(self) -> bool:
        return bool(self._dirty_fields)

    @property
    def dirty_fields(self) -> frozenset[str]:
        return frozenset(self._dirty_fields)

    @property
    def change_notes(self) -> list[str]:
        return list(self._change_notes.values())

    def record_change_note(
        self,
        field_name: str,
        message: str,
        *,
        include_caller: bool = False,
        _stack_depth: int = 1,
    ) -> None:
        if include_caller:
            frame = sys._getframe(_stack_depth)
            filename = frame.f_code.co_filename.rsplit("/", 1)[-1]
            message = f"{message} [{filename}:{frame.f_lineno}]"
        self._change_notes[field_name] = message

    def log_changes(self) -> None:
        if not self.is_dirty:
            return
        logger.info(f"Updates to {type(self).__name__}:")
        for field_name in sorted(self._dirty_fields):
            if field_name in self._change_notes:
                logger.info(f"  {self._change_notes[field_name]}")
            else:
                logger.info(f"  {field_name} was modified")

    def mark_clean(self) -> None:
        self._dirty_fields.clear()
        self._change_notes.clear()
