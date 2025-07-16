"""validate_tags.py

Validates every tag definition YAML file inside `tag_registry/` against the
canonical JSON Schema and (re)generates `registry_index.yaml`.

Exit status:
    0 – all good
    1 – validation errors or duplicates detected
"""
from __future__ import annotations

import sys
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List

from jsonschema import Draft202012Validator, exceptions as js_exceptions


ROOT_DIR = Path(__file__).resolve().parent.parent
TAG_DIR = ROOT_DIR / "tag_registry"
SCHEMA_PATH = ROOT_DIR / "schemas" / "tag_definition.schema.json"
INDEX_PATH = TAG_DIR / "registry_index.yaml"


class ValidationError(Exception):
    """Raised when a tag file fails schema validation."""


class DuplicateTagError(Exception):
    """Raised when duplicate tag_name/version combos are found."""


def load_schema() -> Draft202012Validator:
    """Load and compile the JSON schema."""
    with SCHEMA_PATH.open("r", encoding="utf-8") as fp:
        schema_data = json.load(fp)
    return Draft202012Validator(schema_data)


def validate_tag_file(path: Path, validator: Draft202012Validator) -> Dict[str, Any]:
    """Validate a single YAML file and return its parsed data if valid."""
    with path.open("r", encoding="utf-8") as fp:
        try:
            data = yaml.safe_load(fp) or {}
        except yaml.YAMLError as exc:
            raise ValidationError(f"YAML parsing error in {path}: {exc}") from exc

    # Derive tag_name when missing
    if "tag_name" not in data or not data["tag_name"]:
        data["tag_name"] = path.stem

    # Perform schema validation
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
    if errors:
        msg_lines: List[str] = [f"Schema validation errors in {path}:"]
        for err in errors:
            loc = "/".join(map(str, err.path))
            msg_lines.append(f" • {loc}: {err.message}")
        raise ValidationError("\n".join(msg_lines))

    return data  # valid


def build_registry(tag_files: List[Path], validator: Draft202012Validator) -> List[Dict[str, Any]]:
    """Validate all tag files and return the registry list."""
    registry: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for path in tag_files:
        try:
            data = validate_tag_file(path, validator)
        except (ValidationError, DuplicateTagError) as exc:
            raise exc  # bubble up

        tag_name: str = data.get("tag_name")
        version: str = data.get("version", "0.0.0")

        key = (tag_name, version)
        if key in seen:
            raise DuplicateTagError(f"Duplicate tag/version detected: {tag_name} {version}")
        seen.add(key)

        registry.append({
            "tag_name": tag_name,
            "version": version,
            "file": str(path.relative_to(ROOT_DIR)),
            "description": data.get("description", "")[:120],  # first 120 chars
        })

    # Sort registry entries by tag_name, then version
    registry.sort(key=lambda d: (d["tag_name"], d["version"]))
    return registry


def write_registry_index(registry: List[Dict[str, Any]]):
    """Write registry list to INDEX_PATH as YAML."""
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with INDEX_PATH.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(registry, fp, sort_keys=False)



def main() -> int:
    tag_files = [
        p
        for p in TAG_DIR.glob("*.yaml")
        if p.name not in {"registry_index.yaml"}
    ]

    if not tag_files:
        print("No tag definition YAML files found in tag_registry/", file=sys.stderr)

    try:
        validator = load_schema()
    except FileNotFoundError:
        print(f"Schema not found at {SCHEMA_PATH}", file=sys.stderr)
        return 1

    try:
        registry = build_registry(tag_files, validator)
    except (ValidationError, DuplicateTagError) as exc:
        print(exc, file=sys.stderr)
        return 1

    write_registry_index(registry)
    print(f"Validated {len(registry)} tag file(s). Registry written to {INDEX_PATH.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())