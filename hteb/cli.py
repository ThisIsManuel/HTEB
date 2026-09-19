"""The command accepts a YAML configuration path and help only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from .config import load_config
from .hteb import run_hteb


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="hteb",
        description="Generate or reuse configured transformations and evaluate embedding models.",
    )
    parser.add_argument("config", type=Path, help="YAML configuration path")
    arguments = sys.argv[1:]
    if arguments[:1] == ["evaluate"]:
        arguments = arguments[1:]
    args = parser.parse_args(arguments)
    try:
        config = load_config(args.config)
    except UnicodeDecodeError:
        parser.exit(2, f"Cannot read {args.config}: expected UTF-8 text.\n")
    except yaml.YAMLError as error:
        parser.exit(2, f"Invalid YAML configuration:\n{error}\n")
    except ValidationError as error:
        details = "\n\n".join(
            f"{'.'.join(str(part) for part in item['loc']) or 'configuration'}\n  {item['msg']}"
            for item in error.errors(include_url=False, include_context=False, include_input=False)
        )
        parser.exit(2, f"Invalid HTEB configuration:\n\n{details}\n")
    except OSError as error:
        parser.exit(2, f"Unable to read configuration: {error}\n")
    try:
        path = run_hteb(config)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"Runtime error ({type(error).__name__}): {error}\n")
    print(f"Wrote score to {path}")
