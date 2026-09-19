"""Allow ``python -m hteb`` to behave like the installed command."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":  # pragma: no cover - exercised by package command tests
    main()
