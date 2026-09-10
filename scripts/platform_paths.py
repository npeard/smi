"""Where every platform difference this repo cares about is decided.

Kept in one place because the same three assumptions -- the pixi interpreter
layout, that a link is a symlink, and that reads are UTF-8 -- were repeated
across install, hook registration, preflight and the audit, so a Windows fix
in one left the others wrong.

Hooks deliberately do NOT import this. They are spawned standalone with an
arbitrary cwd, so they stay stdlib-only and self-contained; the duplication
that costs is held in check by a shared test fixture instead.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

WINDOWS = os.name == "nt"


def interpreter(repo: Path) -> Path:
    """Path to the dev environment's Python.

    pixi puts the interpreter at the env root on Windows and under bin/
    everywhere else. Hardcoding the POSIX form made `register_hooks.py`
    abort on every Windows run.
    """
    env = Path(repo) / ".pixi" / "envs" / "dev"
    return env / "python.exe" if WINDOWS else env / "bin" / "python"


def link_dir(src: Path, dest: Path) -> None:
    """Point dest at the directory src.

    A junction on Windows: it needs no elevation and no Developer Mode,
    which os.symlink does (WinError 1314). Directory junctions cannot span
    to files, which is why CLAUDE.md is installed as an @import stub rather
    than a link.
    """
    src = Path(src).resolve()
    dest = Path(dest)
    if WINDOWS:
        # /J is the unprivileged form. mklink is a cmd builtin, so it cannot
        # be exec'd directly.
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(dest), str(src)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise OSError(f"mklink failed for {dest} -> {src}: {result.stderr.strip()}")
        if not is_link(dest):
            raise OSError(
                f"mklink reported success but {dest} is not a junction "
                f"(produced a non-link, e.g. a plain directory, for {dest} -> {src})"
            )
        return
    dest.symlink_to(src, target_is_directory=True)


def is_link(p: Path) -> bool:
    """Whether p is a link of any kind this repo creates."""
    p = Path(p)
    return p.is_symlink() or (WINDOWS and p.is_junction())


def link_target(p: Path) -> Path | None:
    """What p points at, or None when p is not a link.

    os.readlink() on a Windows junction returns an extended-length path
    prefixed "\\\\?\\" (e.g. "\\\\?\\C:\\repo"). resolve() does not strip
    this -- it is preserved through resolve(), producing a path that
    compares unequal to the plain form link_dir() was given -- so the
    prefix is stripped here before resolving.

    A relative target is anchored to the link's own directory, which is what
    the OS does when it follows one. resolve() alone anchors it to the
    process's cwd instead, so the same link read from two directories gave
    two different answers -- and preflight, which runs from wherever the user
    invoked it, would have called a good link a copy.
    """
    p = Path(p)
    if not is_link(p):
        return None
    try:
        raw = os.readlink(p)
    except OSError:
        return None
    target = Path(raw.removeprefix("\\\\?\\"))
    return (target if target.is_absolute() else p.parent / target).resolve()


def verify_link(p: Path, repo: Path) -> bool:
    """Whether p is genuinely a link into repo, rather than a copy of it.

    The check exists because Git Bash's `ln -s` does not fail on a Windows
    machine without Developer Mode -- it deep-copies and reports success. A
    copy is worse than a failure: it looks installed, and then silently
    stops tracking edits to the repo.
    """
    p, repo = Path(p), Path(repo).resolve()
    if not p.exists() or not is_link(p):
        return False
    target = link_target(p)
    if target is None:
        return False
    return repo == target or repo in target.parents
