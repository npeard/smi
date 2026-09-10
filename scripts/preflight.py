#!/usr/bin/env python
"""Verify a project is in a fit state to start work in.

Implements the master AGENTS.md's "Step 0" as one command, so the check
costs a single tool call rather than a handful of agent turns: pre-commit
wired up and current, tests green, on the default branch, clean tree.

Boundary with toolgaps.py, kept deliberately sharp because the two were
nearly duplicates: **preflight asks whether this repo is fit to work in
right now (state); toolgaps asks whether the project has the tooling it
should have at all (capability).** Pre-commit installation and test results
are state and belong here; "is there a type checker at all" does not.

Deliberately stdlib-only and project-agnostic -- it is meant to be copied
into new projects verbatim, per this repo's ``scripts/`` convention. Copied
alone it runs and reports; the two checks that ask what a link is on this
platform (hook registration and skill links) need ``platform_paths.py``
beside it and skip without it. It
assumes a current interpreter rather than degrading to whatever `python3`
the machine ships: every project gets a local pixi environment, and hooks
invoke that environment's python explicitly. See the interpreter check
below, which enforces the assumption instead of hoping for it.

Usage:
    python scripts/preflight.py [--with-tests] [--check-updates] [--strict]

Two checks are opt-in because the common caller is a fast, possibly
offline SessionStart hook:

``--with-tests``
    Actually run the detected test command. Off by default because a suite
    can take minutes; the default merely reports what it *would* run.
``--check-updates``
    Query upstream for newer pre-commit hook revs. Off by default because
    it needs network and costs a ``git ls-remote`` per configured repo.

Exit status is 0 unless ``--strict`` is passed, so that a hook can never
abort a session over a warning.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def _load_platform_paths():
    """The sibling platform module, or None when it was not copied along.

    Loaded by path rather than by name so resolution does not depend on
    sys.path, and so no sys.path mutation is needed -- `ruff --fix` would
    hoist one above the E402 boundary, and this repo ships zero
    suppressions. (An earlier version of this docstring cited the importlib
    pattern the tests use for *hyphenated* hook filenames as the precedent.
    That is a different problem: `platform_paths` is a legal identifier and
    could be imported by name. Loading by path is still the better fit here,
    for the sys.path reason, but the cited precedent did not support it.)

    None is not an error. This file's own docstring and the README both
    advertise it as copyable into a new project verbatim, and
    platform_paths.py is deliberately not on that list -- it is a library
    module. Loading it unconditionally broke that promise the loudest way
    available: a FileNotFoundError traceback from module scope, before a
    single check ran, in a script whose exit contract is 0 unless --strict
    precisely so a SessionStart hook can never abort a session over a
    warning. The checks that need it now skip, which is the idiom every
    other optional sibling here already uses.
    """
    path = Path(__file__).resolve().parent / "platform_paths.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("platform_paths", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_installation_contract():
    """The installer contract, or None when it was not copied along."""
    path = Path(__file__).resolve().parent / "installation_contract.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("installation_contract", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


platform_paths = _load_platform_paths()
installation_contract = _load_installation_contract()

OK, WARN, FAIL = "ok", "warn", "fail"


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, label: str, detail: str = "") -> None:
        self.rows.append((status, label, detail))

    def render(self) -> int:
        width = max(len(s) for s, _, _ in self.rows)
        for status, label, detail in self.rows:
            line = f"[{status:<{width}}] {label}"
            print(f"{line}: {detail}" if detail else line)
        counts = {s: sum(1 for r in self.rows if r[0] == s) for s in (OK, WARN, FAIL)}
        print(
            f"\n{counts[OK]} ok, {counts[WARN]} warning(s), {counts[FAIL]} failure(s)"
        )
        return counts[FAIL] + counts[WARN]


def git(*args: str) -> str | None:
    """Run a git command, returning None rather than raising on failure."""
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=20
        )
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
    ):
        # A deadline matters because --check-updates runs one ls-remote per
        # configured repo, and on a captive portal those block indefinitely.
        return None
    return out.stdout.strip()


def check_repo(report: Report) -> bool:
    if git("rev-parse", "--is-inside-work-tree") != "true":
        report.add(FAIL, "git repository", "not inside a work tree")
        return False
    report.add(OK, "git repository")
    return True


def default_branch(root: Path | None = None) -> str:
    """Best-effort default-branch name, without assuming a remote exists.

    `root` scopes the lookup to a specific checkout. check_audit_owed needs
    that: resolved from the process cwd instead, it answered about whichever
    repo preflight was invoked from.
    """
    at = ("-C", str(root)) if root else ()
    head = git(*at, "symbolic-ref", "--short", "refs/remotes/origin/HEAD")
    if head:
        return head.split("/", 1)[-1]
    for candidate in ("main", "master"):
        if git(*at, "rev-parse", "--verify", "--quiet", f"refs/heads/{candidate}"):
            return candidate
    return "main"


def check_branch(report: Report) -> None:
    current, target = git("rev-parse", "--abbrev-ref", "HEAD"), default_branch()
    if current == target:
        report.add(OK, "on default branch", current)
    else:
        # A feature branch is normal mid-task, so this is informational --
        # it only matters when starting fresh work.
        report.add(WARN, "on default branch", f"on '{current}', not '{target}'")


def check_clean_tree(report: Report) -> None:
    dirty = git("status", "--porcelain")
    if dirty is None:
        report.add(FAIL, "clean working tree", "could not read status")
    elif dirty:
        n = len(dirty.splitlines())
        report.add(WARN, "clean working tree", f"{n} uncommitted change(s)")
    else:
        report.add(OK, "clean working tree")


def declared_floor(root: Path) -> tuple[int, int] | None:
    """The project's minimum Python, from pixi.toml or pyproject.toml."""
    for name, key in (("pixi.toml", "python"), ("pyproject.toml", "requires-python")):
        path = root / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        # Anchored to a line start: unanchored, `python` matched the tail of
        # `ipython = ">=8.0"` and read as a floor of 8.0 -- and a failure here
        # short-circuits every later check.
        match = re.search(
            rf'^\s*{key}\s*=\s*["\x27][^"\x27]*?(\d+)\.(\d+)',
            text,
            re.MULTILINE,
        )
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


# Conventional in-checkout environment directories. A named conda env or a
# ~/.virtualenvs entry is deliberately absent: those live outside the
# checkout by design, and the point below is to tell "not local" apart from
# "this project does not work that way".
LOCAL_ENV_DIRS = (".pixi", ".venv", "venv", "env")


def is_local_env(prefix: Path, root: Path) -> bool:
    """Whether `prefix` is an environment belonging to the checkout at `root`.

    Each candidate is resolved because a git worktree commonly symlinks
    `.pixi` at the parent checkout's environment: compared as written, the
    worktree's own interpreter read as foreign, and a foreign interpreter
    used to discard every remaining check.
    """
    owned = [root.resolve()]
    owned += [
        (root / name).resolve() for name in LOCAL_ENV_DIRS if (root / name).is_dir()
    ]
    return any(
        prefix == candidate or candidate in prefix.parents for candidate in owned
    )


def check_interpreter(report: Report, root: Path) -> bool:
    """Verify the interpreter is project-local and current.

    Every project gets a local environment with a recent interpreter, rather
    than inheriting whatever `python3` the machine ships -- on macOS that is
    still 3.9, which quietly pushes scripts and hooks towards contortions
    for a constraint nobody chose. Checking it here makes the requirement
    mechanical instead of a line of prose that decays.

    Returns whether the remaining checks can be trusted to run, which is a
    question about the *version* only. Locality used to stop them too, so a
    copy of this script into any conda-based project reported one failure and
    inspected nothing -- no branch, tree, pre-commit or test check at all.
    """
    running = sys.version_info[:2]
    prefix = Path(sys.prefix).resolve()
    version = f"{running[0]}.{running[1]}"
    floor = declared_floor(root)
    current = floor is None or running >= floor

    if not is_local_env(prefix, root):
        detail = (
            f"{version} from {prefix} is not a project-local env "
            "(run via the project's task runner)"
        )
        if not current:
            # Named in the same row rather than a second one, so the reason
            # the remaining checks were skipped is visible.
            detail += f"; also below the declared floor {floor[0]}.{floor[1]}"
        # A failure only where the project declares a pixi environment: there,
        # an inherited interpreter is how scripts end up contorted for
        # whatever version the machine ships. Elsewhere it is how the project
        # is built, and preflight is advertised as copyable verbatim.
        report.add(
            FAIL if (root / "pixi.toml").is_file() else WARN, "interpreter", detail
        )
    elif floor is None:
        report.add(WARN, "interpreter", f"{version}, but no floor is declared")
    elif not current:
        report.add(
            FAIL,
            "interpreter",
            f"{version} is below the declared floor {floor[0]}.{floor[1]}",
        )
    else:
        report.add(
            OK, "interpreter", f"{version}, local env, floor {floor[0]}.{floor[1]}"
        )
    return current


def check_precommit_installed(report: Report, root: Path) -> None:
    if not (root / ".pre-commit-config.yaml").is_file():
        report.add(WARN, "pre-commit configured", "no .pre-commit-config.yaml")
        return
    report.add(OK, "pre-commit configured")

    # Resolve via git rather than assuming root/".git" is a directory: in a
    # worktree or submodule it is a file pointing elsewhere, and the hooks
    # live in the parent repo.
    # Run with -C so the returned path is relative to root rather than to
    # wherever this process happens to have been invoked from.
    hook_path = git("-C", str(root), "rev-parse", "--git-path", "hooks/pre-commit")
    hook = (root / hook_path) if hook_path else None
    if (
        hook
        and hook.is_file()
        and "pre-commit" in hook.read_text(encoding="utf-8", errors="replace")
    ):
        report.add(OK, "pre-commit hook installed")
    else:
        report.add(FAIL, "pre-commit hook installed", "run: pre-commit install")


def configured_revs(config: Path) -> list[tuple[str, str]]:
    """Pair each configured repo URL with its pinned rev.

    Keys are collected independently and paired by position rather than by
    requiring `rev:` to follow `repo:` immediately. pre-commit accepts either
    order and permits other keys between them, and a stricter reader silently
    dropped whole entries -- which then read as "hook revs current" while an
    unchecked repo went stale.

    Regex rather than a YAML parse to keep this stdlib-only; a miss degrades
    to zero pairs, which the caller reports as "cannot check".
    """
    keys: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(
        config.read_text(encoding="utf-8", errors="replace").splitlines()
    ):
        stripped = line.strip().lstrip("-").strip()
        for kind in ("repo", "rev"):
            prefix = f"{kind}:"
            if stripped.startswith(prefix):
                value = stripped[len(prefix) :].strip().strip("'\"")
                if value:
                    keys.append((lineno, kind, value))
                break

    repos = [(i, v) for i, kind, v in keys if kind == "repo"]
    revs = [(i, v) for i, kind, v in keys if kind == "rev"]

    pairs: list[tuple[str, str]] = []
    consumed: set[int] = set()
    for position, (lineno, url) in enumerate(repos):
        # An entry's rev lies between the previous repo key and the next one,
        # on either side of the repo key itself.
        lower = repos[position - 1][0] if position else -1
        upper = repos[position + 1][0] if position + 1 < len(repos) else 10**9
        candidates = [
            (i, v) for i, v in revs if lower < i < upper and i not in consumed
        ]
        if candidates and url.startswith("http"):
            # Consume the match: without this, an entry whose rev precedes its
            # repo left that rev available to the *next* entry, which then
            # reported the wrong pin.
            consumed.add(candidates[0][0])
            pairs.append((url, candidates[0][1]))
    return pairs


def latest_tag(url: str) -> tuple[str | None, str | None]:
    """(newest semver tag, problem) -- exactly one of the two is set.

    The two failure modes are reported separately because they point the
    reader somewhere different: "unreachable" means check the network,
    while "no-semver-tags" means this repo pins something this parser
    cannot compare and the rev must be checked by hand. Collapsing both to
    None diagnosed a repo tagged `v1.0` as a network problem.
    """
    out = git("ls-remote", "--tags", "--refs", url)
    if out is None:
        return None, "unreachable"
    if not out:
        # Reachable, exit 0, no tags at all. Not a network problem, and
        # reporting it as one sends the reader to the wrong place.
        return None, "no-semver-tags"
    tags = [line.rsplit("/", 1)[-1] for line in out.splitlines()]

    def key(tag: str) -> tuple[int, ...] | None:
        m = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", tag)
        return tuple(int(g) for g in m.groups()) if m else None

    ranked = sorted((k, t) for t in tags if (k := key(t)) is not None)
    if not ranked:
        return None, "no-semver-tags"
    return ranked[-1][1], None


def check_hook_revs(report: Report, root: Path) -> None:
    config = root / ".pre-commit-config.yaml"
    if not config.is_file():
        return
    configured = configured_revs(config)
    if not configured:
        # The regex found nothing, so there is no basis for a clean answer.
        # Reporting "current" here would be a false pass on a config style
        # this parser does not recognize.
        report.add(WARN, "hook revs current", "could not parse any repo/rev pairs")
        return

    stale, unreachable, untagged = [], [], []
    for url, rev in configured:
        name = url.rsplit("/", 1)[-1]
        latest, problem = latest_tag(url)
        if problem == "unreachable":
            # --check-updates is opt-in precisely because it needs network,
            # so a failed lookup is surfaced rather than folded into "ok".
            unreachable.append(name)
        elif problem == "no-semver-tags":
            untagged.append(name)
        elif latest and latest.lstrip("v") != rev.lstrip("v"):
            stale.append(f"{name} {rev} -> {latest}")
    if stale:
        report.add(
            WARN, "hook revs current", "; ".join(stale) + " (pre-commit autoupdate)"
        )
    elif unreachable:
        report.add(
            WARN, "hook revs current", f"lookup failed: {', '.join(unreachable)}"
        )
    elif untagged:
        report.add(
            WARN,
            "hook revs current",
            f"no comparable tags, check by hand: {', '.join(untagged)}",
        )
    else:
        report.add(OK, "hook revs current")


def _pixi_has_test_task(path: Path) -> bool:
    """Whether pixi.toml defines a `test` task.

    tomllib is imported here rather than at module scope for one specific
    reason: this script has to be able to *run* on an interpreter too old to
    have it, so that check_interpreter can report that fact instead of the
    module dying on import. A diagnostic that cannot execute under the
    condition it diagnoses is useless.
    """
    import tomllib

    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError):
        return False
    tasks = dict(data.get("tasks", {}))
    for feature in data.get("feature", {}).values():
        tasks.update(feature.get("tasks", {}))
    return "test" in tasks


def _npm_has_test_script(path: Path) -> bool:
    try:
        return "test" in json.loads(path.read_text(encoding="utf-8")).get("scripts", {})
    except (ValueError, OSError):
        return False


def _mentions_test(path: Path) -> bool:
    """Crude fallback for formats with no cheap stdlib parser (YAML)."""
    return "test" in path.read_text(encoding="utf-8")


# Priority order: a project's own task runner knows more than a bare pytest
# invocation does (env activation, flags, coverage config). Each entry owns
# its own detector, so adding a runner is one line here and nothing else.
TEST_RUNNERS = (
    ("pixi.toml", "pixi run test", _pixi_has_test_task),
    ("Taskfile.yml", "task test", _mentions_test),
    ("package.json", "npm test", _npm_has_test_script),
)


def check_friction(report: Report, root: Path) -> None:
    """Surface recurring workflow friction, if the miner is present.

    Silent when friction.py is absent: preflight is meant to be copied into
    projects that have no such script, and a missing optional companion is
    not a finding about the project.
    """
    miner = Path(__file__).resolve().parent / "friction.py"
    if not miner.is_file():
        return
    try:
        out = subprocess.run(
            [sys.executable, str(miner), "--json"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        data = json.loads(out.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        report.add(WARN, "friction", "could not run friction.py")
        return

    if warning := data.get("ledger_warning"):
        report.add(WARN, "friction", warning)
    # Surface the denominator. A check that reports only actionable_count
    # hides its own coverage: while BENIGN_EXIT was mis-anchored, 45 of 61
    # errors were filed as benign, unclassified read a reassuring 5, and
    # preflight printed "nothing over the bar" over a classifier that could
    # not see three quarters of its input.
    seen = data.get("errors_seen", 0)
    unclassified = data.get("unclassified", 0)
    if seen and unclassified * 4 >= seen:
        report.add(
            WARN,
            "friction",
            f"{unclassified}/{seen} errors match no class; the classifier is "
            "behind its input (pixi run friction --all)",
        )
    n = data.get("actionable_count", 0)
    if n:
        classes = ", ".join(data.get("actionable", [])[:3])
        more = "..." if n > 3 else ""
        report.add(
            WARN,
            "friction",
            f"{n} class(es) over bar: {classes}{more} (pixi run friction)",
        )
    else:
        report.add(OK, "friction", "nothing over the bar")


def check_audit_owed(report: Report, root: Path) -> None:
    """Report a config audit this branch owes but has not run.

    Branch state. The obligation is an audit for what *this* branch changed --
    unscoped, the warning follows you to an unrelated branch and clearing it
    there discards the original branch's obligation. The marker is gitignored
    and untracked, so no test can gate it and it does not travel between
    machines. Reported here because session start is when it matters: the hook
    that wrote it injected its reminder into a session that has since ended.

    The marker is parsed by audit_assets.owed_assets rather than here. An
    earlier version reimplemented the parse to keep preflight free of project
    structure, but a discharge is only answered by re-hashing the asset, and
    two readers deciding that separately meant one warning about an audit the
    other considered done. Imported, not shelled out: a subprocess in a startup
    check is the cost the earlier version was avoiding, and an import of a
    sibling module in the same directory is not the coupling the rule is about.

    Silent when nothing is owed for the current branch, and silent in projects
    with no audit script, since preflight is copied into repos that have no
    such concept.
    """
    if not (Path(__file__).resolve().parent / "audit_assets.py").is_file():
        return
    if not (root / ".audit-owed").is_file():
        return
    try:
        import audit_assets
    except ImportError:
        return
    # -C root, like check_precommit_installed. Reading the branch from the
    # process cwd made this report on whichever repo preflight happened to be
    # invoked from, and made its own tests depend on the branch the checkout
    # was on -- two of them failed on `main`, which is precisely the state
    # step 0 requires to be green.
    branch = git("-C", str(root), "rev-parse", "--abbrev-ref", "HEAD") or ""
    # On the default branch, report every branch's entries, not just this
    # one's. An obligation recorded against feat/x whose work has been merged
    # is now an obligation about the default branch's contents, and reporting
    # only matching lines meant merging without auditing lost it silently.
    on_default = bool(branch) and branch == default_branch(root)
    try:
        assets = audit_assets.owed_assets(root, branch, any_branch=on_default)
    except OSError:
        return
    if not assets:
        return
    report.add(
        WARN,
        "audit",
        f"{len(assets)} config asset(s) changed on this branch; "
        "run `pixi run audit` and the config-audit skill before integrating",
    )


def check_hooks(report: Report, root: Path) -> None:
    """Report hooks declared in the repo but not registered on this machine.

    Registration is machine state, not repo state -- a fresh clone correctly
    has none -- so it cannot be gated by a test. Session start is when it
    matters, because an unregistered hook is silently absent rather than
    broken. Silent when the repo has no registrar, since preflight is copied
    into projects that have no hooks at all.
    """
    registrar = Path(__file__).resolve().parent / "register_hooks.py"
    if not registrar.is_file():
        return
    try:
        out = subprocess.run(
            [sys.executable, str(registrar), "--check"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        report.add(WARN, "hooks registered", "could not run register_hooks.py")
        return

    if out.returncode == 0:
        report.add(
            OK,
            "hooks registered",
            out.stdout.strip().splitlines()[-1:][0]
            if out.stdout.strip()
            else "all current",
        )
        return
    drift = [line.strip() for line in out.stdout.splitlines() if line.startswith("  ")]
    report.add(
        WARN,
        "hooks registered",
        f"{len(drift)} out of date: {'; '.join(drift[:2])} ({_install_hint()})",
    )


def check_codex_hooks(report: Report, root: Path) -> None:
    """Report Codex hook configuration and the separately persisted trust state."""
    registrar = Path(__file__).resolve().parent / "register_codex_hooks.py"
    if not registrar.is_file():
        return
    command = [sys.executable, str(registrar), "--check"]
    try:
        out = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        report.add(
            WARN,
            "Codex hooks configured",
            "could not run register_codex_hooks.py",
        )
    else:
        if out.returncode == 0:
            report.add(
                OK,
                "Codex hooks configured",
                out.stdout.strip().splitlines()[-1:][0]
                if out.stdout.strip()
                else "all current",
            )
        else:
            report.add(
                WARN,
                "Codex hooks configured",
                f"out of date ({' '.join(command)})",
            )
    report.add(
        WARN,
        "Codex hooks trusted",
        "verify persisted host trust for this repository's /hooks directory",
    )


def _install_hint() -> str:
    """The installer command for this platform, for use in remediation hints.

    Git Bash's `ln -s` silently deep-copies instead of failing on a Windows
    machine without Developer Mode, so the two platforms need different
    installers -- telling a Windows user to run install.sh names the wrong
    command entirely, not just a cosmetic mismatch.

    Falls back to os.name when the sibling was not copied along. That is the
    one platform fact this file can answer for itself -- it is what
    platform_paths.WINDOWS is defined as -- and a remediation hint is worth
    more approximately right than absent.
    """
    windows = platform_paths.WINDOWS if platform_paths else os.name == "nt"
    return "./install.ps1" if windows else "./install.sh"


def installation_checkout(root: Path) -> Path:
    """Resolve the main checkout without assuming where Git stores its metadata."""
    root = root.resolve()
    if not (root / ".git").is_file():
        return root
    # The first porcelain record is the main checkout. NUL delimiters preserve
    # spaces and avoid Git's quoting of unusual paths; a bare repo owns no install.
    listing = git("-C", str(root), "worktree", "list", "--porcelain", "-z")
    if listing:
        main = listing.split("\0\0", 1)[0].split("\0")
        if main[0].startswith("worktree ") and "bare" not in main:
            return Path(main[0].removeprefix("worktree ")).resolve()
    return root


def check_instructions(report: Report, root: Path, home: Path) -> None:
    """Report generated instruction adapters that are missing or stale."""
    root = installation_checkout(root)
    if (
        installation_contract is None
        or platform_paths is None
        or not (root / "install.py").is_file()
        or not (root / installation_contract.GUIDANCE_NAME).is_file()
    ):
        return
    for label, (dest, expected) in zip(
        ("Claude instructions", "Codex instructions"),
        installation_contract.instruction_adapters(root, home),
        strict=True,
    ):
        try:
            current = (
                dest.is_file()
                and not platform_paths.is_link(dest)
                and dest.read_text(encoding="utf-8") == expected
            )
        except (OSError, UnicodeError):
            current = False
        if current:
            report.add(OK, label, f"{dest}: current")
        else:
            report.add(
                WARN,
                label,
                f"{dest}: missing or stale ({_install_hint()})",
            )


def check_skills(
    report: Report, root: Path, installed: Path, label: str = "skills linked"
) -> None:
    """Report skills the repo carries that this machine cannot see.

    Machine state, like hook registration: skills reach a session only through
    the links install.py writes, so a skill added without re-running it is
    invisible to every session while the whole suite stays green -- the same
    "manual step nothing verifies" class as the hook that shipped inert. A
    copy is reported separately from a missing link: it looks installed
    because `.exists()` cannot tell it apart from a real link, but it tracks
    nothing, and Git Bash's `ln -s` produces exactly this on Windows without
    Developer Mode instead of failing outright.

    A link must target the named skill in this checkout or its main checkout:
    sessions in linked worktrees still consume the main installation. Silent
    in projects with no installer, since preflight is copied into other repos.
    """
    source = root / "skills"
    if not source.is_dir() or not (root / "install.py").is_file():
        return
    # The optional sibling owns symlink and Windows junction handling.
    if platform_paths is None:
        return
    carried = {p.name for p in source.iterdir() if p.is_dir()}
    # exists() follows the link, so a dangling one reads as absent -- which is
    # what it is, from a session's point of view.
    unlinked = sorted(name for name in carried if not (installed / name).exists())
    copies = sorted(
        name
        for name in carried
        if (installed / name).exists() and not platform_paths.is_link(installed / name)
    )
    source_roots = {
        source.resolve(),
        (installation_checkout(root) / "skills").resolve(),
    }
    wrong = sorted(
        name
        for name in carried
        if (installed / name).exists()
        and platform_paths.is_link(installed / name)
        and platform_paths.link_target(installed / name)
        not in {(candidate / name).resolve() for candidate in source_roots}
    )
    dangling = sorted(
        p.name
        for p in (installed.iterdir() if installed.is_dir() else [])
        if platform_paths.is_link(p) and not p.exists() and p.name not in carried
    )
    if not unlinked and not copies and not wrong and not dangling:
        report.add(OK, label, f"{installed}: {len(carried)} linked")
        return
    parts = []
    if unlinked:
        parts.append(f"not linked: {', '.join(unlinked)}")
    if copies:
        parts.append(f"copy, not a link: {', '.join(copies)}")
    if wrong:
        parts.append(f"wrong target: {', '.join(wrong)}")
    if dangling:
        parts.append(f"stale link: {', '.join(dangling)}")
    report.add(
        WARN,
        label,
        f"{installed}: {'; '.join(parts)} ({_install_hint()})",
    )


def detect_test_command(root: Path) -> str | None:
    for filename, command, defines_test in TEST_RUNNERS:
        path = root / filename
        if path.is_file() and defines_test(path):
            return command
    if (root / "tests").is_dir() or list(root.glob("test_*.py")):
        return "pytest"
    return None


def check_tests(report: Report, root: Path, run: bool) -> None:
    command = detect_test_command(root)
    if command is None:
        report.add(WARN, "tests", "no test suite configured")
        return
    if not run:
        report.add(OK, "tests", f"'{command}' detected (use --with-tests to run)")
        return
    try:
        result = subprocess.run(command.split(), cwd=root, check=False)
    except FileNotFoundError:
        report.add(FAIL, "tests", f"'{command}' not found on PATH")
        return
    if result.returncode == 0:
        report.add(OK, "tests", f"'{command}' passed")
    else:
        report.add(FAIL, "tests", f"'{command}' failed (exit {result.returncode})")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--with-tests", action="store_true", help="run the suite")
    parser.add_argument(
        "--check-updates", action="store_true", help="query upstream hook revs"
    )
    parser.add_argument(
        "--strict", action="store_true", help="exit nonzero on any warning or failure"
    )
    parser.add_argument(
        "--no-friction",
        action="store_true",
        help="skip the friction report (on by default: local and fast)",
    )
    args = parser.parse_args(argv)

    report = Report()
    if not check_repo(report):
        report.render()
        return 1 if args.strict else 0

    root = Path(git("rev-parse", "--show-toplevel") or ".")
    if not check_interpreter(report, root):
        # An interpreter below the project's floor makes later checks
        # unrunnable, not merely untrustworthy: detect_test_command imports
        # tomllib. Report and stop rather than dying mid-run.
        report.render()
        return 1 if args.strict else 0

    check_branch(report)
    check_clean_tree(report)
    check_precommit_installed(report, root)
    if args.check_updates:
        check_hook_revs(report, root)
    check_tests(report, root, run=args.with_tests)
    check_hooks(report, root)
    check_codex_hooks(report, root)
    check_instructions(report, root, Path.home())
    if installation_contract is not None:
        for label, installed in zip(
            ("Claude skills linked", "Portable skills linked"),
            installation_contract.skill_destinations(Path.home()),
            strict=True,
        ):
            check_skills(report, root, installed, label=label)
    if not args.no_friction:
        check_friction(report, root)
    check_audit_owed(report, root)

    problems = report.render()
    return 1 if (args.strict and problems) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
