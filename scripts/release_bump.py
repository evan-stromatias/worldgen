"""Release bookkeeping after a PR merge: version bump and changelog entry.

The bump part comes from the merged PR's title prefix (``major: ``, ``minor: `` or ``patch: ``);
the changelog entry is the title without the prefix plus any bullet lines in the PR body.
Intended to run in CI on a checkout of main, where ``uv version --bump`` rewrites
``pyproject.toml`` and re-locks ``uv.lock`` in one step.
"""

import os
import re
import subprocess
import sys
import tomllib
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TITLE_RE = re.compile(r"^(major|minor|patch): +(?P<summary>.+)")


def parse_title(title: str) -> tuple[str, str]:
    """Return the bump part and changelog summary from a PR title."""
    match = TITLE_RE.match(title.strip())
    if match is None:
        msg = f"PR title must start with 'major: ', 'minor: ' or 'patch: ', got: {title!r}"
        raise ValueError(msg)
    return match.group(1), match.group("summary").strip()


def body_bullets(body: str) -> list[str]:
    """Bullet lines in the PR body become extra changelog bullets."""
    return [line.strip() for line in body.splitlines() if line.strip().startswith("- ")]


def bump_version(part: str) -> str:
    """Bump the project version with uv and read the result back from pyproject.toml."""
    subprocess.run(["uv", "version", "--bump", part], cwd=ROOT, check=True)
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def prepend_changelog(version: str, summary: str, bullets: list[str]) -> None:
    """Add the dated entry at the top of CHANGELOG.md, matching the existing format."""
    path = ROOT / "CHANGELOG.md"
    lines = [f"{date.today():%Y-%m-%d}, {version}:", f"  - {summary}"]
    lines.extend(f"  {bullet}" for bullet in bullets)
    path.write_text("\n".join(lines) + "\n\n" + path.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> int:
    try:
        part, summary = parse_title(os.environ.get("PR_TITLE", ""))
    except ValueError as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    version = bump_version(part)
    prepend_changelog(version, summary, body_bullets(os.environ.get("PR_BODY", "")))
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output is not None:
        with open(github_output, "a", encoding="utf-8") as handle:
            handle.write(f"version={version}\n")
    print(f"Bumped version to {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
