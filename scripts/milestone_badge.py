"""Generate a shields.io endpoint badge for milestone (TODO) completion.

Runs the milestone-marked tests, counts how many pass vs. how many exist, and
writes ``milestones.json`` in the shields.io *endpoint* schema. CI publishes
that file to the ``badges`` branch; the README points a shields endpoint badge
at its raw URL, so the badge reads e.g. "milestones 5/8" and turns green when a
team has implemented every TODO.

    python scripts/milestone_badge.py        # -> milestones.json
"""

from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

XML = Path("milestone.xml")
OUT = Path("milestones.json")


def run_milestone_tests() -> None:
    # Don't fail the build if a milestone test fails or errors -- the badge is
    # meant to *report* progress, including partial/incorrect attempts.
    subprocess.run(
        [sys.executable, "-m", "pytest", "-m", "milestone", "-q", f"--junitxml={XML}"],
        check=False,
    )


def count() -> tuple[int, int]:
    """Return (done, total) over milestone test cases from the JUnit XML."""
    root = ET.parse(XML).getroot()
    total = done = 0
    for case in root.iter("testcase"):
        total += 1
        skipped = case.find("skipped") is not None
        failed = case.find("failure") is not None or case.find("error") is not None
        if not skipped and not failed:
            done += 1
    return done, total


def main() -> None:
    run_milestone_tests()
    done, total = count()
    if total == 0:
        message, color = "no milestones", "lightgrey"
    elif done == 0:
        message, color = f"{done}/{total}", "red"
    elif done < total:
        message, color = f"{done}/{total}", "yellow"
    else:
        message, color = f"{done}/{total}", "brightgreen"

    badge = {"schemaVersion": 1, "label": "milestones", "message": message, "color": color}
    OUT.write_text(json.dumps(badge))
    print(json.dumps(badge))


if __name__ == "__main__":
    main()
