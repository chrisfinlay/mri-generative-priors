"""Shared test helpers.

Many tests exercise functions that ship as TODOs on the starter branch. We want
`pixi run test` to be green out of the box (so the harness/CI is known-good) and
to *automatically start enforcing* each test as students implement the function.
``skip_if_todo`` skips a test only while the target still raises
NotImplementedError; once implemented, the real assertions run.
"""

import functools

import pytest


def skip_if_todo(fn):
    """Skip while the target is a TODO, and tag it as a `milestone`.

    The ``milestone`` marker lets the badge tooling count how many TODOs a team
    has completed (``pytest -m milestone``), so the two stay in sync
    automatically: every TODO-gated test is exactly one milestone.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as e:
            pytest.skip(f"TODO not implemented yet: {e}")

    return pytest.mark.milestone(wrapper)
