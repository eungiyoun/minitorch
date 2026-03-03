# tests/conftest.py
import pytest

def pytest_addoption(parser):
    """
    Add a custom CLI option so you can run only a subset of op tests by op name.
    Examples:
      pytest --op add
      pytest --op add,mul
      pytest --op all
    """
    parser.addoption(
        "--op",
        action="store",
        default="all",
        help="Run only selected op tests. Example: --op add or --op add,mul or --op all",
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "op(name): mark test with op name")

def pytest_collection_modifyitems(config, items):
    """
    Filter collected tests based on the --op option.
    - If --op is "all" (default), run everything.
    - Otherwise, keep only tests marked with @pytest.mark.op("<name>")
      where <name> is one of the comma-separated values in --op.    
    """
    op = (config.getoption("--op") or "all").strip().lower()
    if op in ("", "all"):
        return

    want = {s.strip() for s in op.split(",") if s.strip()}

    selected = []
    for item in items:
        m = item.get_closest_marker("op")
        if m is None:
            continue
        name = str(m.args[0]).strip().lower()
        if name in want:
            selected.append(item)

    items[:] = selected
