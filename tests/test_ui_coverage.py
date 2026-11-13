"""Must-preserve coverage for every area group: every inventory id and every defect with a test file leads a title of a
test that runs, in the file the map assigns (`python -m tests.ui_coverage --owner <group>` checks one area group's ids).
"""

from __future__ import annotations

from tests import ui_coverage
from tests.ui_support import REPO


def test_every_inventory_id_and_defect_has_a_test_in_its_assigned_file():
    """Every item of tests/ui_must_preserve.json, and every defect with a test file, is covered."""
    gaps = ui_coverage.missing()
    assert gaps == [], "\n".join(f"{item}\t{kind}\t{test}" for item, test, kind in gaps)


def test_the_map_is_whole_and_points_at_real_test_files():
    """The map lists each inventory id once, with its area group ("owner"); every test file it assigns exists (a missing file
    would count as uncovered, but say so plainly)."""
    data = ui_coverage.load_map()
    ids = [item["id"] for item in data["items"]]
    assert len(ids) == len(set(ids)) and len(ids) >= 296
    assert all(item.get("owner") for item in data["items"])
    assert all(defect.get("owner") for defect in data["defects"])
    files = {entry["test"] for entry in (*data["items"], *data["defects"]) if entry.get("test") not in (None, "", "any")}
    missing_files = sorted(f for f in files if not (REPO / f).is_file())
    assert missing_files == []
    n_items, n_defects = ui_coverage.counts()
    assert n_items == len(ids) and n_defects > 0
