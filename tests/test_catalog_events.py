from manga_repaint.catalog import Catalog


def test_job_activity_uses_index_without_scanning_unrelated_history(tmp_path):
    catalog = Catalog(tmp_path)
    first = catalog.add_event("job_ready", {"value": 1}, "book")
    catalog.add_event("gpu_metrics", {"value": 2})
    last = catalog.add_event("page_ready", {"value": 3}, "book")
    with catalog.connect() as connection:
        plan = connection.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM events WHERE job_id=? ORDER BY id DESC LIMIT ?",
            ("book", 60),
        ).fetchall()
    details = " ".join(str(row["detail"]) for row in plan)
    assert "SEARCH events USING INDEX idx_events_job_id" in details
    assert "SCAN events" not in details
    assert [item["id"] for item in catalog.recent_events("book")] == [first, last]
    assert [item["id"] for item in catalog.recent_events("book", limit=1)] == [last]
    # The index is additive and startup may safely run again without data changes.
    reopened = Catalog(tmp_path)
    assert reopened.latest_event_id() == last
    assert reopened.recent_events("book") == catalog.recent_events("book")
