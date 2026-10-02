"""Tests for atomic billing-cycle closing (immutable close snapshots).

Like the batch settlement tests, these run against a temporary *file-backed*
SQLite database (WAL, real write locks) so that concurrent close/batch races
serialize exactly as in production.
"""
from __future__ import annotations

import os
import threading
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from fastapi.testclient import TestClient  # noqa: E402

from app import settlement  # noqa: E402
from app.database import (  # noqa: E402
    Base,
    BillingCycle,
    CycleClose,
    CycleInterval,
    UsageBatch,
    configure_sqlite_engine,
    seed,
)
from app.importer import canonical_json  # noqa: E402
from app.main import app  # noqa: E402
from app.pricing import PricingError  # noqa: E402
from app.schemas import CycleCloseRequest, QuoteIntervalIn  # noqa: E402

V = "us-ny-demo-v2024"
# 2024-07-06 is a Saturday in NY -> only the `leisure` rule
# (tiers: 100 @0.15, 300 @0.18, open @0.22).
DAY1 = {
    "start": "2024-07-06T00:00:00-04:00",
    "end": "2024-07-06T01:00:00-04:00",
    "timezone": "America/New_York",
    "energy": "60",
}
DAY2 = {
    "start": "2024-07-06T01:00:00-04:00",
    "end": "2024-07-06T02:00:00-04:00",
    "timezone": "America/New_York",
    "energy": "60",
}
DAY3 = {
    "start": "2024-07-06T02:00:00-04:00",
    "end": "2024-07-06T03:00:00-04:00",
    "timezone": "America/New_York",
    "energy": "60",
}


@pytest.fixture()
def tmp_engine(tmp_path):
    path = tmp_path / "close.db"
    eng = create_engine(
        f"sqlite:///{path}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    configure_sqlite_engine(eng)
    Base.metadata.create_all(eng)
    with Session(eng) as s:
        seed(s)
        s.commit()
    yield eng, path
    eng.dispose()


def _submit(eng, cycle, key, rev, intervals, version=V):
    class _Req:
        def __init__(self):
            self.version = version
            self.idempotency_key = key
            self.expected_revision = rev
            self.intervals = [QuoteIntervalIn(**iv) for iv in intervals]

    return settlement.submit_batch(cycle, _Req(), engine=eng)


def _close(eng, cycle, close_id, rev, amount):
    req = CycleCloseRequest(close_id=close_id, expected_revision=rev,
                            expected_total_amount=Decimal(amount))
    return settlement.close_cycle(cycle, req, engine=eng)


def _seed_open_cycle(eng, cycle="c1", n=2):
    _submit(eng, cycle, "k1", 0, [DAY1])
    if n >= 2:
        _submit(eng, cycle, "k2", 1, [DAY2])
    if n >= 3:
        _submit(eng, cycle, "k3", 2, [DAY3])


# ------------------------------------------------------------- happy path


def test_close_success_writes_immutable_snapshot(tmp_engine):
    eng, _ = tmp_engine
    _seed_open_cycle(eng, n=2)
    res = _close(eng, "c1", "close-1", 2, "18.60")
    assert res.replayed is False
    body = res.body
    assert body["status"] == "closed"
    assert body["close_id"] == "close-1"
    assert body["revision"] == 2  # close does not advance revision
    assert body["expected_total_amount"] == "18.60"
    assert body["total_amount"] == "18.60"
    assert Decimal(body["total_energy"]) == Decimal("120")
    assert [b["revision"] for b in body["batches"]] == [1, 2]
    # intervals sorted by UTC start (stable detail order)
    starts = [i["start"] for i in body["intervals"]]
    assert starts == sorted(starts) and len(starts) == 2
    with Session(eng) as s:
        cy = s.get(BillingCycle, "c1")
        assert cy.status == "closed"
        assert cy.closed_at is not None
        assert cy.revision == 2
        row = s.get(CycleClose, "c1")
        assert row is not None and row.close_id == "close-1"
        assert row.detail_digest == body["detail_digest"]


def test_detail_digest_covers_batches_and_intervals(tmp_engine):
    eng, _ = tmp_engine
    _seed_open_cycle(eng, n=2)
    res = _close(eng, "c1", "close-1", 2, "18.60")
    # Independently rebuild the canonical detail document from stored rows.
    with Session(eng) as s:
        import hashlib
        cy = s.get(BillingCycle, "c1")
        batches = s.scalars(
            select(UsageBatch).where(UsageBatch.cycle_id == "c1")
            .order_by(UsageBatch.revision, UsageBatch.id)).all()
        intervals = s.scalars(
            select(CycleInterval).where(CycleInterval.cycle_id == "c1")
            .order_by(CycleInterval.start_utc, CycleInterval.end_utc,
                      CycleInterval.seq)).all()
        doc = {
            "version": cy.version_code,
            "revision": cy.revision,
            "total_energy": format(Decimal(cy.total_energy).normalize(), "f"),
            "total_amount": format(Decimal(cy.total_amount).normalize(), "f"),
            "batches": [
                {"revision": b.revision, "idempotency_key": b.idempotency_key,
                 "version": b.version_code, "content_digest": b.content_digest,
                 "batch_energy": format(Decimal(b.batch_energy).normalize(), "f"),
                 "batch_amount": format(Decimal(b.batch_amount).normalize(), "f")}
                for b in batches],
            "intervals": [
                {"seq": r.seq, "revision": r.revision,
                 "start": r.start_utc, "end": r.end_utc,
                 "timezone": r.timezone,
                 "energy": format(Decimal(r.energy).normalize(), "f")}
                for r in intervals],
        }
        digest = hashlib.sha256(canonical_json(doc).encode()).hexdigest()
    assert digest == res.body["detail_digest"]


# ------------------------------------------------------------- replay / conflict


def test_same_close_id_same_content_replays_snapshot(tmp_engine):
    eng, _ = tmp_engine
    _seed_open_cycle(eng, n=2)
    first = _close(eng, "c1", "close-1", 2, "18.60").body
    # stale expected_revision still replays (idempotency precedes revision)
    replay = _close(eng, "c1", "close-1", 99, "18.60")
    assert replay.replayed is True
    assert replay.body == first
    with Session(eng) as s:
        assert s.get(BillingCycle, "c1").revision == 2
        assert len(s.scalars(select(CycleClose)).all()) == 1


def test_same_close_id_different_amount_conflicts(tmp_engine):
    eng, _ = tmp_engine
    _seed_open_cycle(eng, n=2)
    _close(eng, "c1", "close-1", 2, "18.60")
    with pytest.raises(PricingError) as ei:
        _close(eng, "c1", "close-1", 2, "18.61")
    assert ei.value.code == "close_idempotency_conflict"
    assert ei.value.details["close_id"] == "close-1"
    # original snapshot untouched
    got = settlement.get_close("c1", engine=eng)
    assert got["expected_total_amount"] == "18.60"


def test_different_close_id_on_closed_cycle_conflicts(tmp_engine):
    eng, _ = tmp_engine
    _seed_open_cycle(eng, n=2)
    _close(eng, "c1", "close-1", 2, "18.60")
    with pytest.raises(PricingError) as ei:
        _close(eng, "c1", "close-2", 2, "18.60")
    assert ei.value.code == "cycle_closed"
    assert ei.value.details["close_id"] == "close-1"


def test_close_id_is_globally_unique(tmp_engine):
    eng, _ = tmp_engine
    _seed_open_cycle(eng, "c1", n=1)
    _seed_open_cycle(eng, "c2", n=1)
    _close(eng, "c1", "shared-close", 1, "9.00")
    with pytest.raises(PricingError) as ei:
        _close(eng, "c2", "shared-close", 1, "9.00")
    assert ei.value.code == "close_id_in_use"
    assert ei.value.details["other_cycle_id"] == "c1"
    # c2 remains open
    with Session(eng) as s:
        assert s.get(BillingCycle, "c2").status == "open"
        assert s.get(CycleClose, "c2") is None


# ------------------------------------------------------------- preconditions


def test_close_unknown_cycle_404(tmp_engine):
    eng, _ = tmp_engine
    with pytest.raises(PricingError) as ei:
        _close(eng, "ghost", "x", 0, "0.00")
    assert ei.value.status_code == 404
    assert ei.value.code == "cycle_not_found"


def test_close_stale_revision_conflict_and_state_untouched(tmp_engine):
    eng, _ = tmp_engine
    _seed_open_cycle(eng, n=2)
    with pytest.raises(PricingError) as ei:
        _close(eng, "c1", "close-1", 1, "18.60")
    assert ei.value.code == "revision_conflict"
    assert ei.value.details["current_revision"] == 2
    with Session(eng) as s:
        assert s.get(BillingCycle, "c1").status == "open"
        assert s.get(CycleClose, "c1") is None


def test_close_wrong_amount_conflict_and_state_untouched(tmp_engine):
    eng, _ = tmp_engine
    _seed_open_cycle(eng, n=2)
    with pytest.raises(PricingError) as ei:
        _close(eng, "c1", "close-1", 2, "999.00")
    assert ei.value.code == "close_amount_mismatch"
    assert ei.value.details["actual_total_amount"] == "18.60"
    with Session(eng) as s:
        assert s.get(BillingCycle, "c1").status == "open"
        assert s.get(CycleClose, "c1") is None
        # still accepts batches after a failed close
    _submit(eng, "c1", "k3", 2, [DAY3])


def test_close_requires_amount_match_to_cents(tmp_engine):
    eng, _ = tmp_engine
    _seed_open_cycle(eng, n=1)
    # 60 kWh @0.15 = exactly 9.00
    with pytest.raises(PricingError) as ei:
        _close(eng, "c1", "close-1", 1, "9.01")
    assert ei.value.code == "close_amount_mismatch"


# ------------------------------------------------------------- post-close rules


def test_new_batch_rejected_after_close_but_replay_still_works(tmp_engine):
    eng, _ = tmp_engine
    first = _submit(eng, "c1", "k1", 0, [DAY1]).body
    _close(eng, "c1", "close-1", 1, "9.00")

    # brand-new batch is rejected
    with pytest.raises(PricingError) as ei:
        _submit(eng, "c1", "k-new", 1, [DAY2])
    assert ei.value.code == "cycle_closed"
    with Session(eng) as s:
        assert len(s.scalars(select(UsageBatch)).all()) == 1
        assert len(s.scalars(select(CycleInterval)).all()) == 1

    # idempotent replay of a pre-close batch still returns its first response
    replay = _submit(eng, "c1", "k1", 1, [DAY1])
    assert replay.replayed is True
    assert replay.body == first


def test_get_close_404_when_not_closed(tmp_engine):
    eng, _ = tmp_engine
    _seed_open_cycle(eng, n=1)
    with pytest.raises(PricingError) as ei:
        settlement.get_close("c1", engine=eng)
    assert ei.value.status_code == 404
    assert ei.value.code == "close_not_found"


def test_get_cycle_carries_status_and_close_summary(tmp_engine):
    eng, _ = tmp_engine
    _seed_open_cycle(eng, n=2)
    state = settlement.get_cycle("c1", engine=eng)
    assert state["status"] == "open"
    assert state["closed_at"] is None
    assert state["close"] is None
    _close(eng, "c1", "close-1", 2, "18.60")
    state = settlement.get_cycle("c1", engine=eng)
    assert state["status"] == "closed"
    assert state["close"] == {
        "close_id": "close-1", "revision": 2,
        "expected_total_amount": "18.60",
        "detail_digest": state["close"]["detail_digest"],
        "created_at": state["close"]["created_at"]}


# ------------------------------------------------------------- concurrency


def test_concurrent_close_and_batch_only_one_wins(tmp_engine):
    eng, _ = tmp_engine
    _seed_open_cycle(eng, "race", n=1)  # revision 1
    results: dict[str, object] = {}

    def do_close():
        try:
            results["close"] = _close(eng, "race", "close-x", 1, "9.00").body
        except PricingError as e:
            results["close"] = e

    def do_batch():
        try:
            results["batch"] = _submit(eng, "race", "k2", 1, [DAY2]).body
        except PricingError as e:
            results["batch"] = e

    t1 = threading.Thread(target=do_close)
    t2 = threading.Thread(target=do_batch)
    t1.start(); t2.start(); t1.join(); t2.join()

    with Session(eng) as s:
        cy = s.get(BillingCycle, "race")
        if cy.status == "closed":
            assert isinstance(results["close"], dict), results
            assert isinstance(results["batch"], PricingError)
            assert results["batch"].code == "cycle_closed"
            assert cy.revision == 1
            assert len(s.scalars(select(UsageBatch)).all()) == 1
            assert s.get(CycleClose, "race") is not None
        else:
            assert cy.revision == 2
            assert isinstance(results["batch"], dict), results
            assert isinstance(results["close"], PricingError)
            assert results["close"].code == "revision_conflict"
            assert s.get(CycleClose, "race") is None
            assert len(s.scalars(select(UsageBatch)).all()) == 2


def test_concurrent_two_closes_only_one_succeeds(tmp_engine):
    eng, _ = tmp_engine
    _seed_open_cycle(eng, "race", n=2)
    results: dict[str, object] = {}

    def worker(name, cid):
        try:
            results[name] = _close(eng, "race", cid, 2, "18.60")
        except PricingError as e:
            results[name] = e

    t1 = threading.Thread(target=worker, args=("A", "close-a"))
    t2 = threading.Thread(target=worker, args=("B", "close-b"))
    t1.start(); t2.start(); t1.join(); t2.join()
    ok = [n for n, r in results.items() if not isinstance(r, PricingError)]
    err = [r for r in results.values() if isinstance(r, PricingError)]
    assert len(ok) == 1, results
    assert len(err) == 1
    assert err[0].code == "cycle_closed"


def test_concurrent_same_close_id_same_content_both_replay(tmp_engine):
    eng, _ = tmp_engine
    _seed_open_cycle(eng, "race", n=1)
    out: list = []

    def worker():
        try:
            out.append(_close(eng, "race", "same", 1, "9.00"))
        except PricingError as e:
            out.append(e)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(not isinstance(r, PricingError) for r in out), out
    assert out[0].body == out[1].body
    assert {r.replayed for r in out} <= {True, False}
    with Session(eng) as s:
        assert len(s.scalars(select(CycleClose)).all()) == 1


def test_8way_mixed_close_and_batch_exactly_one_wins(tmp_engine):
    eng, _ = tmp_engine
    _seed_open_cycle(eng, "race", n=1)  # revision 1, total 9.00
    results: dict[int, object] = {}

    def worker(i):
        try:
            if i % 2 == 0:
                results[i] = ("close",
                              _close(eng, "race", f"close-{i}", 1, "9.00"))
            else:
                results[i] = ("batch",
                              _submit(eng, "race", f"k-{i}", 1, [DAY2]))
        except PricingError as e:
            results[i] = ("error", e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    winners = [(i, r) for i, (kind, r) in results.items()
               if not isinstance(r, PricingError)]
    assert len(winners) == 1, results
    win_kind = results[winners[0][0]][0]
    with Session(eng) as s:
        cy = s.get(BillingCycle, "race")
        closes = len(s.scalars(select(CycleClose)).all())
        batches = len(s.scalars(select(UsageBatch)).all())
        if win_kind == "close":
            assert cy.status == "closed" and cy.revision == 1
            assert closes == 1 and batches == 1
        else:
            assert cy.status == "open" and cy.revision == 2
            assert closes == 0 and batches == 2


# ------------------------------------------------------------- persistence


def test_close_snapshot_survives_restart(tmp_engine):
    eng, path = tmp_engine
    _seed_open_cycle(eng, n=2)
    closed = _close(eng, "c1", "close-1", 2, "18.60").body
    eng.dispose()

    eng2 = create_engine(
        f"sqlite:///{path}",
        connect_args={"check_same_thread": False, "timeout": 30})
    configure_sqlite_engine(eng2)
    try:
        got = settlement.get_close("c1", engine=eng2)
        assert got == closed
        replay = _close(eng2, "c1", "close-1", 2, "18.60")
        assert replay.replayed is True and replay.body == closed
        # still closed: new batch rejected after restart
        with pytest.raises(PricingError) as ei:
            _submit(eng2, "c1", "k-new", 2, [DAY3])
        assert ei.value.code == "cycle_closed"
    finally:
        eng2.dispose()


def test_migration_adds_status_to_legacy_billing_cycles(tmp_path):
    path = tmp_path / "legacy.db"
    eng = create_engine(
        f"sqlite:///{path}",
        connect_args={"check_same_thread": False, "timeout": 30})
    configure_sqlite_engine(eng)
    # Build a legacy schema lacking status/closed_at (and no cycle_closes).
    with eng.begin() as conn:
        conn.execute(text(
            "CREATE TABLE billing_cycles (cycle_id VARCHAR(64) PRIMARY KEY, "
            "version_code VARCHAR NOT NULL, revision INTEGER, "
            "total_energy VARCHAR(64), total_amount TEXT, "
            "created_at VARCHAR(40), updated_at VARCHAR(40))"))
        conn.execute(text(
            "INSERT INTO billing_cycles VALUES ('old','v',1,'0','0','t','t')"))
    from app.database import _migrate_billing_cycles_schema
    _migrate_billing_cycles_schema(eng)
    # The legacy DB lacks every other table, so inspect via raw SQL only.
    with eng.begin() as conn:
        cols = {r[1] for r in conn.execute(
            text("PRAGMA table_info(billing_cycles)"))}
        assert {"status", "closed_at"} <= cols
        row = conn.execute(text(
            "SELECT status, closed_at FROM billing_cycles "
            "WHERE cycle_id='old'")).one()
        assert row[0] == "open" and row[1] is None
    eng.dispose()


# ------------------------------------------------------------- HTTP surface


def test_http_close_lifecycle():
    client = TestClient(app)
    with client:
        cid = "http-close-cycle"
        for key, rev, iv in (
            ("hk1", 0, {"start": "2024-07-06T00:00:00-04:00",
                        "end": "2024-07-06T01:00:00-04:00",
                        "timezone": "America/New_York", "energy": "60"}),
        ):
            r = client.post(f"/cycles/{cid}/batches", json={
                "version": V, "idempotency_key": key,
                "expected_revision": rev, "intervals": [iv]})
            assert r.status_code == 201, r.text

        # wrong amount -> 409 envelope, cycle stays open
        r = client.post(f"/cycles/{cid}/close", json={
            "close_id": "hclose", "expected_revision": 1,
            "expected_total_amount": "1.00"})
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "close_amount_mismatch"

        # success -> 201
        r = client.post(f"/cycles/{cid}/close", json={
            "close_id": "hclose", "expected_revision": 1,
            "expected_total_amount": "9.00"})
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["status"] == "closed"
        assert len(body["detail_digest"]) == 64

        # replay -> 200 identical
        r2 = client.post(f"/cycles/{cid}/close", json={
            "close_id": "hclose", "expected_revision": 1,
            "expected_total_amount": "9.00"})
        assert r2.status_code == 200 and r2.json() == body

        # new batch after close -> 409 cycle_closed
        r3 = client.post(f"/cycles/{cid}/batches", json={
            "version": V, "idempotency_key": "hk-new",
            "expected_revision": 1, "intervals": [{
                "start": "2024-07-06T01:00:00-04:00",
                "end": "2024-07-06T02:00:00-04:00",
                "timezone": "America/New_York", "energy": "1"}]})
        assert r3.status_code == 409
        assert r3.json()["error"]["code"] == "cycle_closed"

        # GET snapshot
        g = client.get(f"/cycles/{cid}/close")
        assert g.status_code == 200 and g.json() == body

        # GET status view
        cy = client.get(f"/cycles/{cid}").json()
        assert cy["status"] == "closed" and cy["close"]["close_id"] == "hclose"

        # 404s
        assert client.get("/cycles/no-such-cycle/close").status_code == 404


def test_http_get_close_404_when_open():
    client = TestClient(app)
    with client:
        r = client.post("/cycles/open-only/batches", json={
            "version": V, "idempotency_key": "k", "expected_revision": 0,
            "intervals": [{"start": "2024-07-06T00:00:00-04:00",
                           "end": "2024-07-06T01:00:00-04:00",
                           "timezone": "America/New_York", "energy": "60"}]})
        assert r.status_code == 201
        g = client.get("/cycles/open-only/close")
        assert g.status_code == 404
        assert g.json()["error"]["code"] == "close_not_found"
