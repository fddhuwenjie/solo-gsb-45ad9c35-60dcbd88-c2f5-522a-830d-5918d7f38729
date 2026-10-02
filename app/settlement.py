"""账期分批用量的持久化增量结算。

账期用量会分批到达，本模块在 SQLite 上提供"提交即对账期整体重算"的
增量结算语义：

1. **幂等优先**：每次提交带 ``idempotency_key``；先查幂等记录——同键同
   内容直接返回首次响应（不写任何数据、revision 不变），同键不同内容
   拒绝（409 ``idempotency_conflict``）。
2. **乐观 revision**：幂等检查之后再比 ``expected_revision``——首次必须
   为 0，之后必须等于账期当前 revision，否则 409 ``revision_conflict``。
   两个并发请求持同一 revision 时，通过条件写（``WHERE revision = :exp``）
   与 SQLite 写锁保证只有一个生效；落败方在重试时看到新 revision，得到
   409。
3. **UTC 合并、禁止重叠**：历史区间与本批区间全部换算成 UTC 后合并且
   不得重叠（首尾相接允许），随后调用现有计价引擎对账期整体重算——
   累计阶梯因此自然贯穿所有批次。
4. **本批金额**：重算后的未舍入累计金额 − 重算前的未舍入累计金额，再
   四舍五入到分；持久化未舍入累计金额保证重启后本批金额与首次响应一致。

任何区间冲突、规则版本锁定冲突、规则缺口或计价失败都在事务提交前抛出，
不会写入批次、幂等记录，也不会推进 revision。

原子关账（:func:`close_cycle` / ``POST /cycles/{id}/close``）：

1. 在**单个 SQLite 写事务**内重读账期 revision、全部批次、全部区间与
   累计金额，并用计价引擎整体重算；revision 匹配且期望总额与重算累计
   （到分）一致，才以条件写（``WHERE revision=:expected AND
   status='open'``）把账期翻转为 ``closed``，同事务写入不可变关账快照。
2. 快照明细按**批次 revision（id 兜底）、区间 UTC 起点/终点（seq 兜底）
   稳定排序**后组成规范 JSON 取 SHA-256（``detail_digest``）；快照行
   保存首次完整响应体，同 ``close_id`` 同内容重放逐字节返回，换内容
   409 ``close_idempotency_conflict``，换 ``close_id`` 409
   ``cycle_closed``。
3. 关账后账期拒绝新批次（409 ``cycle_closed``，旧批次同键重放仍可读）；
   并发的批次与关账由条件写 + SQLite 写锁互斥，只有一个生效，落败方
   不写入任何数据。任何检查失败都在提交前回滚，账期保持开放。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from . import pricing
from .database import (
    BillingCycle,
    CycleClose,
    CycleInterval,
    UsageBatch,
    engine as default_engine,
)
from .importer import canonical_json
from .pricing import (
    Q2,
    Q6,
    ZERO,
    InputInterval,
    LoadedVersion,
    PricingError,
    load_version,
)
from .schemas import QuoteResponse, SegmentOut, SettlementBatchRequest

# Same-revision write races surface as SQLITE_BUSY (a deferred transaction
# upgrading to a write lock); retry the whole serialized transaction a
# bounded number of times before giving up.
_MAX_LOCK_RETRIES = 20


# --------------------------------------------------------------------- helpers


@dataclass(frozen=True)
class SubmitResult:
    body: dict
    replayed: bool


class _StoredIntervalPayload:
    """Minimal interval payload accepted by ``pricing.quote``.

    Times are already UTC instants rendered with a ``Z`` suffix; the engine
    re-parses them and uses ``timezone`` only for local-time rendering.
    """

    def __init__(self, start_utc: datetime, end_utc: datetime,
                 tz_name: str, energy: Decimal):
        self.start = _utc_z(start_utc)
        self.end = _utc_z(end_utc)
        self.timezone = tz_name
        self._energy = energy

    def energy_decimal(self) -> Decimal:
        return self._energy


def _utc_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_z(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------- digest


def _content_digest(version_code: str,
                    intervals: list[InputInterval]) -> str:
    """幂等内容摘要：规则版本 + 本批区间（UTC 端点/时区/能量，按 UTC 起点
    排序）。``expected_revision`` 是并发控制令牌，刻意不参与摘要。"""
    doc = {
        "version": version_code,
        "intervals": [
            {
                "start": _utc_z(i.start),
                "end": _utc_z(i.end),
                "timezone": str(i.tz),
                "energy": format(i.energy.normalize(), "f"),
            }
            for i in sorted(intervals,
                            key=lambda i: (i.start, i.end, str(i.tz)))
        ],
    }
    return hashlib.sha256(canonical_json(doc).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- quote body


def build_quote_response(version: LoadedVersion,
                         result: pricing.QuoteResult | None) -> QuoteResponse:
    """构造与 ``POST /quote`` 完全一致的报价响应体。

    集中在此处，供 ``/quote``、``/usage-estimate`` 与增量结算共用。
    """
    if result is None:
        return QuoteResponse(
            version=version.code,
            currency=version.currency,
            energy_unit=version.energy_unit,
            total_energy=str(ZERO.quantize(Q6)),
            total_amount=str(ZERO.quantize(Q2)),
            matched_rules=[],
            segments=[],
        )
    return QuoteResponse(
        version=version.code,
        currency=version.currency,
        energy_unit=version.energy_unit,
        total_energy=str(result.total_energy),
        total_amount=str(result.total_amount.quantize(
            Q2, rounding=ROUND_HALF_UP)),
        matched_rules=result.matched_rules,
        segments=[
            SegmentOut(**s.to_dict(version.energy_unit, version.currency))
            for s in result.segments
        ],
    )


# ------------------------------------------------------------------- errors


class _BatchGuardFailed(Exception):
    """批次提交的条件更新在写事务内落败（revision 被并发推进或账期被并发
    关账）；由外层用只读新事务重查归类，调用方收到常规 409。"""


def _cycle_closed(cycle: BillingCycle, *, close_id: str | None = None
                  ) -> PricingError:
    return PricingError(
        f"账期 {cycle.cycle_id} 已关账（revision={cycle.revision}），"
        "拒绝新批次与重复/冲突关账；关账快照不可变",
        code="cycle_closed", status_code=409,
        details={"cycle_id": cycle.cycle_id,
                 "revision": cycle.revision,
                 "close_id": close_id},
    )


def _close_not_found(cycle_id: str, close_id: str) -> PricingError:
    return PricingError(
        f"账期 {cycle_id} 不存在 close_id 为 {close_id!r} 的关账快照",
        code="close_not_found", status_code=404,
        details={"cycle_id": cycle_id, "close_id": close_id},
    )


def _close_amount_mismatch(cycle_id: str, expected: str,
                           actual: Decimal) -> PricingError:
    return PricingError(
        f"账期 {cycle_id} 的 expected_total_amount={expected} 与服务端"
        f"全部批次累计金额 {actual} 不一致，拒绝关账",
        code="close_amount_mismatch", status_code=409,
        details={"cycle_id": cycle_id,
                 "expected_total_amount": expected,
                 "actual_total_amount": str(actual.quantize(Q2))},
    )


def _close_content_conflict(cycle_id: str, close_id: str,
                            existing_digest: str,
                            received_digest: str) -> PricingError:
    return PricingError(
        f"账期 {cycle_id} 的 close_id {close_id!r} 已用于不同的关账内容"
        "（同 close_id 重放必须携带相同的 expected_total_amount）",
        code="close_idempotency_conflict", status_code=409,
        details={"cycle_id": cycle_id, "close_id": close_id,
                 "existing_digest": existing_digest,
                 "received_digest": received_digest},
    )


def _close_id_in_use(close_id: str, other_cycle_id: str) -> PricingError:
    return PricingError(
        f"close_id {close_id!r} 已被另一个账期 {other_cycle_id} 的关账快照"
        "占用；close_id 全局唯一",
        code="close_id_in_use", status_code=409,
        details={"close_id": close_id, "other_cycle_id": other_cycle_id},
    )


def _classify_batch_guard(eng, cycle_id: str, req: SettlementBatchRequest):
    """条件更新落败后的只读归类：已关账 -> cycle_closed；否则按当前
    revision 报 revision_conflict。绝不写入。"""
    with Session(eng) as session:
        cycle = session.get(BillingCycle, cycle_id)
        if cycle is not None and cycle.status == "closed":
            close = cycle.close
            raise _cycle_closed(
                cycle, close_id=close.close_id if close else None)
        current = cycle.revision if cycle is not None else 0
        raise _revision_conflict(cycle_id, req.expected_revision, current)


def _revision_conflict(cycle_id: str, expected: int,
                       current: int) -> PricingError:
    return PricingError(
        f"账期 {cycle_id} 的 expected_revision 过期：期望 {expected}，"
        f"当前 revision 为 {current}",
        code="revision_conflict", status_code=409,
        details={"cycle_id": cycle_id,
                 "expected_revision": expected,
                 "current_revision": current},
    )


def _version_locked(cycle_id: str, locked: str,
                    requested: str) -> PricingError:
    return PricingError(
        f"账期 {cycle_id} 已锁定规则版本 {locked}，不能改用 {requested}；"
        "同一账期的所有批次必须使用相同规则版本",
        code="version_locked", status_code=409,
        details={"cycle_id": cycle_id,
                 "locked_version": locked,
                 "requested_version": requested},
    )


def _idempotency_conflict(cycle_id: str, key: str, existing_digest: str,
                          received_digest: str) -> PricingError:
    return PricingError(
        f"账期 {cycle_id} 的幂等键 {key!r} 已用于不同的提交内容"
        "（同键重放必须与首次提交完全一致）",
        code="idempotency_conflict", status_code=409,
        details={"cycle_id": cycle_id, "idempotency_key": key,
                 "existing_digest": existing_digest,
                 "received_digest": received_digest},
    )


def _overlap_error(cycle_id: str, a, b) -> PricingError:
    """合并去重后发现任意两段重叠时的 400；``a``/``b`` 为 CycleInterval
    行或本批 :class:`InputInterval`。"""
    def part(it):
        if isinstance(it, CycleInterval):
            return {"start": it.start_utc, "end": it.end_utc,
                    "timezone": it.timezone, "revision": it.revision}
        return {"start": _utc_z(it.start), "end": _utc_z(it.end),
                "timezone": str(it.tz)}

    pa, pb = part(a), part(b)
    return PricingError(
        f"账期 {cycle_id} 的用量区间在 UTC 时间线上重叠（{pa['start']} ~ "
        f"{pa['end']} 与 {pb['start']} ~ {pb['end']}）；账期内所有批次的"
        "区间必须互不重叠（首尾相接允许）",
        code="interval_overlap", status_code=400,
        details={"cycle_id": cycle_id,
                 "interval_a": pa, "interval_b": pb},
    )


def _find_overlap(stored_rows: list[CycleInterval],
                  new_intervals: list[InputInterval]):
    """合并历史行与本批区间后返回第一对重叠项，无重叠返回 None。"""
    items: list = list(stored_rows) + list(new_intervals)

    def span(it):
        if isinstance(it, CycleInterval):
            return _parse_z(it.start_utc), _parse_z(it.end_utc)
        return it.start, it.end

    items.sort(key=lambda it: span(it))
    for k in range(1, len(items)):
        a, b = items[k - 1], items[k]
        if span(b)[0] < span(a)[1]:
            return a, b
    return None


# ----------------------------------------------------------------- main flow


def submit_batch(cycle_id: str, req: SettlementBatchRequest, *,
                 engine=None) -> SubmitResult:
    """提交一批账期用量，返回 ``(响应体, 是否幂等重放)``。

    任何失败都抛出 :class:`PricingError`，且不会写入批次、幂等记录或推进
    revision。``engine`` 可注入（测试用临时文件库），默认应用全局引擎。
    """
    eng = engine or default_engine

    # 先用现有引擎语义解析本批区间（时区/DST/区间非法/批内重叠在此暴露），
    # 只有合法区间才进入事务、才计算幂等摘要。
    new_intervals = pricing.parse_usage_intervals(req.intervals)
    digest = _content_digest(req.version, new_intervals)

    attempt = 0
    while True:
        attempt += 1
        try:
            return _submit_once(eng, cycle_id, req, new_intervals, digest)
        except _BatchGuardFailed:
            # 条件更新落败：新事务只读重查，归类为关账拒绝 / revision 冲突，
            # 失败路径本身不写任何数据。
            return _classify_batch_guard(eng, cycle_id, req)
        except OperationalError as exc:  # SQLITE_BUSY same-revision race
            if attempt > _MAX_LOCK_RETRIES or not _is_lock_error(exc):
                raise
            # Serialized retry: re-read revision / idempotency record and
            # re-run all checks from scratch on a fresh transaction.


def _is_lock_error(exc: OperationalError) -> bool:
    msg = str(exc.orig).lower() if exc.orig is not None else ""
    return ("database is locked" in msg
            or "database table is locked" in msg)


def _submit_once(eng, cycle_id: str, req: SettlementBatchRequest,
                 new_intervals: list[InputInterval],
                 digest: str) -> SubmitResult:
    with Session(eng) as session:
        try:
            with session.begin():
                # 1) 幂等优先：同键记录先于 expected_revision 判定。
                existing_batch = session.scalar(
                    select(UsageBatch).where(
                        UsageBatch.cycle_id == cycle_id,
                        UsageBatch.idempotency_key == req.idempotency_key,
                    )
                )
                if existing_batch is not None:
                    if existing_batch.content_digest != digest:
                        raise _idempotency_conflict(
                            cycle_id, req.idempotency_key,
                            existing_batch.content_digest, digest)
                    # 同键同内容：无论 expected_revision 为何，原样返回
                    # 首次响应，不写任何数据。
                    return SubmitResult(json.loads(existing_batch.response_json),
                                        True)

                cycle = session.get(BillingCycle, cycle_id)

                # 2) expected_revision 乐观锁检查（在幂等检查之后）。
                current_revision = cycle.revision if cycle else 0
                if req.expected_revision != current_revision:
                    raise _revision_conflict(
                        cycle_id, req.expected_revision, current_revision)

                # 已关账账期拒绝一切新批次（幂等键重放已在上面返回，走到这
                # 里的一定是新键）；revision/金额快照均不可变。
                if cycle is not None and cycle.status == "closed":
                    raise _cycle_closed(cycle)

                # 3) 规则版本：不存在 -> 404；首次提交锁定版本；后续批次
                #    换版本 -> 409 version_locked。结构问题/规则缺口由下面
                #    的 quote 抛 409（rule_conflict / rule_gap），全部发生
                #    在写批次、区间与推进 revision 之前。
                version = load_version(session, req.version)
                if cycle is not None and cycle.version_code != req.version:
                    raise _version_locked(
                        cycle_id, cycle.version_code, req.version)

                # 4) 新旧区间按 UTC 合并且禁止重叠（首尾相接允许）。
                stored_rows = (
                    session.scalars(
                        select(CycleInterval)
                        .where(CycleInterval.cycle_id == cycle_id)
                        .order_by(CycleInterval.start_utc, CycleInterval.seq)
                    ).all()
                    if cycle is not None else []
                )
                hit = _find_overlap(list(stored_rows), new_intervals)
                if hit is not None:
                    raise _overlap_error(cycle_id, hit[0], hit[1])

                merged_payload = [
                    _StoredIntervalPayload(
                        _parse_z(r.start_utc), _parse_z(r.end_utc),
                        r.timezone, Decimal(r.energy))
                    for r in stored_rows
                ]
                merged_payload.extend(
                    _StoredIntervalPayload(i.start, i.end, str(i.tz), i.energy)
                    for i in new_intervals
                )

                # 5) 用现有计价引擎对账期整体重算：累计阶梯贯穿所有批次。
                result = pricing.quote(version, merged_payload)

                # 6) 本批金额 = 重算后未舍入累计 − 重算前未舍入累计，
                #    差值再四舍五入到分（账期 total_amount 同样到分）。
                prev_total_amount = (
                    Decimal(cycle.total_amount) if cycle is not None else ZERO)
                new_total_amount = result.total_amount
                batch_amount = (new_total_amount - prev_total_amount).quantize(
                    Q2, rounding=ROUND_HALF_UP)

                new_revision = current_revision + 1
                batch_energy = sum((i.energy for i in new_intervals), ZERO)
                now = _now()

                body = {
                    "cycle_id": cycle_id,
                    "version": version.code,
                    "currency": version.currency,
                    "energy_unit": version.energy_unit,
                    "revision": new_revision,
                    "batch_amount": str(batch_amount),
                    "total_energy": str(result.total_energy),
                    "total_amount": str(new_total_amount.quantize(
                        Q2, rounding=ROUND_HALF_UP)),
                    "quote": build_quote_response(version, result).model_dump(),
                }

                if cycle is None:
                    # 首次提交：INSERT 主键冲突意味着并发请求已抢先创建账期。
                    session.add(BillingCycle(
                        cycle_id=cycle_id,
                        version_code=version.code,
                        revision=new_revision,
                        total_energy=str(result.total_energy),
                        total_amount=str(new_total_amount),
                        created_at=now, updated_at=now,
                    ))
                    session.flush()
                else:
                    # 条件推进 revision：仅当 revision 仍等于 expected 且
                    # 账期仍开放时生效；rowcount==0 即被并发同 revision 请求
                    # 抢先，或并发关账已把 status 翻成 closed。
                    guarded = session.execute(
                        update(BillingCycle)
                        .where(BillingCycle.cycle_id == cycle_id,
                               BillingCycle.revision == req.expected_revision,
                               BillingCycle.status == "open")
                        .values(revision=new_revision,
                                total_energy=str(result.total_energy),
                                total_amount=str(new_total_amount),
                                updated_at=now)
                    )
                    if guarded.rowcount != 1:
                        # 无法在此事务内可靠区分两类落败，交给只读重查归类，
                        # 本事务随异常整体回滚。
                        raise _BatchGuardFailed()

                batch_row = UsageBatch(
                    cycle_id=cycle_id,
                    revision=new_revision,
                    idempotency_key=req.idempotency_key,
                    version_code=version.code,
                    content_digest=digest,
                    batch_energy=str(batch_energy),
                    batch_amount=str(batch_amount),
                    response_json=json.dumps(body, ensure_ascii=False),
                    created_at=now,
                )
                session.add(batch_row)
                session.flush()  # 取 batch_row.id，同时暴露幂等键唯一冲突

                next_seq = (
                    max((r.seq for r in stored_rows), default=-1) + 1
                )
                for i in sorted(new_intervals,
                                key=lambda i: (i.start, i.end)):
                    session.add(CycleInterval(
                        cycle_id=cycle_id,
                        revision=new_revision,
                        batch_id=batch_row.id,
                        start_utc=_utc_z(i.start),
                        end_utc=_utc_z(i.end),
                        timezone=str(i.tz),
                        energy=str(i.energy),
                        seq=next_seq,
                    ))
                    next_seq += 1

                return SubmitResult(body, False)

        except IntegrityError as exc:
            # 落库唯一/主键冲突（并发竞态）：只读重查归类为重放 /
            # 幂等冲突 / revision 冲突，绝不写任何数据。
            return _classify_integrity_error(
                eng, cycle_id, req, digest, exc)


def _classify_integrity_error(eng, cycle_id, req, digest, exc):
    with Session(eng) as session:
        batch = session.scalar(
            select(UsageBatch).where(
                UsageBatch.cycle_id == cycle_id,
                UsageBatch.idempotency_key == req.idempotency_key,
            )
        )
        if batch is not None:
            if batch.content_digest == digest:
                return SubmitResult(json.loads(batch.response_json), True)
            raise _idempotency_conflict(
                cycle_id, req.idempotency_key, batch.content_digest, digest)
        cycle = session.get(BillingCycle, cycle_id)
        if cycle is not None:
            if cycle.status == "closed":
                close = cycle.close
                raise _cycle_closed(
                    cycle, close_id=close.close_id if close else None) from exc
            raise _revision_conflict(
                cycle_id, req.expected_revision, cycle.revision) from exc
    raise exc


# --------------------------------------------------------------- read model


def get_cycle(cycle_id: str, *, engine=None) -> dict:
    """返回账期当前状态（含批次摘要与全部区间）；不存在抛 404。"""
    eng = engine or default_engine
    with Session(eng) as session:
        cycle = session.get(BillingCycle, cycle_id)
        if cycle is None:
            raise PricingError(
                f"账期不存在: {cycle_id}",
                code="cycle_not_found", status_code=404,
                details={"cycle_id": cycle_id})
        version = load_version(session, cycle.version_code)
        batches = session.scalars(
            select(UsageBatch)
            .where(UsageBatch.cycle_id == cycle_id)
            .order_by(UsageBatch.revision)
        ).all()
        intervals = session.scalars(
            select(CycleInterval)
            .where(CycleInterval.cycle_id == cycle_id)
            .order_by(CycleInterval.seq)
        ).all()
        return {
            "cycle_id": cycle.cycle_id,
            "version": cycle.version_code,
            "currency": version.currency,
            "energy_unit": version.energy_unit,
            "revision": cycle.revision,
            "status": cycle.status,
            "closed_at": cycle.closed_at,
            "close": (
                _close_summary(cycle.close) if cycle.close is not None else None
            ),
            "total_energy": cycle.total_energy,
            "total_amount": str(Decimal(cycle.total_amount).quantize(Q2)),
            "created_at": cycle.created_at,
            "updated_at": cycle.updated_at,
            "batches": [
                {
                    "revision": b.revision,
                    "idempotency_key": b.idempotency_key,
                    "content_digest": b.content_digest,
                    "batch_energy": b.batch_energy,
                    "batch_amount": str(Decimal(b.batch_amount).quantize(Q2)),
                    "created_at": b.created_at,
                }
                for b in batches
            ],
            "intervals": [
                {
                    "seq": r.seq,
                    "revision": r.revision,
                    "start": r.start_utc,
                    "end": r.end_utc,
                    "timezone": r.timezone,
                    "energy": r.energy,
                }
                for r in intervals
            ],
        }


# ---------------------------------------------------------------------- close


@dataclass(frozen=True)
class CloseResult:
    body: dict
    replayed: bool


class _CloseGuardFailed(Exception):
    """关账的条件翻转在写事务内落败（账期已被并发关账或 revision 已被并发
    批次推进）；由外层用只读新事务重查归类。"""


def _canon_amount(value: Decimal) -> str:
    """金额/用量的规范写法：Decimal 去尾零后的普通十进制串（避免
    1E+1 之类的指数形式），供稳定哈希与快照对齐。"""
    return format(value.normalize(), "f")


def _close_request_digest(version_code: str, revision: int,
                          expected_total_amount: Decimal) -> str:
    """关账请求内容摘要：锁定版本 + 冻结 revision + 期望总额。

    ``close_id`` 是幂等键本身，刻意不参与内容摘要（与批次幂等摘要排除
    idempotency_key 同理）；revision 参与摘要，使同一金额但针对不同
    revision 的关账视为不同内容。
    """
    doc = {
        "version": version_code,
        "revision": revision,
        "expected_total_amount": _canon_amount(
            expected_total_amount.quantize(Q2)),
    }
    return hashlib.sha256(canonical_json(doc).encode("utf-8")).hexdigest()


def _sorted_batches(session: Session, cycle_id: str):
    # revision 单批唯一递增，主键 id 仅作并列时的稳定次序。
    return session.scalars(
        select(UsageBatch)
        .where(UsageBatch.cycle_id == cycle_id)
        .order_by(UsageBatch.revision, UsageBatch.id)
    ).all()


def _sorted_intervals(session: Session, cycle_id: str):
    # 区间在账期内首尾相接/互不重叠；按 UTC 起点、终点、seq 稳定排序。
    return session.scalars(
        select(CycleInterval)
        .where(CycleInterval.cycle_id == cycle_id)
        .order_by(CycleInterval.start_utc, CycleInterval.end_utc,
                  CycleInterval.seq)
    ).all()


def _detail_doc(version_code: str, revision: int, total_energy: Decimal,
                total_amount: Decimal, batches, intervals) -> dict:
    """关账明细规范化文档：批次按 revision、区间按 UTC 端点稳定排序后，
    对其完整业务内容取 SHA-256。任何批次或区间细节漂移都会改变摘要。"""
    return {
        "version": version_code,
        "revision": revision,
        "total_energy": _canon_amount(total_energy),
        "total_amount": _canon_amount(total_amount),
        "batches": [
            {
                "revision": b.revision,
                "idempotency_key": b.idempotency_key,
                "version": b.version_code,
                "content_digest": b.content_digest,
                "batch_energy": _canon_amount(Decimal(b.batch_energy)),
                "batch_amount": _canon_amount(Decimal(b.batch_amount)),
            }
            for b in batches
        ],
        "intervals": [
            {
                "seq": r.seq,
                "revision": r.revision,
                "start": r.start_utc,
                "end": r.end_utc,
                "timezone": r.timezone,
                "energy": _canon_amount(Decimal(r.energy)),
            }
            for r in intervals
        ],
    }


def _snapshot_body(cycle: BillingCycle, close_row: CycleClose,
                   batches, intervals) -> dict:
    """构造（并持久化）的不可变关账快照响应体；重放时逐字节返回。"""
    return {
        "cycle_id": cycle.cycle_id,
        "close_id": close_row.close_id,
        "version": cycle.version_code,
        "currency": None,  # 由 close_cycle/get_close 用版本信息填充
        "revision": close_row.revision,
        "status": "closed",
        "expected_total_amount": str(
            Decimal(close_row.expected_total_amount).quantize(Q2)),
        "total_energy": close_row.total_energy,
        "total_amount": str(Decimal(close_row.total_amount).quantize(Q2)),
        "content_digest": close_row.content_digest,
        "detail_digest": close_row.detail_digest,
        "created_at": close_row.created_at,
        "batches": [
            {
                "revision": b.revision,
                "idempotency_key": b.idempotency_key,
                "content_digest": b.content_digest,
                "batch_energy": b.batch_energy,
                "batch_amount": str(Decimal(b.batch_amount).quantize(Q2)),
                "created_at": b.created_at,
            }
            for b in batches
        ],
        "intervals": [
            {
                "seq": r.seq,
                "revision": r.revision,
                "start": r.start_utc,
                "end": r.end_utc,
                "timezone": r.timezone,
                "energy": r.energy,
            }
            for r in intervals
        ],
    }


def _with_version_fields(session: Session, body: dict) -> dict:
    version = load_version(session, body["version"])
    body["currency"] = version.currency
    body["energy_unit"] = version.energy_unit
    return body


def close_cycle(cycle_id: str, req, *, engine=None) -> CloseResult:
    """原子关账，返回 ``(快照响应体, 是否幂等重放)``。

    在单个 SQLite 写事务内重读账期 revision、全部批次与累计金额：revision
    匹配且期望金额与服务端重算累计一致，才把账期条件翻转为 ``closed`` 并
    写入不可变关账快照。任何检查失败都在提交前抛出，账期保持开放、批次与
    金额不变。``engine`` 可注入（测试用临时文件库）。
    """
    eng = engine or default_engine
    expected_amount = req.expected_total_amount.quantize(Q2)
    attempt = 0
    while True:
        attempt += 1
        try:
            return _close_once(eng, cycle_id, req, expected_amount)
        except _CloseGuardFailed:
            return _classify_close_guard(eng, cycle_id, req, expected_amount)
        except OperationalError as exc:  # SQLITE_BUSY 与批次/关账竞争
            if attempt > _MAX_LOCK_RETRIES or not _is_lock_error(exc):
                raise
            # 新事务重读全部状态后从头再判。


def _close_once(eng, cycle_id: str, req, expected_amount: Decimal
                ) -> CloseResult:
    with Session(eng) as session:
        try:
            with session.begin():
                cycle = session.get(BillingCycle, cycle_id)
                if cycle is None:
                    raise PricingError(
                        f"账期不存在: {cycle_id}",
                        code="cycle_not_found", status_code=404,
                        details={"cycle_id": cycle_id})

                # 1) 已有关账快照：同 close_id 同内容重放；其余一律拒绝。
                existing = session.get(CycleClose, cycle_id)
                if existing is not None:
                    digest = _close_request_digest(
                        cycle.version_code, cycle.revision, expected_amount)
                    if existing.close_id != req.close_id:
                        raise _cycle_closed(cycle, close_id=existing.close_id)
                    if existing.content_digest != digest:
                        raise _close_content_conflict(
                            cycle_id, existing.close_id,
                            existing.content_digest, digest)
                    body = _with_version_fields(
                        session, json.loads(existing.snapshot_json))
                    return CloseResult(body, True)

                # 2) expected_revision 必须等于当前 revision（关账不推进
                #    revision；revision 落后时，并发批次可能正在提交）。
                if req.expected_revision != cycle.revision:
                    raise _revision_conflict(
                        cycle_id, req.expected_revision, cycle.revision)

                # 3) 在同一事务内重读全部批次与区间并重算累计金额。
                version = load_version(session, cycle.version_code)
                batches = _sorted_batches(session, cycle_id)
                intervals = _sorted_intervals(session, cycle_id)
                payload = [
                    _StoredIntervalPayload(
                        _parse_z(r.start_utc), _parse_z(r.end_utc),
                        r.timezone, Decimal(r.energy))
                    for r in intervals
                ]
                result = pricing.quote(version, payload)
                actual_total = result.total_amount

                # 持久化的未舍入累计必须与重算一致（不一致即持久化状态
                # 损坏，绝不在客户侧金额冲突的名义下静默关账）。
                stored_total = Decimal(cycle.total_amount)
                if stored_total != actual_total:
                    raise PricingError(
                        f"账期 {cycle_id} 的持久化累计金额 "
                        f"{stored_total} 与重算金额 {actual_total} 不一致",
                        code="cycle_state_inconsistent", status_code=500,
                        details={"cycle_id": cycle_id,
                                 "stored_total_amount": _canon_amount(
                                     stored_total),
                                 "recomputed_total_amount": _canon_amount(
                                     actual_total)})
                if result.total_energy != Decimal(cycle.total_energy):
                    raise PricingError(
                        f"账期 {cycle_id} 的持久化累计用量与重算结果不一致",
                        code="cycle_state_inconsistent", status_code=500,
                        details={"cycle_id": cycle_id,
                                 "stored_total_energy": cycle.total_energy,
                                 "recomputed_total_energy": _canon_amount(
                                     result.total_energy)})

                # 4) 期望总额（到分）必须与重算累计一致才允许关账。
                if expected_amount != actual_total.quantize(
                        Q2, rounding=ROUND_HALF_UP):
                    raise _close_amount_mismatch(
                        cycle_id, str(expected_amount), actual_total)

                total_energy = result.total_energy
                content_digest = _close_request_digest(
                    cycle.version_code, cycle.revision, expected_amount)
                detail_doc = _detail_doc(
                    cycle.version_code, cycle.revision, total_energy,
                    actual_total, batches, intervals)
                detail_digest = hashlib.sha256(
                    canonical_json(detail_doc).encode("utf-8")).hexdigest()
                now = _now()

                # 5) 先构造快照行（不立即 flush），再做条件翻转：只有
                #    status 仍为 open 且 revision 仍等于 expected 时才生效。
                close_row = CycleClose(
                    cycle_id=cycle_id,
                    close_id=req.close_id,
                    revision=cycle.revision,
                    expected_total_amount=str(expected_amount),
                    content_digest=content_digest,
                    detail_digest=detail_digest,
                    total_energy=_canon_amount(total_energy),
                    total_amount=_canon_amount(actual_total),
                    snapshot_json="",  # 下方翻转成功后回填
                    created_at=now,
                )
                session.add(close_row)
                session.flush()  # 暴露 close_id 全局唯一冲突

                guarded = session.execute(
                    update(BillingCycle)
                    .where(BillingCycle.cycle_id == cycle_id,
                           BillingCycle.revision == req.expected_revision,
                           BillingCycle.status == "open")
                    .values(status="closed",
                            closed_at=now,
                            updated_at=now)
                )
                if guarded.rowcount != 1:
                    # 并发关账或并发批次已抢先：只读重查归类，事务回滚。
                    raise _CloseGuardFailed()

                # 重新按稳定次序读取（当前事务内数据未变，次序与文档一致），
                # 组装快照并回填 JSON；响应体与持久化快照逐字节同源。
                body = _snapshot_body(
                    cycle, close_row, batches, intervals)
                body["currency"] = version.currency
                body["energy_unit"] = version.energy_unit
                close_row.snapshot_json = json.dumps(body, ensure_ascii=False)
                return CloseResult(body, False)

        except IntegrityError as exc:
            # close_id 全局唯一冲突等：只读重查归类，绝不留下半成品。
            return _classify_close_integrity_error(
                eng, cycle_id, req, expected_amount, exc)


def _classify_close_guard(eng, cycle_id, req, expected_amount) -> CloseResult:
    with Session(eng) as session:
        existing = session.get(CycleClose, cycle_id)
        if existing is not None:
            cycle = session.get(BillingCycle, cycle_id)
            if existing.close_id == req.close_id:
                digest = _close_request_digest(
                    cycle.version_code, existing.revision, expected_amount)
                if existing.content_digest == digest:
                    return CloseResult(
                        _with_version_fields(
                            session, json.loads(existing.snapshot_json)),
                        True)
                raise _close_content_conflict(
                    cycle_id, existing.close_id,
                    existing.content_digest, digest)
            raise _cycle_closed(cycle, close_id=existing.close_id)
        cycle = session.get(BillingCycle, cycle_id)
        if cycle is None:
            raise PricingError(
                f"账期不存在: {cycle_id}",
                code="cycle_not_found", status_code=404,
                details={"cycle_id": cycle_id})
        raise _revision_conflict(
            cycle_id, req.expected_revision, cycle.revision)


def _classify_close_integrity_error(eng, cycle_id, req, expected_amount, exc):
    with Session(eng) as session:
        existing = session.get(CycleClose, cycle_id)
        if existing is not None:
            cycle = session.get(BillingCycle, cycle_id)
            digest = _close_request_digest(
                cycle.version_code, existing.revision, expected_amount)
            if existing.close_id == req.close_id:
                if existing.content_digest == digest:
                    return CloseResult(
                        _with_version_fields(
                            session, json.loads(existing.snapshot_json)),
                        True)
                raise _close_content_conflict(
                    cycle_id, existing.close_id,
                    existing.content_digest, digest)
            raise _cycle_closed(cycle, close_id=existing.close_id)
        # 本账期尚无快照：冲突来自 close_id 被别的账期占用。
        owner = session.scalar(
            select(CycleClose).where(CycleClose.close_id == req.close_id))
        if owner is not None:
            raise _close_id_in_use(req.close_id, owner.cycle_id) from exc
    raise exc


def _close_summary(close_row: CycleClose) -> dict:
    return {
        "close_id": close_row.close_id,
        "revision": close_row.revision,
        "expected_total_amount": str(
            Decimal(close_row.expected_total_amount).quantize(Q2)),
        "detail_digest": close_row.detail_digest,
        "created_at": close_row.created_at,
    }


def get_close(cycle_id: str, *, engine=None) -> dict:
    """返回账期的不可变关账快照；账期或快照不存在均 404。"""
    eng = engine or default_engine
    with Session(eng) as session:
        cycle = session.get(BillingCycle, cycle_id)
        if cycle is None:
            raise PricingError(
                f"账期不存在: {cycle_id}",
                code="cycle_not_found", status_code=404,
                details={"cycle_id": cycle_id})
        close_row = session.get(CycleClose, cycle_id)
        if close_row is None:
            raise PricingError(
                f"账期 {cycle_id} 尚未关账",
                code="close_not_found", status_code=404,
                details={"cycle_id": cycle_id})
        return _with_version_fields(
            session, json.loads(close_row.snapshot_json))
