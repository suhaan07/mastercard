"""L3: rolling-window behavioural features.

The card-testing signature the design doc names is *many distinct PANs, small
amounts, high declines, short window, narrow merchant set*. Every one of those
is a windowed aggregate, so this module maintains rolling windows keyed by
account, card, merchant, device and IP, and reads features off them.

Two properties matter more than the feature list:

* **O(1) amortised update.** Windows are deques trimmed on append, so scoring an
  auth event is a handful of dictionary lookups, not a scan. This is what holds
  the p99 latency budget -- there is no model in the world fast enough to rescue
  a feature layer that does a table scan per authorisation.
* **No future information.** Every feature is computed from events strictly
  before the one being scored. The same code path serves training and serving,
  so a feature cannot behave differently in backtest than in production.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from contracts.schemas import AuthEvent

#: Window widths, in seconds. One minute catches bursts; seven days establishes
#: the account's own baseline so "unusual" can mean unusual *for them*.
WINDOWS: dict[str, float] = {
    "1m": 60.0,
    "1h": 3600.0,
    "24h": 86400.0,
    "7d": 604800.0,
}
_MAX_WINDOW = max(WINDOWS.values())


@dataclass(slots=True)
class _Entry:
    ts: float
    declined: bool
    amount: float
    zero_auth: bool
    cvv_bad: bool
    avs_bad: bool
    card_token: str
    merchant_id: str
    pan_suffix6: str


class RollingWindow:
    """Events for one key, trimmed to the widest window on append."""

    __slots__ = ("entries",)

    def __init__(self) -> None:
        self.entries: deque[_Entry] = deque()

    def add(self, e: _Entry) -> None:
        self.entries.append(e)
        cutoff = e.ts - _MAX_WINDOW
        while self.entries and self.entries[0].ts < cutoff:
            self.entries.popleft()

    def within(self, now: float, span: float) -> list[_Entry]:
        cutoff = now - span
        # Entries are append-ordered, so walk back from the newest and stop.
        out: list[_Entry] = []
        for e in reversed(self.entries):
            if e.ts < cutoff:
                break
            out.append(e)
        return out


def pan_entropy(suffixes: list[str]) -> float:
    """Normalised entropy of the numeric gaps between attempted PANs.

    Enumeration walks the number space in near-order, so consecutive suffixes
    differ by a small, highly repetitive amount and the gap distribution
    collapses. Random legitimate traffic produces gaps spread across the range.
    Returns 1.0 for maximal disorder and approaches 0.0 for a clean sequence.
    """
    if len(suffixes) < 3:
        return 1.0
    try:
        nums = sorted(int(s) for s in suffixes)
    except ValueError:
        return 1.0
    gaps = [b - a for a, b in zip(nums, nums[1:]) if b > a]
    if not gaps:
        return 0.0
    counts: dict[int, int] = defaultdict(int)
    for g in gaps:
        # Bucket by order of magnitude: a sequence of gaps of 1, 2, 3 is the
        # same phenomenon as a sequence of 1, 1, 1, and both differ sharply
        # from gaps spread over six orders of magnitude.
        counts[int(math.log10(g)) if g > 0 else 0] += 1
    total = sum(counts.values())
    ent = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return float(min(1.0, ent / math.log2(6)))


def timing_regularity(times: list[float]) -> float:
    """Coefficient of variation of inter-arrival gaps. Low means machine-driven.

    Humans produce ragged gaps. A script producing one attempt every 40 seconds
    produces a CV near zero, and that is one of the few signals a sophisticated
    operator has to actively spend effort to defeat.
    """
    if len(times) < 3:
        return 1.0
    ts = sorted(times)
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    if not gaps:
        return 1.0
    mean = sum(gaps) / len(gaps)
    if mean <= 1e-9:
        return 0.0
    var = sum((g - mean) ** 2 for g in gaps) / len(gaps)
    return float(min(2.0, math.sqrt(var) / mean))


class StreamFeatureStore:
    """Maintains rolling windows and produces feature vectors for auth events."""

    def __init__(self) -> None:
        self._by_account: dict[str, RollingWindow] = defaultdict(RollingWindow)
        self._by_device: dict[str, RollingWindow] = defaultdict(RollingWindow)
        self._by_ip: dict[str, RollingWindow] = defaultdict(RollingWindow)
        self._by_merchant: dict[str, RollingWindow] = defaultdict(RollingWindow)
        self._by_card: dict[str, RollingWindow] = defaultdict(RollingWindow)
        #: First time each account was seen, for account-age features.
        self._first_seen: dict[str, float] = {}
        #: Merchants an account has previously transacted with.
        self._account_merchants: dict[str, set[str]] = defaultdict(set)

    # -- update ------------------------------------------------------------

    def _entry(self, ev: AuthEvent) -> _Entry:
        return _Entry(
            ts=ev.ts.timestamp(),
            declined=not ev.approved,
            amount=ev.amount,
            zero_auth=ev.is_zero_auth,
            cvv_bad=ev.cvv_result.value != "match",
            avs_bad=ev.avs_result.value not in ("match", "partial"),
            card_token=ev.card_token,
            merchant_id=ev.merchant_id,
            pan_suffix6=ev.pan_suffix6,
        )

    def update(self, ev: AuthEvent) -> None:
        """Fold an event into the windows. Call *after* scoring it."""
        e = self._entry(ev)
        self._by_account[ev.account_id].add(e)
        self._by_device[ev.device_id].add(e)
        self._by_ip[ev.ip_id].add(e)
        self._by_merchant[ev.merchant_id].add(e)
        self._by_card[ev.card_token].add(e)
        self._first_seen.setdefault(ev.account_id, e.ts)
        self._account_merchants[ev.account_id].add(ev.merchant_id)

    # -- read --------------------------------------------------------------

    def features(self, ev: AuthEvent) -> dict[str, float]:
        """Feature vector for ``ev``, using only events that preceded it."""
        now = ev.ts.timestamp()
        f: dict[str, float] = {}

        acct = self._by_account.get(ev.account_id)
        dev = self._by_device.get(ev.device_id)
        ip = self._by_ip.get(ev.ip_id)

        for label, span in WINDOWS.items():
            entries = acct.within(now, span) if acct else []
            n = len(entries)
            f[f"acct_n_{label}"] = float(n)
            if n:
                f[f"acct_decline_ratio_{label}"] = sum(e.declined for e in entries) / n
                f[f"acct_zero_auth_ratio_{label}"] = sum(e.zero_auth for e in entries) / n
                f[f"acct_cvv_bad_ratio_{label}"] = sum(e.cvv_bad for e in entries) / n
                f[f"acct_avs_bad_ratio_{label}"] = sum(e.avs_bad for e in entries) / n
                f[f"acct_low_ticket_ratio_{label}"] = sum(e.amount < 5.0 for e in entries) / n
                f[f"acct_distinct_pans_{label}"] = float(len({e.card_token for e in entries}))
                f[f"acct_distinct_merchants_{label}"] = float(
                    len({e.merchant_id for e in entries})
                )
                f[f"acct_mean_amount_{label}"] = sum(e.amount for e in entries) / n
            else:
                for suffix in (
                    "decline_ratio",
                    "zero_auth_ratio",
                    "cvv_bad_ratio",
                    "avs_bad_ratio",
                    "low_ticket_ratio",
                    "distinct_pans",
                    "distinct_merchants",
                    "mean_amount",
                ):
                    f[f"acct_{suffix}_{label}"] = 0.0

        # The card-testing core: enumeration and machine timing, over the hour.
        hour_entries = acct.within(now, WINDOWS["1h"]) if acct else []
        f["acct_pan_entropy_1h"] = pan_entropy([e.pan_suffix6 for e in hour_entries])
        f["acct_timing_cv_1h"] = timing_regularity([e.ts for e in hour_entries])
        f["acct_pans_per_merchant_1h"] = float(
            len({e.card_token for e in hour_entries})
            / max(1, len({e.merchant_id for e in hour_entries}))
        )

        # Shared-infrastructure velocity. A device or subnet driving many
        # accounts is a ring signal that no single account's history reveals.
        for name, win in (("device", dev), ("ip", ip)):
            entries = win.within(now, WINDOWS["1h"]) if win else []
            f[f"{name}_n_1h"] = float(len(entries))
            f[f"{name}_distinct_pans_1h"] = float(len({e.card_token for e in entries}))
            f[f"{name}_decline_ratio_1h"] = (
                sum(e.declined for e in entries) / len(entries) if entries else 0.0
            )

        # Account age and merchant novelty.
        first = self._first_seen.get(ev.account_id)
        f["acct_age_days"] = (now - first) / 86400.0 if first is not None else 0.0
        f["acct_is_new_merchant"] = float(
            ev.merchant_id not in self._account_merchants.get(ev.account_id, ())
        )
        f["acct_known_merchants"] = float(len(self._account_merchants.get(ev.account_id, ())))

        # This event's own attributes.
        f["amount"] = ev.amount
        f["is_zero_auth"] = float(ev.is_zero_auth)
        f["cvv_bad"] = float(ev.cvv_result.value != "match")
        f["avs_bad"] = float(ev.avs_result.value not in ("match", "partial"))
        f["hour_of_day"] = float(ev.ts.hour)
        f["day_of_week"] = float(ev.ts.weekday())

        return f


FEATURE_NAMES: tuple[str, ...] = tuple(
    sorted(
        {
            *(
                f"acct_{stat}_{w}"
                for w in WINDOWS
                for stat in (
                    "n",
                    "decline_ratio",
                    "zero_auth_ratio",
                    "cvv_bad_ratio",
                    "avs_bad_ratio",
                    "low_ticket_ratio",
                    "distinct_pans",
                    "distinct_merchants",
                    "mean_amount",
                )
            ),
            "acct_pan_entropy_1h",
            "acct_timing_cv_1h",
            "acct_pans_per_merchant_1h",
            *(f"{n}_{s}_1h" for n in ("device", "ip") for s in ("n", "distinct_pans", "decline_ratio")),
            "acct_age_days",
            "acct_is_new_merchant",
            "acct_known_merchants",
            "amount",
            "is_zero_auth",
            "cvv_bad",
            "avs_bad",
            "hour_of_day",
            "day_of_week",
        }
    )
)
