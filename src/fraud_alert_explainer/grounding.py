"""Check that an LLM explanation cites only what SHAP ranked: no I/O, no LLM calls.

Same idea as a RAG faithfulness check, made deterministic: the model may talk about the
features it was given and nothing else, and every number it writes must come from the facts.
"""

import re
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

OUTSIDE_MODEL = "outside_model"

# Phrases that count as citing a feature. Checked on lower-cased text.
_ALIASES: dict[str, str] = {
    "amt": (
        r"\b(transaction|purchase|charge|payment) amount\b|\bamount of \$"
        r"|\b(large|high|big|sizable|significant)(-| )(amount|value|purchase|transaction|charge)\b"
    ),
    "amt_vs_card_avg": (
        r"\baverage\b|\b(usual|typical|normal)(ly)? (spend|spending|amount|purchase)"
        r"|\b\d+(\.\d+)?\s?x\b|\btimes (higher|larger|more|the)\b"
    ),
    "category": (
        r"\bcategory\b|\bmerchant type\b|\bonline shopping\b|\bin-store shopping\b|\bgrocer"
        r"|\bgas\b|\btransport|\bentertainment\b|\bdining\b|\brestaurant|\btravel\b"
        r"|\bhealth\b|\bfitness\b|\bhome (goods|improvement)\b|\bkids\b|\bpets?\b"
        r"|\bpersonal care\b|\bmisc"
    ),
    "hour": (
        r"\bhour of (the )?day\b|\btime of (the )?day\b|\b\d{1,2}:\d{2}\b|\b\d{1,2}\s?(am|pm)\b"
        r"|\b(late|night|nighttime|midnight|overnight|early morning|evening|afternoon)\b"
    ),
    "day_of_week": (
        r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekend|weekday)s?\b"
        r"|\bday of (the )?week\b"
    ),
    "txns_prev_1h": r"\b(previous|past|last|preceding) hour\b|\bwithin (an|the|one) hour\b",
    "txns_prev_24h": (
        r"\b(transactions?|purchases?|charges?) (in|over|during|within) the "
        r"(previous|past|last|preceding) (24 hours|day)\b"
    ),
    "amt_prev_24h": (
        r"\b(spend|spent|spending|total)\b[^.]{0,40}\b(24 hours|day)\b"
        r"|\b(24-hour|daily) (spend|spending|total)\b"
    ),
    "mins_since_prev_txn": (
        r"\bminutes? (since|after)\b|\bsince the (card's )?(previous|last|prior) "
        r"(transaction|purchase)\b|\b(shortly|right|soon) after\b|\b(quick|rapid) succession\b"
    ),
    "age": r"\bage\b|\baged\b|\byears old\b|\b\d+-year-old\b|\b\d+ years\b|\belderly\b",
    "distance_km": r"\bdistance\b|\bkm\b|\bkilomet|\bmiles?\b|\bfar (from|away)\b",
    "city_pop": r"\bpopulation\b|\bpopulous\b|\bsmall town\b|\brural\b|\burban\b",
    OUTSIDE_MODEL: (
        r"\bip address\b|\bdevice\b|\bbrowser\b|\bcountry\b|\babroad\b|\bforeign\b|\bgender\b"
        r"|\bincome\b|\bcredit score\b|\bchargeback|\bmerchant name\b|\bemail\b"
    ),
}
_PATTERNS = {name: re.compile(pattern) for name, pattern in _ALIASES.items()}
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def cited_features(text: str) -> set[str]:
    """Features (and the out-of-model bucket) the text talks about."""
    lowered = text.lower()
    return {name for name, pattern in _PATTERNS.items() if pattern.search(lowered)}


def numbers_in(text: str) -> set[float]:
    return {float(match.replace(",", "")) for match in _NUMBER.findall(text)}


def unsupported_numbers(text: str, context: Iterable[str]) -> tuple[float, ...]:
    """Numbers in the text that no fact contains, allowing rounding to 0 or 1 decimals."""
    allowed: set[float] = set()
    for fact in context:
        for number in numbers_in(fact):
            allowed.update({number, round(number), round(number, 1)})
    return tuple(sorted(numbers_in(text) - allowed))


@dataclass(frozen=True)
class Grounding:
    given: tuple[str, ...]  # features SHAP ranked and the prompt listed
    cited: tuple[str, ...]  # features the explanation mentions
    unsupported_numbers: tuple[float, ...]

    @property
    def uncited_by_shap(self) -> tuple[str, ...]:
        """Mentioned, but not among the features the model was given."""
        return tuple(name for name in self.cited if name not in self.given)

    @property
    def coverage(self) -> float:
        """Share of the given features the explanation actually uses."""
        return sum(1 for name in self.given if name in self.cited) / max(len(self.given), 1)

    @property
    def grounded(self) -> bool:
        return not self.uncited_by_shap and not self.unsupported_numbers and self.coverage > 0


def check(explanation: str, given: Sequence[str], facts: Sequence[str]) -> Grounding:
    """Score one explanation against the SHAP-ranked features and the fact lines shown to it."""
    return Grounding(
        given=tuple(given),
        cited=tuple(sorted(cited_features(explanation))),
        unsupported_numbers=unsupported_numbers(explanation, facts),
    )


@dataclass(frozen=True)
class Summary:
    model: str
    n: int
    grounded_rate: float
    off_list_rate: float  # explanations citing a feature SHAP did not rank
    invented_number_rate: float
    mean_coverage: float
    latency_p50_s: float


def summarize(model: str, results: Sequence[Grounding], latencies: Sequence[float]) -> Summary:
    n = len(results)
    if n == 0:
        raise ValueError("no results to summarize")
    return Summary(
        model=model,
        n=n,
        grounded_rate=sum(r.grounded for r in results) / n,
        off_list_rate=sum(bool(r.uncited_by_shap) for r in results) / n,
        invented_number_rate=sum(bool(r.unsupported_numbers) for r in results) / n,
        mean_coverage=statistics.fmean(r.coverage for r in results),
        latency_p50_s=statistics.median(latencies),
    )


def markdown_table(summaries: Sequence[Summary]) -> str:
    rows = [
        "| Model | Alerts | Grounded | Cites a feature SHAP didn't rank | Invented numbers | "
        "SHAP reasons used | Latency p50 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    rows += [
        f"| `{s.model}` | {s.n} | {s.grounded_rate:.0%} | {s.off_list_rate:.0%} | "
        f"{s.invented_number_rate:.0%} | {s.mean_coverage:.0%} | {s.latency_p50_s:.1f}s |"
        for s in summaries
    ]
    return "\n".join(rows)
