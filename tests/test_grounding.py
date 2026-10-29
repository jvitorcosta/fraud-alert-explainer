import pytest

from fraud_alert_explainer.grounding import (
    OUTSIDE_MODEL,
    Grounding,
    check,
    cited_features,
    markdown_table,
    summarize,
    unsupported_numbers,
)

FACTS = [
    "model fraud score: 0.97",
    "transaction amount: $1,077.69",
    "hour of day: 22:00 (10 PM)",
    "merchant category: online shopping",
]
GIVEN = ["amt", "hour", "category"]


@pytest.mark.parametrize(
    ("text", "feature"),
    [
        ("The transaction amount of $1,077.69 is high.", "amt"),
        ("It is 6.2x the card's average purchase.", "amt_vs_card_avg"),
        ("The purchase was for online shopping.", "category"),
        ("It happened late at night, at 22:00.", "hour"),
        ("It took place on a Sunday.", "day_of_week"),
        ("There were 3 transactions in the previous hour.", "txns_prev_1h"),
        ("The card had 9 transactions in the previous 24 hours.", "txns_prev_24h"),
        ("Total spend over the past 24 hours was $2,000.", "amt_prev_24h"),
        ("It came 2.0 minutes since the previous transaction.", "mins_since_prev_txn"),
        ("The cardholder is 74 years old.", "age"),
        ("The merchant is 120 km from home.", "distance_km"),
        ("The cardholder lives in a small town.", "city_pop"),
        ("The purchase came from a new device abroad.", OUTSIDE_MODEL),
    ],
)
def test_cited_features_recognises_each_feature(text: str, feature: str) -> None:
    assert feature in cited_features(text)


def test_cited_features_does_not_read_home_distance_as_a_category() -> None:
    assert cited_features("The merchant is far from the cardholder's home.") == {"distance_km"}


def test_unsupported_numbers_allows_rounding_of_given_facts() -> None:
    assert unsupported_numbers("About $1,078, at 10 PM, score 0.97.", FACTS) == ()


def test_unsupported_numbers_flags_invented_figures() -> None:
    assert unsupported_numbers("That is 5 times the usual $200.", FACTS) == (5.0, 200.0)


def test_check_grounded_when_only_given_features_are_cited() -> None:
    text = (
        "The model flagged this $1,077.69 online shopping purchase because of its high "
        "transaction amount and because it happened at 10 PM."
    )

    result = check(text, GIVEN, FACTS)

    assert result.grounded
    assert result.coverage == pytest.approx(1.0)


def test_check_flags_features_shap_did_not_rank() -> None:
    text = "The transaction amount is high and the cardholder is 74 years old."

    result = check(text, GIVEN, FACTS)

    assert not result.grounded
    assert result.uncited_by_shap == ("age",)


def test_check_is_not_grounded_when_it_cites_nothing() -> None:
    result = check("This looks suspicious.", GIVEN, FACTS)

    assert not result.grounded
    assert result.coverage == 0.0


def test_summarize_and_table_report_rates() -> None:
    good = Grounding(given=("amt",), cited=("amt",), unsupported_numbers=())
    off_list = Grounding(given=("amt",), cited=("age", "amt"), unsupported_numbers=(5.0,))

    summary = summarize("qwen2.5:1.5b", [good, off_list], [1.0, 3.0])

    assert summary.grounded_rate == pytest.approx(0.5)
    assert summary.off_list_rate == pytest.approx(0.5)
    assert summary.invented_number_rate == pytest.approx(0.5)
    assert summary.latency_p50_s == pytest.approx(2.0)
    assert "| `qwen2.5:1.5b` | 2 | 50% | 50% | 50% | 100% | 2.0s |" in markdown_table([summary])


def test_summarize_rejects_empty_results() -> None:
    with pytest.raises(ValueError, match="no results"):
        summarize("m", [], [])
