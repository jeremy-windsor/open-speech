"""English text normalization: what a character-level voice (Supertonic) would misread, spelled out."""

from __future__ import annotations

import pytest

from src.tts.text_normalizer import cardinal, normalize_english, ordinal, year


def test_the_flight_sentence_reads_as_words():
    text = "Dr. Okafor's flight, BA 2049, leaves Heathrow at 6:45 p.m. on March 3rd, 2027; the fare was $1,284.50."
    assert normalize_english(text) == (
        "Doctor Okafor's flight, B A twenty forty-nine, leaves Heathrow at six forty-five P M on March third, "
        "twenty twenty-seven; the fare was one thousand two hundred eighty-four dollars and fifty cents."
    )


@pytest.mark.parametrize(("n", "words"), [
    (0, "zero"), (7, "seven"), (15, "fifteen"), (40, "forty"), (84, "eighty-four"), (100, "one hundred"),
    (1284, "one thousand two hundred eighty-four"), (1_000_000, "one million"),
    (2_500_013, "two million five hundred thousand thirteen"),
])
def test_cardinal(n, words):
    assert cardinal(n) == words


@pytest.mark.parametrize(("n", "words"), [
    (1, "first"), (2, "second"), (3, "third"), (5, "fifth"), (8, "eighth"), (9, "ninth"), (12, "twelfth"),
    (20, "twentieth"), (21, "twenty-first"), (40, "fortieth"), (101, "one hundred first"),
])
def test_ordinal(n, words):
    assert ordinal(n) == words


@pytest.mark.parametrize(("n", "words"), [
    (2027, "twenty twenty-seven"), (1999, "nineteen ninety-nine"), (1900, "nineteen hundred"),
    (2005, "two thousand five"), (1905, "nineteen oh five"),
])
def test_year(n, words):
    assert year(n) == words


@pytest.mark.parametrize(("text", "spoken"), [
    ("It costs $5.", "It costs five dollars."),
    ("Only $0.99 today", "Only ninety-nine cents today"),
    ("A $1 coin and $2.05", "A one dollar coin and two dollars and five cents"),
    ("Raised $1.5 million", "Raised one point five million dollars"),
    ("£3.05 or €1", "three pounds and five pence or one euro"),
    ("We leave at 6 p.m. Then we eat.", "We leave at six P M. Then we eat."),
    ("At 7:00 or 10:05 or 12:30 am", "At seven o'clock or ten oh five or twelve thirty A M"),
    ("Up 50% and 2.5% more", "Up fifty percent and two point five percent more"),
    ("Pi is 3.14", "Pi is three point one four"),
    ("Mr. Smith vs. Jones, Jr., said so", "Mister Smith versus Jones, Junior, said so"),
    ("Agent 007", "Agent zero zero seven"),
    ("1,000,000 fans and 42 cats", "one million fans and forty-two cats"),
    ("Salt & pepper", "Salt  and  pepper"),
])
def test_normalize_english(text, spoken):
    assert normalize_english(text) == spoken


def test_text_without_numbers_and_versions_stay_as_they_are():
    assert normalize_english("Hello there, how are you?") == "Hello there, how are you?"
    assert normalize_english("Version 0.12.0 ships") == "Version 0.12.0 ships"
    assert normalize_english("The doctor is in.") == "The doctor is in."
