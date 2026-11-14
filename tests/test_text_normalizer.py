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
    (2005, "two thousand five"), (1905, "nineteen oh five"), (1066, "ten sixty-six"), (1100, "eleven hundred"),
    (1000, "one thousand"),
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


# Ordinary English sentences, as a listener should hear them.
@pytest.mark.parametrize(("text", "spoken"), [
    # Money with K/M/B, and other currencies
    ("The startup raised $5.2M last year.", "The startup raised five point two million dollars last year."),
    ("A $450K house.", "A four hundred fifty thousand dollars house."),
    ("It cost $3.5B and $2M and $5k.",
     "It cost three point five billion dollars and two million dollars and five thousand dollars."),
    ("The fare was $1,284.50.", "The fare was one thousand two hundred eighty-four dollars and fifty cents."),
    ("It costs ¥1000 or €20 or £3.50.",
     "It costs one thousand yen or twenty euros or three pounds and fifty pence."),
    ("The price is $1,234,567.89.",
     "The price is one million two hundred thirty-four thousand five hundred sixty-seven dollars and eighty-nine cents."),
    ("It cost $1.505 in total.", "It cost one point five zero five dollars in total."),
    ("It cost $2027.", "It cost two thousand twenty-seven dollars."),
    # A letter against a number: left as written
    ("Q3 sales beat Q2.", "Q3 sales beat Q2."),
    ("Save it as MP3 and cut the CO2.", "Save it as MP3 and cut the CO2."),
    ("Watch it in 3D or 4K on 5G.", "Watch it in 3D or 4K on 5G."),
    ("The 100m sprint costs 50p.", "The 100m sprint costs 50p."),
    ("A 1.5x boost in 2.3h at 30kph.", "A 1.5x boost in 2.3h at 30kph."),
    ("Drink H2O, sell B2B, try x2.", "Drink H2O, sell B2B, try x2."),
    ("$5x is not money.", "$5x is not money."),
    # Phone numbers, digit by digit
    ("Call (212) 555-0142 ext. 402 today.",
     "Call two one two, five five five, zero one four two, extension four zero two today."),
    ("Call 555-1234 now.", "Call five five five, one two three four now."),
    ("Call 1-800-555-0199 now.", "Call one, eight zero zero, five five five, zero one nine nine now."),
    ("Dial 212-555-0142.", "Dial two one two, five five five, zero one four two."),
    ("Ask for ext. 12.", "Ask for extension one two."),
    # Years in year context, plain numbers otherwise
    ("1999 people came to the show.", "one thousand nine hundred ninety-nine people came to the show."),
    ("There were 2027 entries.", "There were two thousand twenty-seven entries."),
    ("In 2027 we will move.", "In twenty twenty-seven we will move."),
    ("He has lived here since 1998.", "He has lived here since nineteen ninety-eight."),
    ("It will be done by 2030, from 2026 until 2031.",
     "It will be done by twenty thirty, from twenty twenty-six until twenty thirty-one."),
    ("The class of 1985 met.", "The class of nineteen eighty-five met."),
    ("He was born on March 3rd, 2027.", "He was born on March third, twenty twenty-seven."),
    ("It opened 4 July 1976.", "It opened four July nineteen seventy-six."),
    ("The year 1066 changed England.", "The year ten sixty-six changed England."),
    ("It was built in 1500 AD.", "It was built in fifteen hundred AD."),
    ("He scored 1500 points.", "He scored one thousand five hundred points."),
    ("The best year was 1999.", "The best year was nineteen ninety-nine."),
    ("We moved in 2005 and again in 2010.", "We moved in two thousand five and again in twenty ten."),
    ("It closed in 1900.", "It closed in nineteen hundred."),
    ("The US and UK agreed in 2024.", "The US and UK agreed in twenty twenty-four."),
    ("BORN IN 1990", "BORN IN nineteen ninety"),
    # Decades
    ("The 1990s were wild.", "The nineteen nineties were wild."),
    ("Music of the '80s and the 60s.", "Music of the eighties and the sixties."),
    ("Born in the 1800s or the 1960's.", "Born in the eighteen hundreds or the nineteen sixties."),
    # Titles and street abbreviations
    ("Dr. Smith lives on Elm Dr. in town.", "Doctor Smith lives on Elm Drive in town."),
    ("I saw the Dr. Then I left.", "I saw the Doctor. Then I left."),
    ("He lives at 12 Main St. Then he left.", "He lives at twelve Main Street. Then he left."),
    ("Turn onto Oak St. and stop.", "Turn onto Oak Street and stop."),
    ("St. Louis is far.", "Saint Louis is far."),
    ("We went to St. Paul.", "We went to Saint Paul."),
    ("Mr. and Mrs. Jones met Prof. Lee.", "Mister and Missus Jones met Professor Lee."),
    ("Ask the Dr. about it.", "Ask the Doctor about it."),
    ("John Smith Jr. arrived. Then Mt. Fuji.", "John Smith Junior arrived. Then Mount Fuji."),
    # Negative numbers and amounts
    ("It was -5 degrees.", "It was minus five degrees."),
    ("The balance is -$20.", "The balance is minus twenty dollars."),
    ("-3 is less than 0.", "minus three is less than zero."),
    # Ranges
    ("Take 2-3 tablets, 2–3 times a day.", "Take two to three tablets, two to three times a day."),
    ("Read pages 10–20 tonight.", "Read pages ten to twenty tonight."),
    ("The war lasted 1939-1945.", "The war lasted nineteen thirty-nine to nineteen forty-five."),
    ("The score was 21-14.", "The score was twenty-one-fourteen."),
    ("A 3-year-old and a 10-minute walk.", "A three-year-old and a ten-minute walk."),
    ("COVID-19 hit in 2020.", "COVID-nineteen hit in twenty twenty."),
    # Fractions
    ("Add 1/2 a cup and 3/4 of a spoon.", "Add one half a cup and three quarters of a spoon."),
    ("About 2/3 agreed and 1/8 left.", "About two thirds agreed and one eighth left."),
    ("Open 24/7.", "Open twenty-four/seven."),
    # Times
    ("The race ended at 10:30:45.", "The race ended at ten thirty and forty-five seconds."),
    ("Meet me at 6:45 p.m. tomorrow.", "Meet me at six forty-five P M tomorrow."),
    ("Wake up at 7am.", "Wake up at seven A M."),
    ("It's 9:00. Then lunch.", "It's nine o'clock. Then lunch."),
    # All capitals before a number
    ("TOP 10 tips: GET 50% OFF.", "TOP ten tips: GET fifty percent OFF."),
    ("I WON 3 TIMES.", "I WON three TIMES."),
    ("Flight BA 2049 is late.", "Flight B A twenty forty-nine is late."),
    # Plain numbers
    ("We had 3 cats, 2 dogs and 1 fish.", "We had three cats, two dogs and one fish."),
    ("Room 007 is 0.5 miles away.", "Room zero zero seven is zero point five miles away."),
    ("She came 1st and he came 22nd.", "She came first and he came twenty-second."),
    ("Version 0.12.0 is out.", "Version 0.12.0 is out."),
    ("Use iPhone 15 on Windows 11.", "Use iPhone fifteen on Windows eleven."),
])
def test_ordinary_sentences(text, spoken):
    assert normalize_english(text) == spoken


def test_text_without_numbers_and_versions_stay_as_they_are():
    assert normalize_english("Hello there, how are you?") == "Hello there, how are you?"
    assert normalize_english("Version 0.12.0 ships") == "Version 0.12.0 ships"
    assert normalize_english("The doctor is in.") == "The doctor is in."
