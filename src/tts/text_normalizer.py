"""English text normalization for models that read raw characters: numbers, money, dates, times and titles as words."""

from __future__ import annotations

import re

_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen"
).split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_SCALES = ((10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (1000, "thousand"))
_ORDINAL_WORDS = {
    "one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth",
    "twelve": "twelfth",
}

_TITLES = {
    "Dr": "Doctor", "Mr": "Mister", "Mrs": "Missus", "Ms": "Miz", "Prof": "Professor", "Jr": "Junior",
    "Sr": "Senior", "St": "Saint", "Mt": "Mount", "vs": "versus",
}
_TITLE = re.compile(r"\b(Dr|Mr|Mrs|Ms|Prof|St|Mt)\.(?=\s+[A-Z])|\b(Jr|Sr)\.(?=,)|\b(vs)\.(?=\s)")
_CURRENCIES = {"$": ("dollar", "dollars", "cent", "cents"), "£": ("pound", "pounds", "penny", "pence"),
               "€": ("euro", "euros", "cent", "cents")}
_NUMBER = r"\d{1,3}(?:,\d{3})+|\d+"
_MONEY = re.compile(
    rf"([$£€])\s?({_NUMBER})(?:\.(\d{{1,2}}))?(?:\s?(thousand|million|billion|trillion)\b)?"
)
_TIME = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)(?:\s?([ap])\.?\s?m\b\.?)?", re.IGNORECASE)
_HOUR_MERIDIEM = re.compile(r"\b(1[0-2]|0?[1-9])\s?([ap])\.?\s?m\b\.?", re.IGNORECASE)
_ORDINAL = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)
_PERCENT = re.compile(rf"({_NUMBER})(?:\.(\d+))?\s?%")
_CODE = re.compile(r"\b([A-Z]{2,3})(?=\s?\d)")
# A number on its own, not part of a version ("0.12.0"); a comma or full stop after it may end a clause.
_ALONE = r"(?<![\d.])(?<!\d,)"
_DECIMAL = re.compile(rf"{_ALONE}({_NUMBER})\.(\d+)(?!\d|\.\d)")
_INTEGER = re.compile(rf"{_ALONE}({_NUMBER})(?!\d|[.,]\d)")
_SENTENCE_START = re.compile(r"\s*(?:$|[A-Z\"'(])")


def cardinal(n: int) -> str:
    """1284 -> "one thousand two hundred eighty-four"."""
    if n < 0:
        return f"minus {cardinal(-n)}"
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return f"{_ONES[hundreds]} hundred" + (f" {cardinal(rest)}" if rest else "")
    for value, word in _SCALES:
        if n >= value:
            high, rest = divmod(n, value)
            return f"{cardinal(high)} {word}" + (f" {cardinal(rest)}" if rest else "")
    raise AssertionError(n)


def ordinal(n: int) -> str:
    """3 -> "third", 21 -> "twenty-first", 40 -> "fortieth"."""
    words = cardinal(n)
    cut = max(words.rfind(" "), words.rfind("-")) + 1
    head, last = words[:cut], words[cut:]
    if last in _ORDINAL_WORDS:
        last = _ORDINAL_WORDS[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return head + last


def year(n: int) -> str:
    """2027 -> "twenty twenty-seven", 1900 -> "nineteen hundred", 2005 -> "two thousand five"."""
    high, low = divmod(n, 100)
    if 2000 <= n < 2010:
        return cardinal(n)
    if low == 0:
        return f"{cardinal(high)} hundred"
    return f"{cardinal(high)} {'oh ' + _ONES[low] if low < 10 else cardinal(low)}"


def _integer(text: str) -> int:
    return int(text.replace(",", ""))


def _digits(text: str) -> str:
    return " ".join(_ONES[int(d)] for d in text)


def _money(match: re.Match) -> str:
    one, many, sub_one, sub_many = _CURRENCIES[match.group(1)]
    whole, cents, scale = _integer(match.group(2)), match.group(3), match.group(4)
    if scale:
        amount = cardinal(whole) + (f" point {_digits(cents)}" if cents else "")
        return f"{amount} {scale} {many}"
    cents_value = int(cents.ljust(2, "0")) if cents else 0
    parts = []
    if whole or not cents_value:
        parts.append(f"{cardinal(whole)} {one if whole == 1 else many}")
    if cents_value:
        parts.append(f"{cardinal(cents_value)} {sub_one if cents_value == 1 else sub_many}")
    return " and ".join(parts)


def _meridiem(letter: str, match: re.Match) -> str:
    # A sentence that ended on "p.m." keeps its full stop.
    keep_stop = match.group(0).endswith(".") and _SENTENCE_START.match(match.string, match.end())
    return f"{letter.upper()} M" + ("." if keep_stop else "")


def _time(match: re.Match) -> str:
    hour, minute, letter = int(match.group(1)), int(match.group(2)), match.group(3)
    if minute == 0:
        spoken = cardinal(hour) if letter else f"{cardinal(hour)} o'clock"
    elif minute < 10:
        spoken = f"{cardinal(hour)} oh {_ONES[minute]}"
    else:
        spoken = f"{cardinal(hour)} {cardinal(minute)}"
    return f"{spoken} {_meridiem(letter, match)}" if letter else spoken


def _number(match: re.Match) -> str:
    text = match.group(1)
    value = _integer(text)
    if "," not in text and len(text) == 4 and 1100 <= value <= 2099:
        return year(value)
    if len(text) > 1 and text.startswith("0"):
        return _digits(text)
    return cardinal(value)


def normalize_english(text: str) -> str:
    """Spell out what a character-level English voice misreads: "$1,284.50", "6:45 p.m.", "March 3rd, 2027", "Dr."."""
    text = _TITLE.sub(lambda m: _TITLES[m.group(1) or m.group(2) or m.group(3)], text)
    text = text.replace("&", " and ")
    text = _MONEY.sub(_money, text)
    text = _TIME.sub(_time, text)
    text = _HOUR_MERIDIEM.sub(lambda m: f"{cardinal(int(m.group(1)))} {_meridiem(m.group(2), m)}", text)
    text = _PERCENT.sub(
        lambda m: cardinal(_integer(m.group(1))) + (f" point {_digits(m.group(2))}" if m.group(2) else "") + " percent",
        text,
    )
    text = _ORDINAL.sub(lambda m: ordinal(int(m.group(1))), text)
    text = _CODE.sub(lambda m: " ".join(m.group(1)), text)
    text = _DECIMAL.sub(lambda m: f"{cardinal(_integer(m.group(1)))} point {_digits(m.group(2))}", text)
    return _INTEGER.sub(_number, text)
