"""English text normalization for models that read raw characters: numbers, money, dates, times and titles as words."""

from __future__ import annotations

import re

_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen"
).split()
_TENS = "_ ten twenty thirty forty fifty sixty seventy eighty ninety".split()
_SCALES = ((10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (1000, "thousand"))
_ORDINAL_WORDS = {
    "one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth",
    "twelve": "twelfth",
}
_SCALE_SUFFIXES = {"k": "thousand", "m": "million", "mn": "million", "b": "billion", "bn": "billion"}
_SCALE_WORDS = ("thousand", "million", "billion", "trillion")
_CURRENCIES = {"$": ("dollar", "dollars", "cent", "cents"), "£": ("pound", "pounds", "penny", "pence"),
               "€": ("euro", "euros", "cent", "cents"), "¥": ("yen", "yen", None, None)}
# Two capitals before a number are read letter by letter ("BA 2049"), unless they spell a word.
_SPELLED_WORDS = frozenset({
    "AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "HE", "IF", "IN", "IS", "IT", "ME", "MY", "NO", "OF", "OK", "ON",
    "OR", "SO", "TO", "UP", "US", "WE",
})

_NUMBER = r"\d{1,3}(?:,\d{3})+|\d+"
# A number on its own: no letter or digit against it ("Q3", "4K", "2.3h" stay), not part of a version ("0.12.0").
_ALONE = r"(?<![\w.])(?<!\d,)"
_END = r"(?![\w]|[.,]\d)"

_ABBREVIATION = re.compile(r"\b(Dr|Mr|Mrs|Ms|Prof|St|Mt|Jr|Sr)\.|\b(vs)\.(?=\s)")
_PHONE = re.compile(
    r"(?<![\w-])((?:1-)?(?:\(\d{3}\)\s?|\d{3}-)?\d{3}-\d{4})(?![\w-]|[.,]\d)"
    r"(?:,?\s(?:ext\.?|extension)\s?(\d{1,6})\b)?",
    re.IGNORECASE,
)
_EXTENSION = re.compile(r"\bext\.?\s?(\d{1,6})\b", re.IGNORECASE)
_RANGE = re.compile(r"(?<![\w.,:/$£€¥–-])(\d{1,4})(?:-|\s?–\s?)(\d{1,4})(?![\w/:–-]|[.,]\d)")
_NEGATIVE = re.compile(r"(?<![\w.)])[-−](?=[$£€¥]?\d)")
_MONEY = re.compile(
    rf"([$£€¥])\s?({_NUMBER})(?:\.(\d+))?(?:([A-Za-z]+)|\s?({'|'.join(_SCALE_WORDS)})\b)?(?!\d)"
)
_TIME = re.compile(
    r"\b([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?(?![\d:])(?:\s?([ap])\.?\s?m\b\.?)?", re.IGNORECASE
)
_HOUR_MERIDIEM = re.compile(r"\b(1[0-2]|0?[1-9])\s?([ap])\.?\s?m\b\.?", re.IGNORECASE)
_PERCENT = re.compile(rf"{_ALONE}({_NUMBER})(?:\.(\d+))?\s?%")
_SHORT_DECADE = re.compile(r"(?<![\w'])'?([1-9])0'?s\b")
_DECADE = re.compile(r"(?<![\w.,])(1\d|20)(\d)0'?s\b")
_FRACTION = re.compile(r"(?<![\w.,/:])(\d{1,2})/(\d{1,2})(?![\w/]|[.,]\d)")
_CODE = re.compile(rf"\b([A-Z]{{2}})\s(\d{{1,4}}){_END}")
_YEAR = re.compile(rf"{_ALONE}(1\d{{3}}|20\d{{2}}){_END}")
_ORDINAL = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)
_DECIMAL = re.compile(rf"{_ALONE}({_NUMBER})\.(\d+){_END}")
_INTEGER = re.compile(rf"{_ALONE}({_NUMBER}){_END}")
_SENTENCE_START = re.compile(r"\s*(?:$|[A-Z\"'(])")

_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December|"
    "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
_DAYS = "Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
_YEAR_WORD_BEFORE = re.compile(r"\b(?:in|since|by|of|from|until|to|year)\s+$", re.IGNORECASE)
_DATE_BEFORE = re.compile(
    rf"(?:\b(?:{_MONTHS})\.?,?\s+(?:\d{{1,2}}(?:st|nd|rd|th)?,?\s+)?|\b(?:{_DAYS}),?\s+|\b\d{{1,2}}/\d{{1,2}}/)$"
)
_YEAR_AFTER = re.compile(r"\s+(?:AD|BC|BCE|CE)\b|\s+to\s+(?:1\d{3}|20\d{2})\b")
_SENTENCE_END_AFTER = re.compile(r"[\"')\]”’]*(?:[.!?]+(?=[\s\"')\]”’]|$)|\s*(?:\n|$))")
# "Elm St.", "12 Main St.", "5th St.": a street, not "St. Louis" or a title.
_CAPITAL_BEFORE = re.compile(r"[^\s.!?:]\s+[A-Z][\w'-]*\s+$")
_NUMBER_BEFORE = re.compile(r"\b\d+\w*(?:\s+[A-Za-z][\w'-]*)?\s+$")
_DETERMINER_BEFORE = re.compile(r"\b(?:the|a|my|your|his|her|our|their|this|that)\s+$", re.IGNORECASE)


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
    """2027 -> "twenty twenty-seven", 1900 -> "nineteen hundred", 2005 -> "two thousand five", 1066 -> "ten sixty-six"."""
    high, low = divmod(n, 100)
    if 1000 <= n < 1010 or 2000 <= n < 2010:
        return cardinal(n)
    if low == 0:
        return f"{cardinal(high)} hundred"
    return f"{cardinal(high)} {'oh ' + _ONES[low] if low < 10 else cardinal(low)}"


def _plural(words: str) -> str:
    return words[:-1] + "ies" if words.endswith("y") else words + "s"


def _integer(text: str) -> int:
    return int(text.replace(",", ""))


def _digits(text: str) -> str:
    return " ".join(_ONES[int(d)] for d in text)


def _abbreviation(match: re.Match) -> str:
    if match.group(2):
        return "versus"
    word = match.group(1)
    before, after = match.string[: match.start()], match.string[match.end():]
    next_upper = re.match(r"\s+[A-Z]", after)
    next_word = re.match(r"\s+\w", after)
    sentence_end = bool(next_upper) or not after.strip() or after.startswith("\n")
    after_number = bool(_NUMBER_BEFORE.search(before))
    after_name = bool(_CAPITAL_BEFORE.search(before))
    if (word == "St" and (after_name or after_number)) or (
        word == "Dr" and (after_number or (after_name and not next_upper))
    ):
        return ("Street" if word == "St" else "Drive") + ("." if sentence_end else "")
    if word == "St":
        return "Saint" if next_upper else match.group(0)
    if word == "Mt":
        return "Mount" if next_upper else match.group(0)
    spoken = {"Dr": "Doctor", "Mr": "Mister", "Mrs": "Missus", "Ms": "Miz", "Prof": "Professor",
              "Jr": "Junior", "Sr": "Senior"}[word]
    if word in ("Jr", "Sr"):
        return spoken + ("." if sentence_end else "")
    if word == "Dr" and sentence_end and (not next_word or _DETERMINER_BEFORE.search(before)):
        return spoken + "."
    if not next_word and sentence_end:
        return spoken + "."
    return spoken


def _phone(match: re.Match) -> str:
    number, extension = match.group(1), match.group(2)
    spoken = ", ".join(_digits(group) for group in re.findall(r"\d+", number))
    return spoken + (f", extension {_digits(extension)}" if extension else "")


def _range(match: re.Match) -> str:
    low, high = match.group(1), match.group(2)
    if int(low) >= int(high) or any(len(n) > 1 and n.startswith("0") for n in (low, high)):
        return match.group(0)
    return f"{low} to {high}"


def _money(match: re.Match) -> str:
    one, many, sub_one, sub_many = _CURRENCIES[match.group(1)]
    whole, cents, letters = _integer(match.group(2)), match.group(3), match.group(4)
    scale = match.group(5)
    if letters:
        scale = _SCALE_SUFFIXES.get(letters.lower()) or (letters.lower() if letters.lower() in _SCALE_WORDS else None)
        if scale is None:
            return match.group(0)
    if scale or (cents and (len(cents) > 2 or sub_one is None)):
        amount = cardinal(whole) + (f" point {_digits(cents)}" if cents else "")
        return f"{amount} {scale} {many}" if scale else f"{amount} {many}"
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
    hour, minute, second, letter = int(match.group(1)), int(match.group(2)), match.group(3), match.group(4)
    if minute == 0:
        spoken = cardinal(hour) if letter else f"{cardinal(hour)} o'clock"
    elif minute < 10:
        spoken = f"{cardinal(hour)} oh {_ONES[minute]}"
    else:
        spoken = f"{cardinal(hour)} {cardinal(minute)}"
    if second is not None:
        seconds = int(second)
        spoken += f" and {cardinal(seconds)} second{'' if seconds == 1 else 's'}"
    return f"{spoken} {_meridiem(letter, match)}" if letter else spoken


def _fraction(match: re.Match) -> str:
    top, bottom = int(match.group(1)), int(match.group(2))
    if not 1 <= top < bottom <= 10:
        return match.group(0)
    name = {2: "half", 4: "quarter"}.get(bottom) or ordinal(bottom)
    if top > 1:
        name = "halves" if bottom == 2 else name + "s"
    return f"{cardinal(top)} {name}"


def _code(match: re.Match) -> str:
    letters, number = match.group(1), match.group(2)
    if letters in _SPELLED_WORDS:
        return match.group(0)
    spoken = year(int(number)) if len(number) == 4 and not number.startswith("0") else number
    return f"{' '.join(letters)} {spoken}"


def _year(match: re.Match) -> str:
    value = int(match.group(1))
    before, after = match.string[: match.start()], match.string[match.end():]
    if (
        _YEAR_WORD_BEFORE.search(before) or _DATE_BEFORE.search(before) or _YEAR_AFTER.match(after)
        or (1100 <= value and _SENTENCE_END_AFTER.match(after))
    ):
        return year(value)
    return match.group(0)


def _number(match: re.Match) -> str:
    text = match.group(1)
    if len(text) > 1 and text.startswith("0"):
        return _digits(text)
    return cardinal(_integer(text))


def normalize_english(text: str) -> str:
    """Spell out what a character-level English voice misreads: "$1,284.50", "6:45 p.m.", "March 3rd, 2027", "Dr."."""
    text = _ABBREVIATION.sub(_abbreviation, text)
    text = text.replace("&", " and ")
    text = _PHONE.sub(_phone, text)
    text = _EXTENSION.sub(lambda m: f"extension {_digits(m.group(1))}", text)
    text = _RANGE.sub(_range, text)
    text = _NEGATIVE.sub("minus ", text)
    text = _MONEY.sub(_money, text)
    text = _TIME.sub(_time, text)
    text = _HOUR_MERIDIEM.sub(lambda m: f"{cardinal(int(m.group(1)))} {_meridiem(m.group(2), m)}", text)
    text = _PERCENT.sub(
        lambda m: cardinal(_integer(m.group(1))) + (f" point {_digits(m.group(2))}" if m.group(2) else "") + " percent",
        text,
    )
    text = _DECADE.sub(lambda m: _plural(year(int(m.group(1) + m.group(2) + "0"))), text)
    text = _SHORT_DECADE.sub(lambda m: _plural(_TENS[int(m.group(1))]), text)
    text = _FRACTION.sub(_fraction, text)
    text = _CODE.sub(_code, text)
    text = _YEAR.sub(_year, text)
    text = _ORDINAL.sub(lambda m: ordinal(int(m.group(1))), text)
    text = _DECIMAL.sub(lambda m: f"{cardinal(_integer(m.group(1)))} point {_digits(m.group(2))}", text)
    return _INTEGER.sub(_number, text)
