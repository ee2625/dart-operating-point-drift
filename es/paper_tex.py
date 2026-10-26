"""LaTeX text helpers and the rounding rules every generated number follows.

Rounding is half-up on the decimal string (decimal.ROUND_HALF_UP), the rule FORENSICS' comparison
uses: a regenerated value matches a printed one when it rounds to it. Every shift (a fraction to a
percentage) and every rounding happens in Decimal, never in binary floating point: 0.0045 as a
percentage is 0.45%, which rounds to 0.5%, although 100 * 0.0045 is 0.44999999999999996 in binary.
Negative numbers print a real minus. Macros are text-mode: inside math write \\text{\\macro}."""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal, localcontext
from typing import Any, Callable

MINUS = "\\textminus{}"
_MACRO = re.compile(r"[A-Za-z]+")
#: Plain text in, LaTeX out, one character at a time: a replacement is never scanned again, so nothing is escaped twice.
#: | < > are mapped because OT1 (pdflatex's default) prints them as an em dash, an inverted ! and an inverted ?.
#: Strings that already hold LaTeX (the formatters' output, "$<$0.001", "\\$", "$\\times$") must not be passed through.
_ESCAPES = {"&": "\\&", "%": "\\%", "$": "\\$", "#": "\\#", "_": "\\_", "{": "\\{", "}": "\\}",
            "~": "\\textasciitilde{}", "^": "\\textasciicircum{}", "\\": "\\textbackslash{}",
            "|": "\\textbar{}", "<": "\\textless{}", ">": "\\textgreater{}"}


def is_missing(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    return isinstance(x, str) and x.strip() == ""


def macro_name_ok(name: str) -> bool:
    return _MACRO.fullmatch(name) is not None


def tex_escape(s: Any) -> str:
    return "".join(_ESCAPES.get(ch, ch) for ch in str(s))


def _decimal(x: Any) -> Decimal:
    """The shortest decimal string of a float (its repr) as a Decimal: the digits a reader of the CSV sees."""
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"non-finite value {x!r}")
    return Decimal(repr(v))


def _quantize(d: Decimal, exp: int) -> Decimal:
    """Round half up to the decimal position 10**exp. The context holds more digits than any float has, so a big value
    never raises decimal.InvalidOperation."""
    with localcontext() as ctx:
        ctx.prec = 400
        return d.quantize(Decimal(1).scaleb(exp), rounding=ROUND_HALF_UP)


def _fixed(d: Decimal, places: int) -> str:
    s = f"{_quantize(d, -places):f}"
    return s if places > 0 else s.split(".")[0]


def round_half_up(x: float, places: int) -> str:
    return _fixed(_decimal(x), places)


def _thousands(digits: str) -> str:
    whole, _, frac = digits.partition(".")
    groups: list[str] = []
    while len(whole) > 3:
        groups.insert(0, whole[-3:])
        whole = whole[:-3]
    groups.insert(0, whole)
    return "{,}".join(groups) + (("." + frac) if frac else "")


def _signed(x: float, body: str) -> str:
    nonzero = set(body) - set("0.,{}\\$")
    return (MINUS if float(x) < 0 and nonzero else "") + body


def _require(x: Any) -> float:
    """The value as a finite float, or ValueError: a missing, non-numeric, non-finite or too large value never prints."""
    if is_missing(x):
        raise ValueError("missing value")
    try:
        v = float(x)
    except OverflowError as exc:                                  # an int beyond the float range
        raise ValueError(f"{x!r} is too large") from exc
    except TypeError as exc:
        raise ValueError(f"{x!r} is not a number") from exc
    if not math.isfinite(v):
        raise ValueError(f"non-finite value {x!r}")
    return v


def fmt_dec(x: Any, places: int) -> str:
    v = _require(x)
    return _signed(v, round_half_up(abs(v), places))


def fmt_int(x: Any) -> str:
    v = _require(x)
    if abs(v - round(v)) > 1e-9:
        raise ValueError(f"{x!r} is not an integer")
    return _signed(v, _thousands(str(abs(int(round(v))))))


def fmt_pct(x: Any, places: int = 1) -> str:
    """A fraction as a percentage: shifted by 100 in Decimal (scaleb), then rounded half up."""
    v = _require(x)
    return _signed(v, _fixed(_decimal(abs(v)).scaleb(2), places)) + "\\%"


def fmt_usd(x: Any, places: int = 0) -> str:
    v = _require(x)
    return _signed(v, "\\$" + _thousands(round_half_up(abs(v), places)))


def fmt_pval(p: Any) -> str:
    v = _require(p)
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"p-value {p!r} is outside [0, 1]")
    return "$<$0.001" if v < 0.001 else round_half_up(v, 3)


def fmt_g(x: Any, sig: int = 3) -> str:
    """x to ``sig`` significant digits, rounded half up on the decimal string at the adjusted exponent, trailing zeros
    dropped, and always plain digits (never 1e-05 or 1.2e+06), so that the macro reads in running text."""
    if sig < 1:
        raise ValueError(f"sig must be at least 1, got {sig}")
    v = _require(x)
    d = _decimal(abs(v))
    if d == 0:
        return "0"
    s = f"{_quantize(d, d.adjusted() - sig + 1):f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return _signed(v, s)


def fmt_frac(k: Any, n: Any) -> str:
    return f"{fmt_int(k)}/{fmt_int(n)}"


def fmt_ci(lo: Any, hi: Any, one: Callable[[Any], str]) -> str:
    return f"[{one(lo)}, {one(hi)}]"


_YES, _NO = ("true", "1", "yes"), ("false", "0", "no")


def fmt_yesno(v: Any) -> str:
    """yes or no for a boolean (True, "true", 1, "yes", ...); any other value is an error, never a silent "no"."""
    s = str(v).strip().lower()
    if s in _YES:
        return "yes"
    if s in _NO:
        return "no"
    raise ValueError(f"cannot read {v!r} as yes or no")


def fmt_registered(v: Any) -> str:
    """The registered status of a test from its boolean `confirmatory`: "confirmatory", or "estimand" when the size check
    demoted it to an estimand with an interval. Anything that is not a boolean is an error, never a silent "estimand"."""
    return "confirmatory" if fmt_yesno(v) == "yes" else "estimand"


def fmt_utc_minute(v: Any) -> str:
    """An ISO 8601 timestamp with a UTC offset (a "Z" too) as "2026-10-25 15:30 UTC": the stamp of a run or of a registration, converted
    to UTC and cut at the minute. A date without a time, a time without an offset or anything else is an error, never a guess."""
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"cannot read {v!r} as a timestamp")
    try:
        stamp = datetime.fromisoformat(v.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"cannot read {v!r} as an ISO 8601 timestamp") from exc
    if stamp.tzinfo is None or "T" not in v:
        raise ValueError(f"{v!r} has no time and UTC offset (for example 2026-10-25T15:30:00Z)")
    return stamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def fmt_approx_label(v: Any) -> str:
    """What an `approximate` flag says, as words that read in a sentence: "labelled approximate" or "not labelled approximate"."""
    return "labelled approximate" if fmt_yesno(v) == "yes" else "not labelled approximate"


def fmt_reasons(v: Any) -> str:
    """The reasons of a demotion (a list of strings, or one string) as one text, separated by semicolons; "none" for no
    reason, so that a macro is never empty."""
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, (list, tuple)) or not all(isinstance(r, str) for r in v):
        raise ValueError(f"cannot read {v!r} as a list of reasons")
    return tex_escape("; ".join(r.strip() for r in v)) if v else "none"


def fmt_where(values: Any) -> str:
    """The labels of a row (law, scenario, persistence), joined by commas, as far as the row has them: an H-E row has no law."""
    parts = [str(v) for v in values if not is_missing(v)]
    if not parts:
        raise ValueError("the row has none of the labels")
    return tex_escape(", ".join(parts))


FORMATS: dict[str, Callable[[Any], str]] = {
    "int": fmt_int,
    "dec1": lambda v: fmt_dec(v, 1), "dec2": lambda v: fmt_dec(v, 2),
    "dec3": lambda v: fmt_dec(v, 3), "dec4": lambda v: fmt_dec(v, 4),
    "pct0": lambda v: fmt_pct(v, 0), "pct1": lambda v: fmt_pct(v, 1), "pct2": lambda v: fmt_pct(v, 2),
    "usd0": lambda v: fmt_usd(v, 0), "usd2": lambda v: fmt_usd(v, 2),
    "pval": fmt_pval, "g3": lambda v: fmt_g(v, 3),
    "text": tex_escape, "raw": lambda v: str(v),
    "yesno": fmt_yesno, "registered": fmt_registered, "reasons": fmt_reasons,
    "url": lambda v: "\\url{" + str(v) + "}",
    "date": lambda v: tex_escape(str(v)[:10]),
    "utcmin": fmt_utc_minute, "approxlabel": fmt_approx_label,
}
PAIR_FORMATS: dict[str, Callable[[Any, Any], str]] = {
    "frac": fmt_frac,
    "ci1": lambda lo, hi: fmt_ci(lo, hi, FORMATS["dec1"]),
    "ci2": lambda lo, hi: fmt_ci(lo, hi, FORMATS["dec2"]),
    "ci3": lambda lo, hi: fmt_ci(lo, hi, FORMATS["dec3"]),
    "cipct1": lambda lo, hi: fmt_ci(lo, hi, FORMATS["pct1"]),
    "ciusd0": lambda lo, hi: fmt_ci(lo, hi, FORMATS["usd0"]),
    "ciusd2": lambda lo, hi: fmt_ci(lo, hi, FORMATS["usd2"]),
}


#: Formats of a value made of the several columns of one row (Entry.column "law,scenario,persistence"); a column of the row
#: may be missing, which the format decides.
ROW_FORMATS: dict[str, Callable[[Any], str]] = {"where": fmt_where}


def tabular(colspec: str, header: list[str], rows: list[list[str]], midrules_before=()) -> str:
    lines = ["\\begin{tabular}{" + colspec + "}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for i, row in enumerate(rows):
        if i in set(midrules_before) and i > 0:
            lines.append("\\midrule")
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def longtable(colspec: str, header: list[str], rows: list[list[str]], caption: str, label: str) -> str:
    head = " & ".join(header) + " \\\\"
    lines = ["\\begin{longtable}{" + colspec + "}",
             "\\caption{" + caption + "}\\label{" + label + "} \\\\", "\\toprule", head, "\\midrule", "\\endfirsthead",
             "\\toprule", head, "\\midrule", "\\endhead", "\\bottomrule", "\\endfoot"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines.append("\\end{longtable}")
    return "\n".join(lines) + "\n"
