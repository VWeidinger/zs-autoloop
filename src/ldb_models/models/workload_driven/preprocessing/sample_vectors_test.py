"""Tests for the SQL ``LIKE``/``NOT LIKE`` -> regex translation in
``construct_filter_sample``.

The translation used to escape only ``.``, ``(``, ``)`` and ``?`` and left every
other regex metacharacter (``*``, ``+``, ``[`` ...) untouched, so a literal such
as ``%*`` produced the invalid pattern ``^.*?*$`` and ``re.compile`` raised
``re.error: multiple repeat``. The fix escapes the whole literal with
``re.escape`` before translating the SQL wildcards.

These tests pin two things:
  * for every literal the *old* code handled, the new output is identical, and
  * literals that used to crash now match with correct (literal) semantics.
"""

import re

import pandas as pd
import pytest

from ldb_models.models.workload_driven.preprocessing import sample_vectors


# --- reference implementation of the pre-fix behaviour -----------------------


def _old_like_regex(literal: str) -> str:
    """Reproduce exactly the LIKE->regex translation from before the fix."""
    regex = literal
    for esc_char in [".", "(", ")", "?"]:
        regex = regex.replace(esc_char, "\\" + esc_char)
    regex = regex.replace("%", ".*?")
    return f"^{regex}$"


def _old_like_series(values, literal: str) -> list[bool]:
    regex = _old_like_regex(literal)
    return (
        pd.Series(values)
        .astype(str)
        .str.contains(regex, regex=True, na=False)
        .astype(bool)
        .tolist()
    )


# --- helpers -----------------------------------------------------------------


def _make_inputs(values, literal, operator="LIKE", datatype="character varying"):
    table, col = "t", "c"
    table_samples = {table: pd.DataFrame({col: values})}
    column_stats = [{"attname": col, "tablename": table, "data_type": datatype}]
    filter_column = {"operator": operator, "col_name": col, "literal": literal}
    return table_samples, column_stats, filter_column


def _new_result(values, literal, operator="LIKE") -> list[bool]:
    table_samples, column_stats, filter_column = _make_inputs(
        values, literal, operator
    )
    return sample_vectors.construct_filter_sample(
        table_samples, column_stats, filter_column
    ).tolist()


# A value corpus that exercises the escaped characters and wildcard positions.
VALUES = [
    "abc",
    "aXc",
    "a.c",
    "a(b)c",
    "a?c",
    "xabcy",
    "abcdef",
    "",
    "ABC",
    "Fe2O3",
    "O2",
    "a%c",
    "a*b",
    "a+b",
]

# Literals composed only of characters the old translation handled correctly
# (alphanumerics, space, "%", ".", "(", ")", "?"). For these the new and old
# behaviour must be byte-for-byte identical.
SAFE_LITERALS = [
    "abc",
    "a%c",
    "%abc",
    "abc%",
    "%abc%",
    "a.c",
    "a(b)c",
    "a?c",
    "%",
    "",
    "Fe%",
    "%O2%",
    "%b%c%",
]


# --- behaviour-identical guarantee ------------------------------------------


@pytest.mark.parametrize("literal", SAFE_LITERALS)
def test_like_identical_to_old_for_supported_literals(literal):
    assert _new_result(VALUES, literal, "LIKE") == _old_like_series(VALUES, literal)


@pytest.mark.parametrize("literal", SAFE_LITERALS)
def test_not_like_identical_to_old_for_supported_literals(literal):
    old_not_like = [not x for x in _old_like_series(VALUES, literal)]
    assert _new_result(VALUES, literal, "NOT LIKE") == old_not_like


# --- regression: literals that used to raise re.error ------------------------

# (literal, value_that_matches, value_that_does_not_match)
CRASHING_LITERALS = [
    ("a**b", "a**b", "axxb"),
    ("**", "**", "*"),
    ("+abc", "+abc", "abc"),
    ("a[b", "a[b", "ab"),
    ("%*", "xyz*", "xyz"),  # the exact shape from the original traceback
]


@pytest.mark.parametrize("literal", [c[0] for c in CRASHING_LITERALS])
def test_old_translation_was_broken(literal):
    """Guard: these literals really did make the old code raise re.error."""
    with pytest.raises(re.error):
        re.compile(_old_like_regex(literal))


@pytest.mark.parametrize("literal, hit, miss", CRASHING_LITERALS)
def test_previously_crashing_literals_now_match_literally(literal, hit, miss):
    result = _new_result([hit, miss], literal, "LIKE")
    assert result == [True, False]


# --- the newly supported "_" single-char wildcard ---------------------------


def test_underscore_is_single_char_wildcard():
    # "a_c" matches a single character between a and c ...
    assert _new_result(["abc", "aXc"], "a_c", "LIKE") == [True, True]
    # ... but not zero or two characters.
    assert _new_result(["ac", "abbc"], "a_c", "LIKE") == [False, False]
