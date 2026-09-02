"""Tests for split parsing, formula reading, and the N-4 leakage audit."""

from __future__ import annotations

import json

import pytest

from im2latex.data.splits import (
    Sample,
    audit,
    normalize_whitespace,
    parse_split_file,
    read_formulas,
    read_jsonl,
    summarize,
    tokenize_for_audit,
    write_jsonl,
)


def write_formula_file(tmp_path, formulas, encoding="latin-1", trailing_newline=True):
    """Write a formula list the way upstream does: one per line, Latin-1, LF endings."""
    path = tmp_path / "formulas.lst"
    text = "\n".join(formulas) + ("\n" if trailing_newline else "")
    path.write_bytes(text.encode(encoding))
    return path


def sample(index: int, formula: str, image: str | None = None) -> Sample:
    return Sample(
        image=image or f"img{index}.png",
        formula_index=index,
        formula=formula,
        render_type="basic",
    )


# --------------------------------------------------------------------- reading formulas


def test_read_formulas_reports_latin1_when_utf8_fails(tmp_path):
    path = write_formula_file(tmp_path, ["x^2", "\\alpha \xe7 \\beta"])
    formulas, encoding = read_formulas(path)
    assert encoding == "latin-1"
    assert formulas == ["x^2", "\\alpha \xe7 \\beta"]


def test_read_formulas_prefers_utf8_when_it_decodes(tmp_path):
    path = write_formula_file(tmp_path, ["x^2", "y_1"], encoding="utf-8")
    _, encoding = read_formulas(path)
    assert encoding == "utf-8"


def test_read_formulas_does_not_split_on_embedded_carriage_returns(tmp_path):
    """Regression test for a silent label-corruption bug.

    The upstream corpus contains 1,005 bare CR characters *inside* formula text. The
    split files address formulas by line number, so if a CR is treated as a line break
    the list gains phantom entries and every formula after the first one is paired with
    the wrong image. str.splitlines() does exactly that; splitting on "\\n" does not.
    """
    formulas = ["a^2", "b \r c", "d^3", "e_1"]
    path = write_formula_file(tmp_path, formulas)

    read, _ = read_formulas(path)

    assert len(read) == 4, "a carriage return inside a formula must not create a line"
    assert read[2] == "d^3", "indices after the CR must not shift"
    # Demonstrate that the naive read would have been wrong.
    assert len(path.read_bytes().decode("latin-1").splitlines()) == 5


def test_read_formulas_ignores_the_trailing_newline(tmp_path):
    path = write_formula_file(tmp_path, ["a", "b", "c"], trailing_newline=True)
    formulas, _ = read_formulas(path)
    assert formulas == ["a", "b", "c"]


def test_read_formulas_handles_a_file_with_no_trailing_newline(tmp_path):
    path = write_formula_file(tmp_path, ["a", "b"], trailing_newline=False)
    formulas, _ = read_formulas(path)
    assert formulas == ["a", "b"]


# ------------------------------------------------------------------------ normalization


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  x^2  ", "x^2"),
        ("a\tb", "a b"),
        ("a \r b", "a b"),
        ("a    b", "a b"),
        ("\\frac{1}{2}", "\\frac{1}{2}"),
    ],
)
def test_normalize_whitespace(raw, expected):
    assert normalize_whitespace(raw) == expected


def test_tokenize_for_audit_keeps_control_sequences_whole():
    assert tokenize_for_audit(r"\frac{1}{2}") == ["\\frac", "{", "1", "}", "{", "2", "}"]


def test_tokenize_for_audit_handles_escaped_literals():
    assert tokenize_for_audit(r"\{ \alpha") == ["\\{", "\\alpha"]


# -------------------------------------------------------------------------- split files


def test_parse_split_file_pairs_formulas_by_index(tmp_path):
    formulas = ["zero", "one", "two"]
    path = tmp_path / "train.lst"
    path.write_text("2 abc basic\n0 def basic\n", encoding="latin-1")

    result = parse_split_file(path, formulas)

    assert [s.formula for s in result.samples] == ["two", "zero"]
    assert [s.image for s in result.samples] == ["abc.png", "def.png"]
    assert result.malformed_lines == 0


def test_parse_split_file_normalizes_the_formula_it_pairs(tmp_path):
    path = tmp_path / "train.lst"
    path.write_text("0 abc basic\n", encoding="latin-1")
    result = parse_split_file(path, ["  a\tb  "])
    assert result.samples[0].formula == "a b"


def test_parse_split_file_counts_rather_than_raises_on_bad_lines(tmp_path):
    path = tmp_path / "train.lst"
    path.write_text(
        "0 ok basic\nnot-a-number x basic\n\n   \n1 also_ok basic\n", encoding="latin-1"
    )

    result = parse_split_file(path, ["a", "b"])

    assert len(result.samples) == 2
    assert result.malformed_lines == 1


def test_parse_split_file_counts_out_of_range_indices(tmp_path):
    path = tmp_path / "train.lst"
    path.write_text("0 ok basic\n99 missing basic\n", encoding="latin-1")

    result = parse_split_file(path, ["only-one"])

    assert len(result.samples) == 1
    assert result.out_of_range_indices == 1


def test_parse_split_file_does_not_double_the_png_extension(tmp_path):
    path = tmp_path / "train.lst"
    path.write_text("0 already.png basic\n", encoding="latin-1")
    assert parse_split_file(path, ["a"]).samples[0].image == "already.png"


# ------------------------------------------------------------------------------- jsonl


def test_jsonl_round_trips_including_non_ascii(tmp_path):
    samples = [sample(0, "\\alpha \xe7"), sample(1, "x^2")]
    path = tmp_path / "train.jsonl"

    assert write_jsonl(samples, path) == 2
    assert read_jsonl(path) == samples


def test_jsonl_is_one_object_per_line(tmp_path):
    path = tmp_path / "train.jsonl"
    write_jsonl([sample(0, "a"), sample(1, "b")], path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["formula"] == "a"


# --------------------------------------------------------------------------- statistics


def test_summarize_counts_tokens_and_unique_formulas():
    samples = [sample(0, "a b"), sample(1, "a b"), sample(2, "a b c")]
    stats = summarize("train", samples)

    assert stats.samples == 3
    assert stats.unique_formulas == 2
    assert stats.total_tokens == 2 + 2 + 3
    assert stats.max_token_length == 3
    assert stats.distinct_tokens == 3


def test_median_token_length_averages_the_middle_pair_on_an_even_count():
    """statistics.median, not the upper median: lengths 1,2,3,4 -> 2.5, not 3."""
    samples = [sample(i, " ".join("x" * (i + 1))) for i in range(4)]
    assert summarize("train", samples).median_token_length == 2.5


def test_summarize_handles_an_empty_split():
    stats = summarize("test", [])
    assert stats.samples == 0
    assert stats.mean_token_length == 0.0
    assert stats.median_token_length == 0.0
    assert stats.max_token_length == 0


# -------------------------------------------------------------------------------- audit


def test_audit_reports_no_leakage_for_disjoint_splits():
    splits = {
        "train": [sample(0, "a"), sample(1, "b")],
        "validate": [sample(2, "c")],
        "test": [sample(3, "d")],
    }
    _, leakage = audit(splits)

    assert not leakage.has_index_leakage
    assert all(count == 0 for count in leakage.shared_formula_indices.values())


def test_audit_detects_a_shared_formula_index():
    """The failure N-4 exists to prevent: the same formula trained and tested on."""
    splits = {
        "train": [sample(0, "a"), sample(7, "shared")],
        "validate": [sample(1, "b")],
        "test": [sample(7, "shared")],
    }
    _, leakage = audit(splits)

    assert leakage.has_index_leakage
    assert leakage.shared_formula_indices["train|test"] == 1
    assert leakage.shared_formula_indices["train|validate"] == 0


def test_audit_reports_duplicate_formula_text_separately_from_indices():
    """Different indices, identical LaTeX - an upstream property, not a pipeline defect."""
    splits = {
        "train": [sample(0, "x^2")],
        "validate": [sample(1, "x^2")],
        "test": [sample(2, "y")],
    }
    _, leakage = audit(splits)

    assert not leakage.has_index_leakage
    assert leakage.shared_formula_text["train|validate"] == 1


def test_audit_detects_a_shared_image_that_indices_and_text_both_miss():
    """The contamination the canonical splits actually contain.

    Two different formula indices, LaTeX differing only by a leading '%', but the same
    rendered PNG - so index and text overlap are both zero while the model is
    nonetheless trained on an image it is later tested on.
    """
    splits = {
        "train": [sample(10, "\\frac{a}{b}", image="dup.png")],
        "validate": [sample(11, "z")],
        "test": [sample(12, "%\\frac{a}{b}", image="dup.png")],
    }
    _, leakage = audit(splits)

    assert leakage.shared_formula_indices["train|test"] == 0
    assert leakage.shared_formula_text["train|test"] == 0
    assert leakage.shared_images["train|test"] == 1
    assert leakage.has_image_leakage
    assert not leakage.has_index_leakage


def test_audit_reports_no_image_leakage_when_images_are_distinct():
    splits = {
        "train": [sample(0, "a", image="one.png")],
        "validate": [sample(1, "b", image="two.png")],
        "test": [sample(2, "c", image="three.png")],
    }
    _, leakage = audit(splits)
    assert not leakage.has_image_leakage


def test_audit_lists_tokens_that_train_never_saw():
    splits = {
        "train": [sample(0, "a b")],
        "validate": [sample(1, "a z")],
        "test": [sample(2, "a")],
    }
    _, leakage = audit(splits)

    assert leakage.tokens_unseen_in_train["validate"] == ["z"]
    assert leakage.tokens_unseen_in_train["test"] == []
    assert "train" not in leakage.tokens_unseen_in_train
