"""
Unit tests for transcript_engine.identity.self_identification.

Pure text-pattern tests — no audio, no embeddings. These check the
self-introduction vs. name-mention distinction the mission brief calls out
as the most important failure mode to avoid (section 8): "Neel will join
tomorrow" must never be read as Neel introducing himself.
"""

from __future__ import annotations

from transcript_engine.identity.self_identification import extract_self_identifications


def test_im_x_is_high_confidence():
    found = extract_self_identifications("Hi everyone, I'm Neel.")
    assert len(found) == 1
    assert found[0].name == "Neel"
    assert found[0].confidence == "high"


def test_my_name_is_x_is_high_confidence():
    found = extract_self_identifications("Hello, my name is Sarah, I'll be leading this.")
    assert len(found) == 1
    assert found[0].name == "Sarah"
    assert found[0].confidence == "high"


def test_this_is_x_is_medium_confidence():
    found = extract_self_identifications("This is Mahar, thanks for having me.")
    assert len(found) == 1
    assert found[0].name == "Mahar"
    assert found[0].confidence == "medium"


def test_x_here_is_medium_confidence():
    found = extract_self_identifications("John here, can everyone hear me?")
    assert len(found) == 1
    assert found[0].name == "John"
    assert found[0].confidence == "medium"


def test_third_person_future_tense_is_not_self_identification():
    # The exact false-positive the mission brief warns about: a name mention
    # is not a self-introduction just because the sentence is about someone.
    found = extract_self_identifications("Neel will join tomorrow.")
    assert found == []


def test_this_is_the_report_is_not_a_name():
    found = extract_self_identifications("This is the report we discussed.")
    assert found == []


def test_this_is_possessive_is_excluded():
    found = extract_self_identifications("This is Neel's report, not mine.")
    assert found == []


def test_no_identification_in_ordinary_speech():
    found = extract_self_identifications("So the numbers for Q3 look good overall.")
    assert found == []


def test_multiple_identifications_in_one_segment_are_all_returned():
    found = extract_self_identifications("I'm Neel. Also, this is Mahar joining late.")
    names = {f.name for f in found}
    assert names == {"Neel", "Mahar"}


def test_day_name_after_this_is_not_treated_as_a_person():
    found = extract_self_identifications("This is Monday's agenda.")
    assert found == []
