import pytest

from postags import (  # pylint: disable=unused-import
    ACCENTEDFORM,
    ACTIVE,
    CASE,
    FEMININE,
    FIRST_PERSON,
    GENDER,
    GERUND,
    GERUNDIVE,
    INDICATIVE,
    LEMMA,
    MASCULINE,
    MOOD,
    NEUTER,
    NOMINATIVE,
    NUMBER,
    PART_OF_SPEECH,
    PARTICIPLE,
    PASSIVE,
    PERSON,
    PLURAL,
    PRESENT,
    SINGULAR,
    TENSE,
    VERB,
    VOCATIVE,
    VOICE,
    morpheus_to_parses,
    tag_distance,
)


def test_morpheus_to_parses_handles_off_by_one_error_for_last_code():
    """
    Verifies that the parser processes all morphological codes, including the last one.

    The original code contained an off-by-one error in its processing loop, which
    this test is designed to catch. It asserts that a feature code at the end of
    the input string (e.g., 'pl' for plural) is correctly identified.
    """
    # Morpheus output for "amantur" (they are loved)
    # V,ama,1 V pres ind pass 3rd pl
    wordform = "amantur"
    nl = "V amantur,amare pres ind pass 3rd pl"
    parses = morpheus_to_parses(wordform, nl)

    # Expected parse
    expected = {
        LEMMA: "amare",
        ACCENTEDFORM: "amantur",
        PART_OF_SPEECH: VERB,
        TENSE: PRESENT,
        MOOD: INDICATIVE,
        VOICE: PASSIVE,
        PERSON: "3rd",
        NUMBER: PLURAL,
    }

    assert len(parses) == 1
    # Check that the single parse matches the expected dictionary
    assert parses[0] == expected


def test_morpheus_to_parses_participle_with_multiple_genders_parses_correctly():
    """
    Verifies that a participle with multiple possible genders (masc/fem/neut)
    parses correctly, leveraging the special 'P' prefix logic.

    This test also serves as a regression test to ensure the final 'participle'
    code, though ignored by the buggy loop, does not break the parse because
    the mood is correctly set beforehand.
    """
    wordform = "amans"
    nl = "P amans,amare pres act masc/fem/neut sg nom participle"
    parses = morpheus_to_parses(wordform, nl)

    # The code generates multiple parses for gender, let's check the masculine one.
    expected_masc_parse = {
        LEMMA: "amare",
        ACCENTEDFORM: "ama_ns",
        PART_OF_SPEECH: VERB,
        MOOD: PARTICIPLE,  # Correctly set from 'P' prefix
        TENSE: PRESENT,  # Correctly set from 'pres'
        VOICE: ACTIVE,
        GENDER: MASCULINE,
        NUMBER: SINGULAR,
        CASE: NOMINATIVE,
    }

    # We expect 3 parses (masc, fem, neut) because of the final loop
    assert len(parses) == 3
    # Check if our expected masculine parse exists in the results
    found_match = any(p == expected_masc_parse for p in parses)
    assert (
        found_match
    ), f"The expected masculine parse for 'amans' was not found in the results.\nGot: {parses}"


def test_morpheus_to_parses_genderless_noun_expands_to_three_genders():
    """
    Tests that a noun parse without a gender but with a case
    is correctly expanded into three separate parses (masc, fem, neut).
    This tests the `elif parse.get(GENDER, "") == ""` block.
    """
    # A fake Morpheus output for a 3rd declension adjective/noun like 'felix'
    # which could be any gender. We simulate a parse that is missing a gender.
    # N,felix,icis adj3 nom sg
    wordform = "felix"
    nl = "N felix,felicis nom sg adj3"

    parses = morpheus_to_parses(wordform, nl)

    # Assert that the code correctly generated 3 distinct parses
    assert len(parses) == 3, "Should have expanded into 3 gendered parses"

    # Extract the genders from the resulting parses
    genders_found = {p.get(GENDER) for p in parses}

    # Assert that we have exactly one of each gender
    assert genders_found == {MASCULINE, FEMININE, NEUTER}

    # Also assert that the other features are identical
    base_parse = parses[0].copy()
    del base_parse[GENDER]
    for p in parses[1:]:
        current_parse = p.copy()
        del current_parse[GENDER]
        assert (
            current_parse == base_parse
        ), "All expanded parses should be identical except for their gender"


def test_morpheus_to_parses_neuter_gerundive_generates_additional_gerund_parse():
    """
    Tests that a singular, neuter, non-nominative gerundive parse
    correctly generates a second, alternative parse as a gerund.
    This tests the `if parse.get(MOOD, "") == GERUNDIVE` block.
    """
    # Morpheus output for "amandum"
    wordform = "amandum"
    # We simplify the morpheus input to just trigger the logic
    nl = "<NL>V a^mandum,amo  gerundive neut acc sg                     conj1,are_vb"

    parses = morpheus_to_parses(wordform, nl)

    # We expect exactly two resulting parses
    assert len(parses) == 2, "Should have created an additional gerund parse"

    # Find the gerundive and gerund parses in the output
    gerundive_parse = next((p for p in parses if p.get(MOOD) == GERUNDIVE), None)
    gerund_parse = next((p for p in parses if p.get(MOOD) == GERUND), None)

    # Check that both were found
    assert gerundive_parse is not None, "Original gerundive parse is missing"
    assert gerund_parse is not None, "Alternative gerund parse was not created"

    # Check that they are identical except for the MOOD
    gerundive_copy = gerundive_parse.copy()
    gerund_copy = gerund_parse.copy()

    del gerundive_copy[MOOD]
    del gerund_copy[MOOD]

    assert gerundive_copy == gerund_copy, "Parses should be identical besides mood"


def test_morpheus_to_parses_standard_parse_is_unchanged_by_final_loop():
    """
    Tests that a standard parse that does not meet the special conditions
    passes through the final loop unchanged. This is a negative test case.
    """
    # A standard, unambiguous verb form: "amo"
    wordform = "amo"
    nl = "V amo,amare 1st sg pres ind act"

    parses = morpheus_to_parses(wordform, nl)

    # We expect exactly one parse, unmodified
    assert len(parses) == 1

    expected = {
        LEMMA: "amare",
        ACCENTEDFORM: "amo",
        PART_OF_SPEECH: VERB,
        PERSON: FIRST_PERSON,
        NUMBER: SINGULAR,
        TENSE: PRESENT,
        MOOD: INDICATIVE,
        VOICE: ACTIVE,
    }
    assert parses[0] == expected


def test_morpheus_to_parses_handles_multiple_slash_codes_multiplicatively():
    """
    Tests the first loop's ability to handle multiple feature codes with slashes.

    When multiple codes contain slashes (e.g., 'masc/fem' and 'nom/acc'), the
    number of generated parses should be the product of the number of options in each.
    This test ensures the combinatorial expansion is working correctly.
    (2 genders * 2 cases = 4 expected parses).
    """
    wordform = "amati"
    # P,amatus,a,um perf pass masc/fem nom/acc pl participle
    nl = "P amati,amare perf pass masc/fem/neut nom/voc pl participle"
    parses = morpheus_to_parses(wordform, nl)

    # 3 genders * 2 cases = 6 parses
    assert len(parses) == 6, "Expected parses to multiply (3 genders * 2 cases = 6)"

    # Create a set of tuples representing the expected combinations
    expected_combinations = {
        (MASCULINE, NOMINATIVE),
        (MASCULINE, VOCATIVE),
        (FEMININE, NOMINATIVE),
        (FEMININE, VOCATIVE),
        (NEUTER, NOMINATIVE),
        (NEUTER, VOCATIVE),
    }

    # Extract the actual combinations from the results
    actual_combinations = {(p.get(GENDER), p.get(CASE)) for p in parses}

    assert (
        actual_combinations == expected_combinations
    ), "Did not find all expected gender/case combinations"


def test_morpheus_to_parses_ignores_unmapped_feature_codes():
    """
    Verifies that a morphological code not present in `featMap` is ignored
    without causing an error or altering the parse.

    The `setfeature` function is designed to silently pass on unknown codes. This
    test confirms that behavior, ensuring the parser is robust against unexpected
    or malformed input from Morpheus.
    """
    wordform = "amo"
    # A standard parse with a junk code at the end
    nl = "V amo,amare 1st sg pres ind act some_unknown_feature"
    parses = morpheus_to_parses(wordform, nl)

    assert len(parses) == 1, "An unknown feature should not create extra parses"

    expected = {
        LEMMA: "amare",
        ACCENTEDFORM: "amo",
        PART_OF_SPEECH: VERB,
        PERSON: "1st",
        NUMBER: SINGULAR,
        TENSE: PRESENT,
        MOOD: INDICATIVE,
        VOICE: ACTIVE,
    }
    assert (
        parses[0] == expected
    ), "The final parse should not be affected by the unknown code"


def test_morpheus_to_parses_handles_conflicting_features_gracefully():
    """
    Tests how the parser handles contradictory information in the input string.

    The `setfeature` function will not overwrite an already-set feature unless
    the `overwrite` flag is True (which it isn't in this loop). This test
    verifies that the first-encountered feature wins, preventing a later,
    conflicting code from corrupting the parse.
    """
    wordform = "est"
    # A 3rd person form with a conflicting '1st' person code at the end.
    nl = "V est,esse 3rd sg pres ind act 1st"
    parses = morpheus_to_parses(wordform, nl)

    assert len(parses) == 1
    # '3rd' is set first and should not be overwritten by '1st'
    assert parses[0].get(PERSON) == "3rd", "The first person value set should be kept"


def test_morpheus_to_parses_genderless_noun_without_case_is_not_expanded():
    """
    This is a negative test for the final loop's gender expansion logic.

    The logic to expand a genderless noun into M/F/N parses should only trigger
    if a case is also present (`parse.get(CASE, "") != ""`). This test ensures that
    a genderless parse *without* a case passes through the loop unmodified.
    """
    # A fake Morpheus output for a noun where case is not specified.
    wordform = "civis"
    nl = "N civis,civis sg adj3"  # Note: 'nom' or other case is missing
    parses = morpheus_to_parses(wordform, nl)

    assert len(parses) == 1, "Should not expand to multiple genders without a case"
    assert parses[0].get(GENDER) is None, "Gender should remain unspecified"


def test_morpheus_to_parses_nominative_gerundive_is_not_expanded_to_gerund():
    """
    This is a negative test for the final loop's gerund/gerundive logic.

    The logic to create an alternative gerund parse from a gerundive should only
    trigger if the gerundive is *not* in the nominative case. This test verifies
    that a nominative gerundive passes through the loop unmodified.
    """
    wordform = "amandum"
    # A gerundive in the nominative case
    nl = "V amandum,amare gerundive neut nom sg"
    parses = morpheus_to_parses(wordform, nl)

    assert len(parses) == 1, "Nominative gerundive should not be expanded"
    assert parses[0].get(MOOD) == GERUNDIVE, "Parse should remain a gerundive"


def test_morpheus_to_parses_interaction_of_slash_codes_and_final_loop_expansion():
    """
    Tests the complex interaction between the first and second loops.

    This test uses an input with a slash-separated gender ('masc/neut') where one
    of the resulting parses ('neut') meets the criteria for the final loop's
    gerundive-to-gerund expansion.

    - The first loop should create two initial parses (masc, neut).
    - The second loop should process both. The neuter parse should be expanded
      into an additional gerund parse.
    - Expected final output: 3 parses (Masc-Gerundive, Neut-Gerundive, Neut-Gerund).
    """
    wordform = "amandi"
    nl = "V amandi,amare gerundive masc/neut gen sg"
    parses = morpheus_to_parses(wordform, nl)

    assert len(parses) == 3, "Expected 1 initial split + 1 final expansion = 3 parses"

    # Check for the masculine gerundive
    assert any(
        p.get(GENDER) == MASCULINE and p.get(MOOD) == GERUNDIVE for p in parses
    ), "Missing masculine gerundive parse"

    # Check for the neuter gerundive
    assert any(
        p.get(GENDER) == NEUTER and p.get(MOOD) == GERUNDIVE for p in parses
    ), "Missing original neuter gerundive parse"

    # Check for the newly created neuter gerund
    assert any(
        p.get(GENDER) == NEUTER and p.get(MOOD) == GERUND for p in parses
    ), "Missing the expanded neuter gerund parse"


def test_tag_distance_accepts_valid_9_char_tags():
    """
    Verifies the function computes distance for valid 9-char tags
    without raising an error.
    """
    tag1 = "v1spia---"
    tag2 = "v3spia---"
    assert tag_distance(tag1, tag2) == 1


def test_tag_distance_accepts_valid_12_char_tags():
    """
    Verifies the function computes distance for valid 12-char tags
    without raising an error.
    """
    tag1 = "V--piap-s---"
    tag2 = "V--piap-p---"
    assert tag_distance(tag1, tag2) == 1


def test_tag_distance_rejects_mismatched_length_tags():
    """
    Verifies the function correctly raises a ValueError for tags of
    different, albeit valid, lengths.
    """
    tag_9_char = "v1spia---"
    tag_12_char = "V--piap-s---"
    with pytest.raises(ValueError, match="Mismatched or invalid tag lengths"):
        tag_distance(tag_9_char, tag_12_char)


def test_tag_distance_rejects_invalid_length_tags():
    """
    Verifies the function correctly raises a ValueError for tags of a length
    other than 9 or 12.
    """
    short_tag1 = "short"
    short_tag2 = "short"
    with pytest.raises(ValueError, match="Mismatched or invalid tag lengths"):
        tag_distance(short_tag1, short_tag2)
