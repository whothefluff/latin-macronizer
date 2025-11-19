import os
import re
import sqlite3
import subprocess
import sys
import types
from collections import defaultdict

import pytest


@pytest.fixture
def stub_modules():

    def dummy_morpheus_parser(wordform, _nl):
        """Return a non-empty list that mimics a successful parse.
        (The contents don't matter, only that it's not empty)
        """
        return [[f"{wordform}-lemma", f"{wordform}-accented"]]

    # Minimal stubs for dependencies so we can import macronizer
    postags = types.ModuleType("postags")
    postags.LEMMA = 0
    postags.ACCENTEDFORM = 1
    postags.removemacrons = lambda s: s
    postags.unicodeaccents = lambda s: s
    postags.tag_distance = lambda a, b: 0
    postags.parse_to_ldt = lambda p: "TAG"
    postags.morpheus_to_parses = dummy_morpheus_parser
    sys.modules["postags"] = postags

    lemmas = types.ModuleType("lemmas")
    lemmas.lemma_frequency = {}
    lemmas.word_lemma_freq = {}
    lemmas.wordform_to_corpus_lemmas = {}
    sys.modules["lemmas"] = lemmas

    mac_end = types.ModuleType("macronized_endings")
    mac_end.tag_to_endings = {}
    sys.modules["macronized_endings"] = mac_end

    yield
    for name in ("postags", "lemmas", "macronized_endings"):
        sys.modules.pop(name, None)


@pytest.fixture(name="macronizer")
def macronizer_fixture(
    # pylint: disable=redefined-outer-name, unused-argument
    stub_modules,
    monkeypatch,
    tmp_path,
):

    import importlib  # pylint: disable=import-outside-toplevel

    import macronizer as mod  # pylint: disable=import-outside-toplevel

    importlib.reload(mod)

    macrons_txt = tmp_path / "macrons.txt"
    macrons_txt.write_text("", encoding="utf-8")
    monkeypatch.setattr(mod, "MACRONS_FILE", str(macrons_txt))

    return mod


@pytest.fixture(name="create_config_ini")
def create_config_ini_fixture(tmp_path):
    """
    A fixture that creates a temporary config.ini file and returns its path.
    """

    def _create(content: str):
        config_file = tmp_path / "test_config.ini"
        config_file.write_text(content, encoding="utf-8")
        return str(config_file)

    return _create


@pytest.fixture(name="db_conn")
def db_conn_fixture():
    """Provides a fresh, in-memory sqlite3 database connection for each test."""

    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture(name="functional_macronizer")
def functional_macronizer_fixture(monkeypatch, tmp_path):
    """
    A dedicated, self-contained fixture for testing structured output.

    It creates functional stubs for the 'postags' module to ensure that
    methods like removemacrons and unicodeaccents behave correctly for the
    tests that depend on them.
    """

    # --- Part 1: Create functional stubs ---
    def _remove_macrons(text):
        macron_map = str.maketrans("āēīōūȳĀĒĪŌŪȲ_", "aeiouyaeiouy ")
        return text.translate(macron_map).replace(" ", "")

    def _unicodeaccents(text):
        text = re.sub("a_", "ā", text)
        text = re.sub("e_", "ē", text)
        text = re.sub("i_", "ī", text)
        text = re.sub("o_", "ō", text)
        text = re.sub("u_", "ū", text)
        text = re.sub("y_", "ȳ", text)
        text = re.sub("A_", "Ā", text)
        text = re.sub("E_", "Ē", text)
        text = re.sub("I_", "Ī", text)
        text = re.sub("O_", "Ō", text)
        text = re.sub("U_", "Ū", text)
        text = re.sub("Y_", "Ȳ", text)
        return text

    postags_stub = types.ModuleType("postags")
    postags_stub.removemacrons = _remove_macrons
    postags_stub.unicodeaccents = _unicodeaccents
    # Add other required attributes so the module can load
    postags_stub.LEMMA = 0
    postags_stub.ACCENTEDFORM = 1
    postags_stub.tag_distance = lambda a, b: 0
    postags_stub.parse_to_ldt = lambda p: "TAG"
    postags_stub.morpheus_to_parses = lambda w, n: [[f"{w}-l", f"{w}-a"]]
    sys.modules["postags"] = postags_stub

    # Create minimal stubs for other dependencies
    lemmas_stub = types.ModuleType("lemmas")
    lemmas_stub.lemma_frequency = {}
    lemmas_stub.word_lemma_freq = {}
    lemmas_stub.wordform_to_corpus_lemmas = {}
    sys.modules["lemmas"] = lemmas_stub

    mac_end_stub = types.ModuleType("macronized_endings")
    mac_end_stub.tag_to_endings = {}
    sys.modules["macronized_endings"] = mac_end_stub

    # --- Part 2: Import and prepare the macronizer module ---
    import importlib  # pylint: disable=import-outside-toplevel

    import macronizer as mod  # pylint: disable=import-outside-toplevel

    importlib.reload(mod)  # Ensure a fresh import with our stubs

    macrons_txt = tmp_path / "macrons.txt"
    macrons_txt.write_text("", encoding="utf-8")
    monkeypatch.setattr(mod, "MACRONS_FILE", str(macrons_txt))

    yield mod  # Provide the freshly-prepared module to the tests

    # --- Part 3: Teardown ---
    for name in ("postags", "lemmas", "macronized_endings"):
        sys.modules.pop(name, None)


def test_run_external_maps_filenotfound(macronizer, monkeypatch):

    def raise_fnf(*_a, **_k):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(subprocess, "run", raise_fnf)

    with pytest.raises(macronizer.ExternalDependencyError) as ei:
        macronizer.run_external(["no-such-binary"], tool_name="toolX")
    assert "toolX" in str(ei.value)


def test_run_external_maps_timeout(macronizer, monkeypatch):

    def raise_timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="cmd", timeout=1)

    monkeypatch.setattr(subprocess, "run", raise_timeout)

    with pytest.raises(macronizer.ExternalDependencyError) as ei:
        macronizer.run_external(["sleep", "999"], tool_name="toolY", timeout=0.01)
    assert "timed out" in str(ei.value)


def test_run_external_maps_calledprocesserror_and_includes_stderr(
    macronizer, monkeypatch
):
    def raise_cpe(*a, **k):
        raise subprocess.CalledProcessError(7, "cmd", stderr=b"boom")

    monkeypatch.setattr(subprocess, "run", raise_cpe)

    with pytest.raises(macronizer.ExternalDependencyError) as ei:
        macronizer.run_external(["false"], tool_name="toolZ")
    msg = str(ei.value)
    assert "exit 7" in msg and "boom" in msg


def test_run_external_success(macronizer, monkeypatch):
    class OK:
        returncode = 0

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: OK())
    assert macronizer.run_external(["true"]) is not None


def test_crunchwords_raises_when_cruncher_missing(
    macronizer, tmp_path, monkeypatch, db_conn
):
    wl = macronizer.Wordlist(db_conn)
    wl.reinitializedatabase()

    # Point MORPHEUS_DIR to a temp dir WITHOUT cruncher
    monkeypatch.setattr(macronizer, "MORPHEUS_DIR", str(tmp_path))

    with pytest.raises(macronizer.ExternalDependencyError) as ei:
        wl.crunchwords({"abc"})
    assert "cruncher not found" in str(ei.value)


def test_crunchwords_inserts_unknown_when_no_output_and_cleans_tempfiles(
    macronizer, tmp_path, monkeypatch, db_conn
):
    wl = macronizer.Wordlist(db_conn)
    wl.reinitializedatabase()

    # Provide executable cruncher so path check passes
    cruncher = tmp_path / "bin" / "cruncher"
    cruncher.parent.mkdir(parents=True, exist_ok=True)
    cruncher.write_text("", encoding="utf-8")
    os.chmod(cruncher, 0o755)
    monkeypatch.setattr(macronizer, "MORPHEUS_DIR", str(tmp_path))

    # Track temp files created via NamedTemporaryFile used inside module
    created = []
    orig_ntf = macronizer.NamedTemporaryFile

    def tracking_ntf(*a, **k):
        f = orig_ntf(*a, **k)
        created.append(f.name)
        return f

    monkeypatch.setattr(macronizer, "NamedTemporaryFile", tracking_ntf)

    # Mock external run: write nothing to output file (empty morpheus output)
    def fake_run(*_args, **_kwargs):
        return None

    monkeypatch.setattr(macronizer, "run_external", fake_run)

    wl.crunchwords({"sineparse"})

    # DB has unknown row (wordform present, others NULL)
    wl.dbcursor.execute(
        "SELECT wordform, morphtag, lemma, accented FROM morpheus WHERE wordform=?",
        ("sineparse",),
    )
    rows = wl.dbcursor.fetchall()
    assert rows and all(r[1:] == (None, None, None) for r in rows)

    # Temp files removed in finally
    for name in created:
        assert not os.path.exists(name)


def test_crunchwords_sets_morphlib_env(macronizer, tmp_path, monkeypatch, db_conn):
    wl = macronizer.Wordlist(db_conn)
    wl.reinitializedatabase()

    # Provide executable cruncher
    cruncher = tmp_path / "bin" / "cruncher"
    cruncher.parent.mkdir(parents=True, exist_ok=True)
    cruncher.write_text("", encoding="utf-8")
    os.chmod(cruncher, 0o755)
    monkeypatch.setattr(macronizer, "MORPHEUS_DIR", str(tmp_path))

    observed_env = {}

    # Write minimal well-formed morpheus output (one word line + one parse line)
    def fake_run(*_args, **kwargs):
        observed_env.update(kwargs.get("env") or {})
        kwargs["stdout"].write(b"sine\n<NL></NL>\n")
        kwargs["stdout"].flush()

    monkeypatch.setattr(macronizer, "run_external", fake_run)

    wl.crunchwords({"sine"})
    assert observed_env.get("MORPHLIB") == str(tmp_path / "stemlib")


def test_addtags_raises_when_rft_annotate_missing(macronizer):
    t = macronizer.Tokenization("arma virumque cano")
    non_existent_dir = "/definitely/missing"

    with pytest.raises(macronizer.ExternalDependencyError) as exc_info:
        t.addtags(rftagger_dir=non_existent_dir)
    expected_path_in_error = os.path.join(non_existent_dir, "rft-annotate")
    assert "not found or not executable" in str(exc_info.value)
    assert expected_path_in_error in str(exc_info.value)


def test_addtags_reads_output_from_external_using_tempfiles(
    macronizer, tmp_path, mocker
):
    # Single word tokenization; no sentence-end, no enclitics
    t = macronizer.Tokenization("arma")

    # Create a dummy rftagger directory and a fake executable inside it
    dummy_rftagger_dir = tmp_path / "rftagger"
    dummy_rftagger_dir.mkdir()
    rft_annotate_path = dummy_rftagger_dir / "rft-annotate"
    rft_annotate_path.touch(mode=0o755)  # Mark as executable

    # Mock the external command runner to fake the tool's behavior
    def fake_run(cmd, **_kwargs):
        # The command should include the full path to our fake executable
        assert cmd[0] == str(rft_annotate_path)
        out_path = cmd[-1]
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("arma\tNOUN\n")

    mocker.patch("macronizer.run_external", side_effect=fake_run)

    t.addtags(rftagger_dir=str(dummy_rftagger_dir))
    tok = next(token for token in t.tokens if token.isword)
    assert tok.tag == "NOUN"


class TestTokenMacronize:
    """
    Tests for the Token.macronize method.
    """

    def test_does_not_crash_on_unknown_word_with_empty_accented_form(self, macronizer):
        """
        Scenario: The macronizer has no information for the word "ignotus", so its
        accented form is an empty string.

        The code should not crash. It should initialize `i` and `j`
        explicitly and gracefully handle the empty accented form, returning the
        original plain text.
        """
        # arrange: A token for a word with no known macronization
        token = macronizer.Token("ignotus")
        token.accented = [""]  # The list of possible accentuations is empty

        # act: Run the macronize function. We use `performutov` to bypass
        # an early exit
        token.macronize(
            domacronize=True, alsomaius=False, performutov=True, performitoj=False
        )

        # assert: The code returned the original word as expected (no crash)
        assert token.macronized == "ignotus"

    def test_skeleton_check_bails_out_on_mismatched_words(self, macronizer):
        """
        GIVEN a token and an accented form that are fundamentally different words,
        WHEN macronize is called,
        THEN it should bail out and return the original plain text.
        """
        # Arrange
        token = macronizer.Token("amica")
        token.accented = ["ami_cus"]  # Mismatched skeleton

        # Act
        token.macronize(
            domacronize=True, alsomaius=False, performutov=False, performitoj=False
        )

        # Assert
        assert token.macronized == "amica"

    def test_skeleton_check_allows_ij_orthographic_variants(self, macronizer):
        """
        GIVEN a token with 'I' and an accented form with 'j',
        WHEN macronize is called,
        THEN it should NOT bail out and should perform the alignment correctly.
        """
        # Arrange
        token = macronizer.Token("Iulius")
        token.accented = ["ju_lius"]  # Skeleton matches after normalization

        # Act
        token.macronize(
            domacronize=True, alsomaius=False, performutov=False, performitoj=True
        )

        # Assert
        assert token.macronized == "Ju_lius"

    def test_skeleton_check_allows_uv_orthographic_variants(self, macronizer):
        """
        GIVEN a token with 'u' and an accented form with 'v',
        WHEN macronize is called,
        THEN it should NOT bail out and should perform the alignment correctly.
        """
        # Arrange
        token = macronizer.Token("uoluit")
        token.accented = ["vo_lvit"]  # Skeleton matches after normalization

        # Act
        token.macronize(
            domacronize=True, alsomaius=False, performutov=True, performitoj=False
        )

        # Assert
        assert token.macronized == "vo_lvit"

    def test_handles_trailing_macron_correctly(self, macronizer):
        """
        GIVEN an accented form with a macron at the very end,
        WHEN macronize is called,
        THEN the final macronized string should include that trailing macron.
        (This was the primary bug in the original implementation).
        """
        # Arrange
        token = macronizer.Token("porta")
        token.accented = ["porta_"]

        # Act
        token.macronize(
            domacronize=True, alsomaius=False, performutov=False, performitoj=False
        )

        # Assert
        assert token.macronized == "porta_"

    def test_handles_leading_macron_correctly(self, macronizer):
        """
        This test is synthetic and only checks for correctness of the alignment logic.

        GIVEN an accented form with a leading macron,
        WHEN macronize is called,
        THEN the resulting string should correctly include the leading macron.
        """
        # Arrange
        token = macronizer.Token("test")
        token.accented = ["_test"]

        # Act
        token.macronize(
            domacronize=True, alsomaius=False, performutov=False, performitoj=False
        )

        # Assert
        assert token.macronized == "_test"

    def test_handles_and_cleans_up_multiple_trailing_macrons(self, macronizer):
        """
        GIVEN a malformed accented string with multiple trailing macrons,
        WHEN macronize is called,
        THEN it should align them and correctly apply the __ -> _ cleanup rule.
        """
        # Arrange
        token = macronizer.Token("causa")
        token.accented = ["ca_usa__"]

        # Act
        token.macronize(
            domacronize=True, alsomaius=False, performutov=False, performitoj=False
        )

        # Assert
        assert token.macronized == "ca_usa_"

    def test_domacronize_false_still_performs_uv_orthography_changes(self, macronizer):
        """
        GIVEN domacronize=False but performutov=True,
        WHEN macronize is called on a word with a 'u'/'v' difference,
        THEN it should perform the u->v change but not add the macron.
        """
        # Arrange
        token = macronizer.Token("uoluit")
        token.accented = ["vo_lvit"]

        # Act
        token.macronize(
            domacronize=False, alsomaius=False, performutov=True, performitoj=False
        )

        # Assert
        assert token.macronized == "volvit"

    def test_domacronize_false_still_performs_ij_orthography_changes(self, macronizer):
        """
        GIVEN domacronize=False but performitoj=True,
        WHEN macronize is called on a word with a 'i'/'j' difference,
        THEN it should perform the i->j change but not add the macron.
        """
        # Arrange
        token = macronizer.Token("eius")
        token.accented = ["e_jus"]

        # Act
        token.macronize(
            domacronize=False, alsomaius=False, performutov=False, performitoj=True
        )

        # Assert
        assert token.macronized == "ejus"

    def test_alsomaius_flag_adds_macron_before_consonantal_j(self, macronizer):
        """
        GIVEN the alsomaius flag is True,
        WHEN macronize is called on a word like 'eius',
        THEN it should add a macron on the vowel preceding the 'j'.
        """
        # Arrange
        token = macronizer.Token("eius")
        token.accented = ["ejus"]  # Accented form without the macron

        # Act
        token.macronize(
            domacronize=True, alsomaius=True, performutov=False, performitoj=True
        )

        # Assert
        # The logic first changes 'ejus' to 'e_jus', then aligns 'eius' to it.
        assert token.macronized == "e_jus"

    def test_alsomaius_flag_does_not_add_macron_for_known_short_prefixes(
        self, macronizer
    ):
        """
        GIVEN the alsomaius flag is True,
        WHEN macronize is called on a word with a known short-j prefix,
        THEN it should NOT add a macron on the vowel preceding the 'j'.
        """
        # Arrange
        token = macronizer.Token("reiecit")
        token.accented = ["rejecit"]  # 'rej' is in var prefixeswithshortj

        # Act
        token.macronize(
            domacronize=True, alsomaius=True, performutov=False, performitoj=True
        )

        # Assert
        # The 'alsomaius' logic should be skipped, and no macron should be added.
        assert token.macronized == "rejecit"

    def test_sets_macronized_attribute_from_get_macronized(self, macronizer, mocker):
        """
        Verifies that Token.macronize() sets the `macronized` attribute
        with the return value from its method `get_macronized`.
        """
        # Arrange
        token = macronizer.Token("testword")
        stubbed_return_value = "MACRONIZED_FORM"
        mocker.patch.object(token, "get_macronized", return_value=stubbed_return_value)

        # Act
        token.macronize(
            domacronize=True, alsomaius=False, performutov=False, performitoj=False
        )

        # Assert
        assert token.macronized == stubbed_return_value


class TestTokenizationScanverses:
    """
    Tests for Tokenization.scanverses, focusing on prioritization in possiblescans.
    """

    @pytest.fixture
    def scanverses_setup(self, macronizer):
        """
        Helper to run a minimal scansion with a single token and a custom automaton.
        """

        def _setup_and_run(accented_list, automaton=None):
            if automaton is None:
                # Neutral automaton: accepts any sequence of L/S with zero penalty.
                automaton = {(0, "L"): (0, "L", 0), (0, "S"): (0, "S", 0)}

            t = macronizer.Tokenization("")
            tok = macronizer.Token("word")
            tok.accented = accented_list[:]  # copy
            tok.isword = True

            t.tokens = [tok]
            t.scanverses([automaton])
            return t.tokens[0].accented[0]

        return _setup_and_run

    def test_regression_single_candidate_no_ambiguity(self, scanverses_setup):
        """
        One unambiguous candidate should remain selected.
        """
        selected = scanverses_setup(accented_list=["ba_"])
        assert selected == "ba_"

    def test_regression_two_candidates_order_preserved_with_neutral_meter(
        self, scanverses_setup
    ):
        """
        Two unambiguous candidates: the first is preferred.
        """
        selected = scanverses_setup(accented_list=["ba_", "ba"])
        assert selected == "ba_"

    def test_new_behavior_ambiguous_variants_are_not_penalized(self, scanverses_setup):
        """
        Variants of the same ambiguous candidate ('ba_^' -> ['ba', 'ba_'])
        should not be reprioritized. With equal penalties, the 'L' scansion ('ba_')
        should win over the 'S' scansion ('ba') due to ordering in possiblescans.
        """
        selected = scanverses_setup(accented_list=["ba_^"])
        assert selected == "ba_"

    def test_new_behavior_mixed_candidates_prioritization(self, scanverses_setup):
        """
        Variants of the top candidate ('ba_^') should be preferred over a lower-ranked
        second candidate ('ba_'). We still expect the selected accented form to come
        from the first candidate's expansion.
        """
        selected = scanverses_setup(accented_list=["ba_^", "ba_"])
        assert selected == "ba_"

    def test_meter_can_override_ambiguous_variant_preference(self, scanverses_setup):
        """
        A meter penalty can force the short ('ba') to be chosen over the long ('ba_'),
        even when both variants belong to the top candidate.
        """
        meter_prefers_short = {
            (0, "S"): (0, "S", 0),  # No penalty for short
            (0, "L"): (0, "L", 5),  # High penalty for long
        }
        selected = scanverses_setup(
            accented_list=["ba_^"], automaton=meter_prefers_short
        )
        assert selected == "ba"

    def test_strong_meter_can_override_lexical_preference(self, scanverses_setup):
        """
        A strong meter penalty can overcome REPRIORITIZE_PENALTY to select
        a lower-ranked candidate ('ba' over 'ba_').
        """
        meter_prefers_short = {
            (0, "S"): (0, "S", 0),  # Penalty 0
            (0, "L"): (0, "L", 5),  # Penalty 5
        }
        # 'ba_': base 0 + meter 5 = 5
        # 'ba' : base 1 + meter 0 = 1  <-- wins
        selected = scanverses_setup(
            accented_list=["ba_", "ba"], automaton=meter_prefers_short
        )
        assert selected == "ba"


def test_macronizer_init_stores_rftagger_dir_from_config(
    macronizer, create_config_ini, db_conn
):
    """
    Verifies that Macronizer.__init__ correctly reads the config file
    and stores the value in the `rftagger_dir` attribute.
    """
    # Arrange
    ini_content = "[paths]\nrftagger_dir = /path/from/config"
    config_path = create_config_ini(ini_content)

    # Act
    mz = macronizer.Macronizer(db_conn, config_path=config_path)

    # Assert
    assert mz.rftagger_dir == "/path/from/config"


def test_macronizer_settext_passes_configured_path_to_addtags(
    macronizer, mocker, db_conn
):
    """
    Verifies that Macronizer.settext calls tokenization.addtags
    using the value stored in `self.rftagger_dir`.
    """
    # Arrange
    mz = macronizer.Macronizer(db_conn, config_path="dummy.ini")
    mz.rftagger_dir = "/path/stored/in/self"

    # We only need to mock two things:
    # 1. The Wordlist method that hits the database.
    # 2. The Tokenization class to intercept the `addtags` call.
    mocker.patch.object(mz.wordlist, "loadwords")

    # Create a mock instance that will be returned when Tokenization() is called.
    mock_tokenization_instance = mocker.MagicMock()
    mocker.patch("macronizer.Tokenization", return_value=mock_tokenization_instance)

    # Act
    mz.settext("some text")

    # Assert
    mock_tokenization_instance.addtags.assert_called_once_with("/path/stored/in/self")


def test_tokenization_addtags_uses_provided_dir_to_build_executable_path(macronizer):
    """
    Verifies that Tokenization.addtags uses the `rftagger_dir` argument
    it receives to construct the path to the external executable.
    """
    # Arrange
    tokenization = macronizer.Tokenization("test")
    non_existent_dir = "/this/path/definitely/does/not/exist"

    # Act & Assert
    # We expect an error because the path is invalid. We check that the
    # error message contains the correctly constructed path, which proves
    # the argument was used as intended.
    with pytest.raises(macronizer.ExternalDependencyError) as exc_info:
        tokenization.addtags(rftagger_dir=non_existent_dir)

    expected_path_in_error = os.path.join(non_existent_dir, "rft-annotate")
    assert expected_path_in_error in str(exc_info.value)


def test_evaluate_calculates_accuracy_correctly_with_stub(macronizer, mocker):
    """
    Tests the evaluate function with a mix of correct and incorrect vowels.
    """

    def stub_remove_macrons(text):
        macron_map = str.maketrans("āēīōūȳăĕĭŏŭ", "aeiouyaeiou")
        return text.translate(macron_map)

    mocker.patch("macronizer.postags.removemacrons", side_effect=stub_remove_macrons)

    # Arrange
    gold = "canō"
    macronized = "cano"

    # Act
    accuracy, html_output = macronizer.evaluate(gold, macronized)

    # Assert
    assert accuracy == 0.5
    expected_html = 'can<span class="wrong">o</span>'
    assert html_output == expected_html


def test_evaluate_handles_no_vowels_gracefully(macronizer):
    """
    Tests that the evaluate function returns 1.0 accuracy
    as there are no vowels to be incorrect about.
    """
    # Arrange
    gold = "psst"
    macronized = "psst"

    # Act
    accuracy, html_output = macronizer.evaluate(gold, macronized)

    # Assert
    assert accuracy == 1.0
    assert html_output == "psst"


def test_evaluate_raises_on_text_mismatch(macronizer):
    """
    Tests that evaluate() raises an InvalidArgumentError if the underlying
    plain text of the two strings does not match.
    """
    # Arrange
    gold = "arma"
    macronized = "arms"  # Mismatched last character

    # Act & Assert
    with pytest.raises(macronizer.InvalidArgumentError) as exc_info:
        macronizer.evaluate(gold, macronized)

    assert "Text mismatch" in str(exc_info.value)


class TestWordlist:
    """Tests for the `Wordlist` class."""

    def test_loadwordfromdb_raises_unrelated_errors_directly(
        self, macronizer, mocker, monkeypatch, db_conn
    ):
        """
        Verifies that a non-database error is not caught and masked.
        """
        # Arrange
        wl = macronizer.Wordlist(db_conn)
        mock_cursor = mocker.MagicMock()
        mock_cursor.execute.side_effect = TypeError("A programming mistake!")
        monkeypatch.setattr(wl, "dbcursor", mock_cursor)

        # Act & Assert
        with pytest.raises(TypeError) as exc_info:
            wl.loadwordfromdb("some_word")

        assert "A programming mistake!" in str(exc_info.value)
        assert not isinstance(exc_info.value, macronizer.DatabaseError)

    def test_loadwordfromdb_converts_sqlite_error_to_database_error(
        self, macronizer, mocker, monkeypatch, db_conn
    ):
        """
        Verifies that a genuine sqlite3.Error is correctly caught and re-raised.
        """
        # Arrange
        wl = macronizer.Wordlist(db_conn)
        mock_cursor = mocker.MagicMock()
        mock_error = sqlite3.OperationalError("mocked DB failure")
        mock_cursor.execute.side_effect = mock_error
        monkeypatch.setattr(wl, "dbcursor", mock_cursor)

        # Act & Assert
        with pytest.raises(macronizer.DatabaseError) as exc_info:
            wl.loadwordfromdb("some_word")

        msg = str(exc_info.value)
        assert "Query failed" in msg
        assert "mocked DB failure" in msg


def test_token_show_prints_correctly_formatted_output(macronizer, capsys):
    """
    This test verifies the the output sent to stdout is correctly formatted.
    """
    # Arrange
    token = macronizer.Token("arma")
    token.tag = "NOUN"
    token.lemma = "arma"
    token.accented = ["arma_"]
    expected_string = "arma\tNOUN\tarma\tarma_"
    expected_output = expected_string.expandtabs(16) + "\n"  # print() adds a newline

    # Act
    token.show()
    captured = capsys.readouterr()  # Capture what was printed

    # Assert
    assert captured.out == expected_output


def test_tokenization_scanverses_handles_elision_correctly(macronizer):
    """
    Regression Test for the `followingtext` vowel-peeking logic.

    This test verifies that elision is handled correctly. A word ending
    in a vowel ('vita') should have its final syllable elided when the
    next word begins with a vowel ('est').

    - 'vi_ta' is Long-Short. Elision leaves the first syllable 'vīt-' (L).
    - 'est' is long by position because the vowel 'e' is followed by 'st' (L).
    - The correct elided scansion is therefore 'LL'.
    """
    # Arrange
    tokenization = macronizer.Tokenization("vita est")

    # Manually set up tokens with explicit accentuation for a deterministic test.
    tok1 = macronizer.Token("vita")
    tok1.isword = True
    tok1.accented = ["vi_ta"]  # Explicitly Long-Short

    tok2 = macronizer.Token("est")
    tok2.isword = True
    tok2.accented = ["est"]  # Vowel is short, but syllable is long by position

    tokenization.tokens = [tok1, macronizer.Token(" "), tok2]

    # This automaton defines a valid verse as a simple "LL" sequence.
    # It will accept the elided scansion and reject any other path.
    # The tuples are (next_state, foot_to_append, penalty).
    meter = {
        (0, "L"): (1, "L", 0),  # From start (0), on 'L', go to state 1.
        (1, "L"): (0, "L", 0),  # From state 1, on 'L', go to end (0).
    }

    # Act
    tokenization.scanverses([meter])

    # Assert
    # The winning scansion path must be the elided one, resulting in "LL".
    assert tokenization.scannedfeet == ["LL"]


def test_toascii_handles_y_diaeresis_correctly(macronizer):

    assert macronizer.toascii("test_ÿ_test") == "test_y_test"


def test_crunchwords_parsing(macronizer, monkeypatch, db_conn):
    """
    Verifies the Morpheus output parser
    """
    # Arrange
    wl = macronizer.Wordlist(db_conn)
    wl.reinitializedatabase()

    morpheus_output = (
        "arma\n"
        "<NL>N arma  neut nom/voc/acc pl</NL>\n"
        "cano\n"
        "<NL>V ca^no_,cano  pres ind act 1st sg</NL>\n"
    )

    # Mock the external Morpheus call to return our controlled output.
    # We also mock the executable check to prevent a FileNotFoundError.
    monkeypatch.setattr(os.path, "isfile", lambda _: True)
    monkeypatch.setattr(os, "access", lambda _p, _m: True)

    def fake_run(*_args, stdout, **_kwargs):
        stdout.write(morpheus_output.encode("utf-8"))
        stdout.flush()

    monkeypatch.setattr(macronizer, "run_external", fake_run)

    input_words = {"arma", "virumque", "cano"}

    # Act
    wl.crunchwords(input_words)

    # Assert
    # Query the database to verify the final state of each word.
    wl.dbcursor.execute(
        "SELECT wordform, accented FROM morpheus WHERE wordform IN (?, ?, ?)",
        ("arma", "cano", "hehehe"),
    )
    results = dict(wl.dbcursor.fetchall())

    assert results.get("arma") is not None
    assert results.get("virumque") is None
    assert results.get("cano") is not None


def test_crunchwords_parsing_raises_on_dangling_word(macronizer, monkeypatch, db_conn):
    """
    Verifies that the Morpheus output parser raises exception on a truncated
    output file where a final word line is not followed by its parse line.

    This ensures no data is silently lost with anomalous Morpheus output.
    """
    # Arrange
    wl = macronizer.Wordlist(db_conn)
    wl.reinitializedatabase()

    # This Morpheus output is intentionally malformed.
    morpheus_output = (
        "arma\n"
        "<NL>N arma  neut nom/voc/acc pl</NL>\n"
        "cano\n"
        "<NL>V ca^no_,cano  pres ind act 1st sg</NL>\n"
        "hehehe"  # Dangling line
    )

    # Mock the external Morpheus call to return our controlled output.
    # We also mock the executable check to prevent a FileNotFoundError.
    monkeypatch.setattr(os.path, "isfile", lambda _: True)
    monkeypatch.setattr(os, "access", lambda _p, _m: True)

    def fake_run(*_args, stdout, **_kwargs):
        stdout.write(morpheus_output.encode("utf-8"))
        stdout.flush()

    monkeypatch.setattr(macronizer, "run_external", fake_run)

    # The input set must contain all words we expect to process.
    input_words = {"arma", "virumque", "cano", "hehehe"}

    # Act & Assert
    with pytest.raises(macronizer.ParsingError):
        wl.crunchwords(input_words)


@pytest.mark.parametrize(
    "gold, macronized, description",
    [
        ("ārma", "arm", "macronized is shorter"),
        ("arm", "ārma", "macronized is longer"),
    ],
)
def test_evaluate_raises_on_mismatched_lengths(
    macronizer, gold, macronized, description
):
    """
    Verifies that evaluate() raises an InvalidArgumentError immediately if the
    input strings have different lengths.
    """
    # Arrange done by parametrize

    # Act & Assert
    with pytest.raises(macronizer.InvalidArgumentError) as exc_info:
        macronizer.evaluate(gold, macronized)

    # Check that the error message clearly indicates a length mismatch.
    assert (
        "length" in str(exc_info.value).lower()
    ), f"Test failed for case: {description}"


def test_scanverse_handles_penalties_greater_than_100(macronizer):
    """
    Verifies that the scansion algorithm can find a valid path even if its
    total penalty exceeds 100.

    This test provides two possible accented forms for a word:
    1. "a_": Scans as 'L'. This is the lexically preferred form (base penalty 0).
    2. "a":  Scans as 'S'. This is the disfavored form (base penalty 1).

    We then provide a meter that strongly prefers 'S' over 'L':
    - It REJECTS the 'L' scansion entirely by having no valid transition for it.
    - It ACCEPTS the 'S' scansion but applies a high meter penalty (101).

    The total penalty for the only valid path ('S') is:
    1 (lexical REPRIORITIZE_PENALTY) + 101 (meter penalty) = 102.
    """
    # Arrange
    tokenization = macronizer.Tokenization("word")

    # Manually set up a token with two competing accentuations that will be
    # correctly parsed by the real `segmentaccented` function.
    token = macronizer.Token("word")
    token.isword = True
    token.accented = ["a_", "a"]  # "a_" has priority; "a" is the fallback.
    tokenization.tokens = [token]

    # This meter rejects 'L' and accepts 'S' with a very high penalty.
    meter = {
        # No path for 'L', so it's an invalid scansion.
        (0, "S"): (0, "S", 101)  # Path for 'S' is valid, but costs 101.
    }

    # Act
    tokenization.scanverses([meter])

    # Assert
    assert tokenization.scannedfeet == ["S"]


def test_tokenization_get_structured_output_orchestrates_calls_to_token_methods(
    macronizer, mocker
):
    """
    Verifies that Tokenization.get_structured_output correctly calls
    get_macronized() and get_structured_output() on each of its tokens.
    """
    # Arrange
    # Create a tokenization with a word, a space, and another word.
    tokenization = macronizer.Tokenization("arma virumque")
    # Mock the two methods on the Token class that will be called.
    # We use side_effect to return a unique value for each call,
    # allowing us to verify that the flow is correct.
    mocker.patch(
        "macronizer.Token.get_macronized",
        side_effect=["m_form1", "m_form2", "m_form3"],
    )
    mock_get_structured = mocker.patch(
        "macronizer.Token.get_structured_output",
        side_effect=["result1", "result2", "result3"],
    )
    # These are the arguments we expect to be passed to get_macronized()
    # for each of the three tokens.
    expected_macronize_args = {
        "domacronize": True,
        "alsomaius": False,
        "performutov": True,
        "performitoj": False,
    }

    # Act
    final_results = tokenization.get_structured_output(**expected_macronize_args)

    # Assert
    # The final list must be the collected results from get_structured_output.
    assert final_results == ["result1", "result2", "result3"]
    # Verify that get_structured_output was called with the output
    # from get_macronized for each respective token.
    assert mock_get_structured.call_args_list[0] == mocker.call("m_form1")
    assert mock_get_structured.call_args_list[1] == mocker.call("m_form2")
    assert mock_get_structured.call_args_list[2] == mocker.call("m_form3")


class TestTokenGetStructuredOutput:
    """
    Tests for the Token.get_structured_output method.
    """

    @pytest.fixture
    def _setup_token(self, functional_macronizer):
        """Helper fixture to create and configure a token for tests."""

        def _factory(text, accented_forms=None, is_unknown=False):
            # Use the Token class from the dedicated fixture's module
            token = functional_macronizer.Token(text)
            if accented_forms is not None:
                token.accented = accented_forms
            token.isunknown = is_unknown
            return token

        return _factory

    def test_returns_correct_structure_for_non_word(self, _setup_token):
        token = _setup_token(" ")
        result = token.get_structured_output(" ")
        assert result["is_word"] is False

    def test_returns_full_bitmask_for_unknown_word(self, _setup_token):
        token = _setup_token("ignotus", is_unknown=True)
        result = token.get_structured_output("ignotus")
        assert result["uncertainty_mask"] == 127  # 2^7 - 1

    def test_returns_zero_mask_for_unambiguous_word(self, _setup_token):
        token = _setup_token("non", accented_forms=["no_n"])
        result = token.get_structured_output("no_n")
        assert result["uncertainty_mask"] == 0

    def test_sets_correct_bit_for_single_ambiguous_vowel(self, _setup_token):
        token = _setup_token("uenit", accented_forms=["ue_nit", "uenit"])
        result = token.get_structured_output("ue_nit")
        assert result["uncertainty_mask"] == 2  # 2^1

    def test_sets_correct_bits_for_multiple_ambiguous_vowels(self, _setup_token):
        token = _setup_token(
            "mala", accented_forms=["ma_la_", "mala_", "ma_la", "mala"]
        )
        result = token.get_structured_output("ma_la_")
        assert result["uncertainty_mask"] == 10

    def test_populates_candidates_list_correctly(self, _setup_token):
        token = _setup_token(
            "mala", accented_forms=["ma_la_", "mala_", "ma_la", "mala"]
        )
        result = token.get_structured_output("mala")
        assert result["candidates"] == ["malā", "māla", "mala"]

    def test_returns_empty_candidates_list_for_unambiguous_word(self, _setup_token):
        token = _setup_token("quorum", accented_forms=["quo_rum"])
        result = token.get_structured_output("quo_rum")
        assert result["candidates"] == []

    def test_returns_correct_mask_when_candidates_have_mismatched_skeletons(
        self, _setup_token
    ):
        token = _setup_token("amica", accented_forms=["ami_ca", "ami_ca_", "ami_cus"])
        result = token.get_structured_output("ami_ca")
        # 'amicus' is ignored; mask is based on 'ami_ca' vs 'ami_ca_'
        assert result["uncertainty_mask"] == 16

    def test_formats_final_macronized_word_using_unicodeaccents_stub(
        self, _setup_token
    ):
        token = _setup_token("test", accented_forms=["te_st"])
        result = token.get_structured_output("te_st")
        assert result["macronized"] == "tēst"

    def test_deduplicates_candidates_after_processing_special_notation(
        self, _setup_token
    ):
        """
        Verifies that candidates are made unique *after* processing notations
        """
        # Arrange
        accented_forms = ["pro_spera", "pro_spera_", "pro_spe^ra", "pro_spe^ra_"]
        token = _setup_token("prospera", accented_forms=accented_forms)

        # Act
        # The first form 'pro_spera' will be the best guess
        result = token.get_structured_output("pro_spera")

        # Assert
        # The primary macronized form should be the unicode version of the best guess.
        assert result["macronized"] == "prōspera"
        # The candidates list should contain only the unique, alternative forms.
        #    'pro_spera_' and 'pro_spe^ra_' both become 'prōsperā'.
        #    The duplicate 'prōspera' from 'pro_spe^ra' should be removed.
        assert result["candidates"] == ["prōsperā"]
        # The uncertainty mask should be based on the ambiguity between the
        #    unique candidates: 'prōspera' vs 'prōsperā'. The final 'a' is ambiguous.
        #    The word 'prospera' is 8 chars long, so the last char is at index 7 (2^7)
        assert result["uncertainty_mask"] == 128


class TestCandidateCasing:
    """
    Tests that the `candidates` list in the structured output
    correctly preserves the casing of the original input token.
    """

    def test_candidates_for_uppercase_input_are_recased_to_uppercase(
        self, functional_macronizer
    ):
        """
        GIVEN an all-caps token "VENIT",
        WHEN its candidates are generated from title-cased ("VE_NIT") and
        lowercase ("venit") lemmas,
        THEN the final candidates list must also be all-caps.
        """
        # Arrange
        token = functional_macronizer.Token("VENIT")
        token.accented = ["ve_nit", "venit"]
        primary_macronized_form = "VE_NIT"

        # Act
        result = token.get_structured_output(primary_macronized_form)

        # Assert
        assert result["macronized"] == "VĒNIT"
        assert result["candidates"] == ["VENIT"]

    def test_candidates_for_titlecase_input_are_recased_to_titlecase(
        self, functional_macronizer
    ):
        """
        GIVEN a title-cased token "Venit",
        WHEN its candidates are generated, including a lowercase one,
        THEN the final candidate must be re-cased to title-case.
        """
        # Arrange
        token = functional_macronizer.Token("Venit")
        token.accented = ["ve_nit", "venit"]
        primary_macronized_form = "Ve_nit"

        # Act
        result = token.get_structured_output(primary_macronized_form)

        # Assert
        assert result["macronized"] == "Vēnit"
        assert result["candidates"] == ["Venit"]

    def test_candidates_for_lowercase_input_remain_lowercase_as_expected(
        self, functional_macronizer
    ):
        """
        GIVEN a lowercase token "venit" with multiple valid lowercase candidates,
        WHEN its candidates are generated,
        THEN the final list should correctly remain lowercase.
        This test serves as a non-regression check for the most common use case.
        """
        # Arrange
        token = functional_macronizer.Token("venit")
        token.accented = ["ve_nit", "venit"]
        primary_macronized_form = "ve_nit"

        # Act
        result = token.get_structured_output(primary_macronized_form)

        # Assert
        assert result["macronized"] == "vēnit"
        assert result["candidates"] == ["venit"]


class TestCapitalizationAndVerseLogic:
    """
    Tests the logic for handling capitalized words, especially at the start of verses/sentences.
    """

    # Word with both proper noun lemma and not
    AUGUSTUS_DATA = {
        "augustus": [
            ("TAG", "Augustus", "Augustus_acc"),  # Proper noun (Capitalized)
            ("TAG", "augustus", "augustus_acc"),  # lower case
        ]
    }
    # Word with only not proper noun lemma
    VERUM_DATA = {
        "verum": [
            ("TAG", "verum", "ve_rum"),
        ]
    }
    # Word with only a proper noun lemma
    CAESAR_DATA = {
        "caesar": [
            ("TAG", "Caesar", "Caesar_acc"),
        ]
    }

    @classmethod
    def setup_class(cls):
        """Injects minimal fake modules with necessary attributes into sys.modules."""
        # Check if stubs already exist from other tests, if not, create them.
        if "postags" not in sys.modules:
            postags = types.ModuleType("postags")
            postags.tag_distance = lambda a, b: 0
            postags.removemacrons = lambda s: s
            sys.modules["postags"] = postags

        if "lemmas" not in sys.modules:
            lemmas = types.ModuleType("lemmas")
            sys.modules["lemmas"] = lemmas
        # Ensure the required attributes exist for the import to succeed.
        lemmas_module = sys.modules["lemmas"]
        if not hasattr(lemmas_module, "lemma_frequency"):
            lemmas_module.lemma_frequency = {}
        if not hasattr(lemmas_module, "word_lemma_freq"):
            lemmas_module.word_lemma_freq = {}
        if not hasattr(lemmas_module, "wordform_to_corpus_lemmas"):
            lemmas_module.wordform_to_corpus_lemmas = {}

        if "macronized_endings" not in sys.modules:
            mac_end = types.ModuleType("macronized_endings")
            mac_end.tag_to_endings = {}
            sys.modules["macronized_endings"] = mac_end

    @pytest.fixture(name="macronizer_verse_test_fixture")
    def macronizer_verse_test_fixture_func(self, mocker):
        """
        Provides a factory to test Tokenization.getaccents, which contains the core capitalization logic.
        """
        from macronizer import Tokenization

        mocker.patch("macronizer.Tokenization.levenshtein", return_value=0)

        def _run_test(text_input: str, mock_data: dict):
            """Factory function to run a single test case."""
            word_to_find = list(mock_data.keys())[0]
            mock_wordlist = mocker.MagicMock()
            mock_wordlist.formtotaglemmaaccents = mock_data

            mock_wordlist.formtoaccenteds = defaultdict(list)
            if word_to_find in mock_data:
                accented_forms = [p[2] for p in mock_data[word_to_find]]
                mock_wordlist.formtoaccenteds[word_to_find] = accented_forms

            tokenization = Tokenization(text_input)
            for t in tokenization.tokens:
                if t.isword:
                    t.tag = "TAG"
                    t.lemma = t.text.lower()

            tokenization.getaccents(mock_wordlist)

            word_token = next(
                (t for t in tokenization.tokens if t.text.lower() == word_to_find),
                None,
            )
            assert (
                word_token is not None
            ), f"Test setup failed: token for '{word_to_find}' not found in '{text_input}'"
            return word_token

        return _run_test

    @pytest.mark.parametrize(
        "description, text_input, db_data_key, expected_isunknown, expected_accented",
        [
            # === SCENARIO: Only not proper noun available ('verum') ===
            (
                "lower, start, only lower lemma -> macronize",
                "verum et.",
                "VERUM",
                False,
                ["ve_rum"],
            ),
            (
                "lower, mid-sentence, only lower lemma -> macronize",
                "et verum.",
                "VERUM",
                False,
                ["ve_rum"],
            ),
            (
                "Capitalized, start, only lower lemma -> macronize (forgiven)",
                "Verum et.",
                "VERUM",
                False,
                ["ve_rum"],
            ),
            (
                "Capitalized, mid-sentence, only lower lemma -> unknown",
                "et Verum.",
                "VERUM",
                True,
                ["Verum"],
            ),
            (
                "Capitalized, mid-sentence sequential, only lower lemma -> macronize (forgiven)",
                "Et Verum.",
                "VERUM",
                False,
                ["ve_rum"],
            ),
            (
                "ALLCAPS, start, only lower lemma -> macronize (forgiven)",
                "VERUM et.",
                "VERUM",
                False,
                ["ve_rum"],
            ),
            (
                "ALLCAPS, mid-sentence, only lower lemma -> unknown",
                "et VERUM.",
                "VERUM",
                True,
                ["VERUM"],
            ),
            # === SCENARIO: Only proper noun lemma available ('Caesar') ===
            (
                "lower, start, only proper noun -> unknown",
                "caesar et.",
                "CAESAR",
                True,
                ["caesar"],
            ),
            (
                "lower, mid-sentence, only proper noun -> unknown",
                "et caesar.",
                "CAESAR",
                True,
                ["caesar"],
            ),
            (
                "Capitalized, start, only proper noun -> macronize",
                "Caesar et.",
                "CAESAR",
                False,
                ["Caesar_acc"],
            ),
            (
                "Capitalized, mid-sentence sequential, only proper noun -> macronize",
                "Et Caesar.",
                "CAESAR",
                False,
                ["Caesar_acc"],
            ),
            (
                "Capitalized, mid-sentence, only proper noun -> macronize",
                "et Caesar.",
                "CAESAR",
                False,
                ["Caesar_acc"],
            ),
            (
                "ALLCAPS, start, only proper noun -> macronize (forgiven)",
                "CAESAR et.",
                "CAESAR",
                False,
                ["Caesar_acc"],
            ),
            (
                "ALLCAPS, mid-sentence, only proper noun -> unknown",
                "et CAESAR.",
                "CAESAR",
                True,
                ["CAESAR"],
            ),
            # === SCENARIO: Both proper noun and not available ('augustus' + 'Augustus') ===
            (
                "lower, start, both options -> macronize from lower only",
                "augustus et.",
                "AUGUSTUS",
                False,
                ["augustus_acc"],
            ),
            (
                "lower, mid-sentence, both options -> macronize from lower",
                "et augustus.",
                "AUGUSTUS",
                False,
                ["augustus_acc"],
            ),
            (
                "Capitalized, start, both options -> consider both",
                "Augustus et.",
                "AUGUSTUS",
                False,
                ["Augustus_acc", "augustus_acc"],
            ),
            (
                "Capitalized, start (newline), both options -> consider both",
                "finis\nAugustus",
                "AUGUSTUS",
                False,
                ["Augustus_acc", "augustus_acc"],
            ),
            (
                "Capitalized, start (punct), both options -> consider both",
                "finis. Augustus",
                "AUGUSTUS",
                False,
                ["Augustus_acc", "augustus_acc"],
            ),
            (
                "Capitalized, mid-sentence, both options -> macronize from Title only",
                "et Augustus.",
                "AUGUSTUS",
                False,
                ["Augustus_acc"],
            ),
            (
                "Capitalized, mid-sentence sequential, both options -> consider both",
                "Divus Augustus.",
                "AUGUSTUS",
                False,
                ["Augustus_acc", "augustus_acc"],
            ),
            (
                "ALLCAPS, start, both options -> consider both",
                "AUGUSTUS et.",
                "AUGUSTUS",
                False,
                ["Augustus_acc", "augustus_acc"],
            ),
            (
                "ALLCAPS, mid-sentence, both options -> unknown",
                "et AUGUSTUS.",
                "AUGUSTUS",
                True,
                ["AUGUSTUS"],
            ),
        ],
    )
    def test_capitalization_logic(
        self,
        macronizer_verse_test_fixture,
        description,
        text_input,
        db_data_key,
        expected_isunknown,
        expected_accented,
    ):
        """
        Tests the complete matrix of capitalization, word position, and lemma availability.
        """
        db_data_map = {
            "VERUM": self.VERUM_DATA,
            "CAESAR": self.CAESAR_DATA,
            "AUGUSTUS": self.AUGUSTUS_DATA,
        }
        db_data = db_data_map[db_data_key]

        token = macronizer_verse_test_fixture(text_input, db_data)

        assert (
            token.isunknown == expected_isunknown
        ), f"Failed isunknown check for: {description}"

        # For cases where we expect both proper and common noun lemmas, we must
        # assert the correct order, not just the content.
        if db_data_key == "AUGUSTUS" and len(expected_accented) > 1:
            # The proper noun 'Augustus_acc' must be the primary suggestion.
            assert (
                token.accented[0] == "Augustus_acc"
            ), f"Failed primary candidate check for: {description}"
            # Verify the set of all candidates is correct.
            assert set(token.accented) == set(
                expected_accented
            ), f"Failed full candidate set check for: {description}"
        else:
            # For all other cases, an order-agnostic check is sufficient.
            assert sorted(token.accented) == sorted(
                expected_accented
            ), f"Failed accented check for: {description}"

    def test_all_caps_sequence_is_macronized(self, mocker):
        """
        Verifies that a sequence of ALL CAPS words is treated as a stylistic choice,
        and that the proper noun preference is correctly applied.
        """
        from macronizer import Tokenization

        # Arrange
        text_input = "DIVUS IULIUS CAESAR"
        # 'iulius' has both lemma types, making it the critical test case.
        mock_data = {
            "divus": [("TAG", "divus", "di_vus_acc")],
            "iulius": [
                ("TAG", "Iulius", "Iulius_acc"),
                ("TAG", "iulius", "iulius_acc"),
            ],
            "caesar": [("TAG", "Caesar", "Caesar_acc")],
        }

        mock_wordlist = mocker.MagicMock()
        mock_wordlist.formtotaglemmaaccents = mock_data

        # Act
        tokenization = Tokenization(text_input)
        for t in tokenization.tokens:
            if t.isword:
                t.tag = "TAG"
                t.lemma = t.text.lower()
        tokenization.getaccents(mock_wordlist)

        # Assert
        divus_token = next(t for t in tokenization.tokens if t.text == "DIVUS")
        iulius_token = next(t for t in tokenization.tokens if t.text == "IULIUS")
        caesar_token = next(t for t in tokenization.tokens if t.text == "CAESAR")
        # The first word is forgiven because it's at the start.
        assert not divus_token.isunknown and divus_token.accented == ["di_vus_acc"]
        # The proper noun lemma is preferred (ranked first),
        # but the common noun is still included as a valid candidate.
        assert not iulius_token.isunknown
        assert iulius_token.accented[0] == "Iulius_acc"
        assert set(iulius_token.accented) == {"Iulius_acc", "iulius_acc"}
        # The forgiveness and preference logic continues for the third word.
        assert not caesar_token.isunknown and caesar_token.accented == ["Caesar_acc"]

    def test_title_case_sequence_is_macronized(self, mocker):
        """
        Verifies that a sequence of Title Case words is treated as a stylistic choice,
        and that the proper noun preference is correctly applied.
        """
        from macronizer import Tokenization

        # Arrange
        text_input = "Divus Iulius Caesar"
        # 'iulius' has both lemma types, making it the critical test case.
        mock_data = {
            "divus": [("TAG", "divus", "di_vus_acc")],
            "iulius": [
                ("TAG", "Iulius", "Iulius_acc"),
                ("TAG", "iulius", "iulius_acc"),
            ],
            "caesar": [("TAG", "Caesar", "Caesar_acc")],
        }

        mock_wordlist = mocker.MagicMock()
        mock_wordlist.formtotaglemmaaccents = mock_data

        # Act
        tokenization = Tokenization(text_input)
        for t in tokenization.tokens:
            if t.isword:
                t.tag = "TAG"
                t.lemma = t.text.lower()
        tokenization.getaccents(mock_wordlist)

        # Assert
        divus_token = next(t for t in tokenization.tokens if t.text == "Divus")
        iulius_token = next(t for t in tokenization.tokens if t.text == "Iulius")
        caesar_token = next(t for t in tokenization.tokens if t.text == "Caesar")

        # The first word is forgiven because it's at the start.
        assert not divus_token.isunknown and divus_token.accented == ["di_vus_acc"]
        # The proper noun lemma is preferred (ranked first),
        # but the common noun is still included as a valid candidate.
        assert not iulius_token.isunknown
        assert iulius_token.accented[0] == "Iulius_acc"
        assert set(iulius_token.accented) == {"Iulius_acc", "iulius_acc"}
        # The forgiveness and preference logic continues for the third word.
        assert not caesar_token.isunknown and caesar_token.accented == ["Caesar_acc"]

    def test_tokenizer_flags_verse_start_correctly(self):
        """
        Tests the prerequisite: word after a newline is flagged as `is_context_start`.
        """
        from macronizer import Tokenization

        tokenization = Tokenization("quem non\nFors ignara dedit.")
        fors_token = next(t for t in tokenization.tokens if t.text == "Fors")
        assert (
            fors_token.is_context_start is True
        ), "The word 'Fors' at the start of a new line should be flagged as a sentence/verse start."

    def test_tokenizer_flags_prose_sentence_start_correctly(self):
        """
        Tests the prerequisite: word after punctuation is flagged as `is_context_start`.
        """
        from macronizer import Tokenization

        tokenization = Tokenization("Finis. Novum initium.")
        novum_token = next(t for t in tokenization.tokens if t.text == "Novum")
        assert (
            novum_token.is_context_start is True
        ), "The word 'Novum' after a period should be flagged as a sentence start."

    def test_tokenizer_does_not_flag_mid_sentence_word(self):
        """
        Tests the prerequisite: a mid-sentence word is NOT flagged as `is_context_start`.
        """
        from macronizer import Tokenization

        tokenization = Tokenization("Arma virumque cano.")
        virumque_token = next(t for t in tokenization.tokens if t.text == "virumque")
        assert (
            virumque_token.is_context_start is False
        ), "The word 'virumque' mid-sentence should not be flagged as a sentence start."
