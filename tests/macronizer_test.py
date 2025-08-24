import importlib
import os
import sqlite3
import subprocess
import sys
import types

import pytest


@pytest.fixture
def stub_modules():
    # Minimal stubs for dependencies so we can import macronizer
    postags = types.ModuleType("postags")
    postags.LEMMA = 0
    postags.ACCENTEDFORM = 1
    postags.removemacrons = lambda s: s
    postags.unicodeaccents = lambda s: s
    postags.tag_distance = lambda a, b: 0
    postags.parse_to_ldt = lambda p: "TAG"
    postags.morpheus_to_parses = lambda wordform, nl: []
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
    # pytest runs the `stub_modules` fixture first, which mocks dependencies.
    # This allows the subsequent import of `macronizer` inside this fixture to succeed.
    import macronizer as mod  # pylint: disable=import-outside-toplevel

    importlib.reload(mod)

    db_path = tmp_path / "test_macronizer.db"
    macrons_txt = tmp_path / "macrons.txt"
    macrons_txt.write_text("", encoding="utf-8")

    monkeypatch.setattr(mod, "USE_DB", True)
    monkeypatch.setattr(mod, "DB_NAME", str(db_path))
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


def test_crunchwords_raises_when_cruncher_missing(macronizer, tmp_path, monkeypatch):
    wl = macronizer.Wordlist()
    wl.reinitializedatabase()

    # Point MORPHEUS_DIR to a temp dir WITHOUT cruncher
    monkeypatch.setattr(macronizer, "MORPHEUS_DIR", str(tmp_path))

    with pytest.raises(macronizer.ExternalDependencyError) as ei:
        wl.crunchwords({"abc"})
    assert "cruncher not found" in str(ei.value)


def test_crunchwords_inserts_unknown_when_no_output_and_cleans_tempfiles(
    macronizer, tmp_path, monkeypatch
):
    wl = macronizer.Wordlist()
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


def test_crunchwords_sets_morphlib_env(macronizer, tmp_path, monkeypatch):
    wl = macronizer.Wordlist()
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


def test_macronizer_init_stores_rftagger_dir_from_config(macronizer, create_config_ini):
    """
    Verifies that Macronizer.__init__ correctly reads the config file
    and stores the value in the `rftagger_dir` attribute.
    """
    # Arrange
    ini_content = "[paths]\nrftagger_dir = /path/from/config"
    config_path = create_config_ini(ini_content)

    # Act
    mz = macronizer.Macronizer(config_path=config_path)

    # Assert
    assert mz.rftagger_dir == "/path/from/config"


def test_macronizer_settext_passes_configured_path_to_addtags(macronizer, mocker):
    """
    Verifies that Macronizer.settext calls tokenization.addtags
    using the value stored in `self.rftagger_dir`.
    """
    # Arrange
    mz = macronizer.Macronizer(config_path="dummy.ini")
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
        self, macronizer, mocker, monkeypatch
    ):
        """
        Verifies that a non-database error is not caught and masked.
        """
        # Arrange
        wl = macronizer.Wordlist()
        mock_cursor = mocker.MagicMock()
        mock_cursor.execute.side_effect = TypeError("A programming mistake!")
        monkeypatch.setattr(wl, "dbcursor", mock_cursor)

        # Act & Assert
        with pytest.raises(TypeError) as exc_info:
            wl.loadwordfromdb("some_word")

        assert "A programming mistake!" in str(exc_info.value)
        assert not isinstance(exc_info.value, macronizer.DatabaseError)

    def test_loadwordfromdb_converts_sqlite_error_to_database_error(
        self, macronizer, mocker, monkeypatch
    ):
        """
        Verifies that a genuine sqlite3.Error is correctly caught and re-raised.
        """
        # Arrange
        wl = macronizer.Wordlist()
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
