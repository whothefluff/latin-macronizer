import importlib
import os
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


def test_addtags_raises_when_rft_annotate_missing(macronizer, monkeypatch):
    t = macronizer.Tokenization("arma virumque cano")
    # Point to non-existent dir => binary not found
    monkeypatch.setattr(macronizer, "RFTAGGER_DIR", "/definitely/missing")

    with pytest.raises(macronizer.ExternalDependencyError) as ei:
        t.addtags()
    assert "rft" in str(ei.value)
    assert "not found or not executable" in str(ei.value)


def test_addtags_reads_output_from_external_using_tempfiles(
    macronizer, tmp_path, monkeypatch
):
    # Single word tokenization; no sentence-end, no enclitics
    t = macronizer.Tokenization("arma")

    # Provide executable rft-annotate so path check passes
    rft = tmp_path / "rft-annotate"
    rft.write_text("", encoding="utf-8")
    os.chmod(rft, 0o755)
    monkeypatch.setattr(macronizer, "RFTAGGER_DIR", str(tmp_path))

    # Fake external tool writes the expected "token<TAB>tag" line to out path
    def fake_run(cmd, **_kwargs):
        out_path = cmd[-1]
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("arma\tNOUN\n")

    monkeypatch.setattr(macronizer, "run_external", fake_run)

    t.addtags()
    tok = next(tok for tok in t.tokens if tok.isword)
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
