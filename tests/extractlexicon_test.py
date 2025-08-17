import builtins
import importlib.util
import io
import os
import xml.etree.ElementTree as ET
from unittest.mock import Mock

import pytest

from extractlexicon import (
    create_lemma_frequency_file,
    create_lexicon_and_endings_data,
    create_training_corpus,
)

XML_FILE_1_WITH_XSEG_END = """
<treebank>
    <sentence id="1">
        <word id="1" form="WordOne" lemma="word" postag="n-s---mn-" head="0" relation="PRED"/>
        <word id="2" form="Enclitic" lemma="other" postag="p--------" head="3" relation="XSEG"/>
    </sentence>
</treebank>
"""

XML_FILE_2_NORMAL = """
<treebank>
    <sentence id="1">
        <word id="1" form="WordTwo" lemma="word" postag="n-s---fn-" head="0" relation="PRED"/>
    </sentence>
</treebank>
"""

XML_POSTAG_FORMAT_TEST = """
<treebank>
    <sentence id="1">
        <word id="1" form="amo" lemma="amo" postag="v1spia---" head="0" relation="PRED"/>
    </sentence>
</treebank>
"""

XML_FOR_PATH_TESTS = """
<treebank>
    <sentence id="1">
        <word id="1" form="PathTest" lemma="path" postag="n-s---nn-" head="0" relation="PRED"/>
    </sentence>
</treebank>
"""

FAKE_XML_PATHS = ["treebank/file1.xml", "treebank/file2.xml"]
FAKE_XML_DATA = {
    "treebank/file1.xml": XML_FILE_1_WITH_XSEG_END,
    "treebank/file2.xml": XML_FILE_2_NORMAL,
    "treebank/postag_test.xml": XML_POSTAG_FORMAT_TEST,
}


class NonClosingStringIO(io.StringIO):
    """A StringIO buffer that doesn't close on `with` exit."""

    def close(self):
        """Do nothing, allowing value inspection after the `with` block."""


@pytest.fixture(name="mock_dependencies_buffer")
def mock_dependencies_fixture(monkeypatch):
    """A pytest fixture to set up all common mocks for tests."""
    # Mock config parser to prevent reading real files
    monkeypatch.setattr("configparser.ConfigParser.read", lambda *args, **kwargs: None)

    # Mock file system and XML parsing
    def mock_et_parse(filepath):
        if filepath not in FAKE_XML_DATA:
            raise FileNotFoundError(f"Mock file not found: {filepath}")
        xml_string = FAKE_XML_DATA.get(filepath)
        return ET.ElementTree(ET.fromstring(xml_string))

    monkeypatch.setattr("xml.etree.ElementTree.parse", mock_et_parse)

    # Mock `open` to intercept file writes and prevent disk access
    original_open = builtins.open
    mock_output_buffer = NonClosingStringIO()

    def mock_open_selective(filename, mode="r", **kwargs):
        if mode == "w":
            # Intercept *any* file being opened for writing
            # and redirect it to the in-memory buffer.
            return mock_output_buffer
        if filename == "corpus-supplement.txt" and mode == "r":
            # Return an empty supplement file for isolated testing
            return io.StringIO("")
        # Fallback to the real `open` if needed (it won't be in these tests)
        return original_open(filename, mode=mode, **kwargs)

    monkeypatch.setattr("builtins.open", mock_open_selective)

    # Return the mock buffer so tests can inspect its contents
    return mock_output_buffer


@pytest.fixture(name="mock_output_file_only")
def mock_output_file_only_fixture(monkeypatch):
    """A focused fixture that only mocks the output files to test real file reads."""
    original_open = builtins.open
    mock_output_buffer = NonClosingStringIO()

    def mock_open_for_write(filename, mode="r", **kwargs):
        # 1. Intercept writes to the main corpus file and redirect to memory
        if filename == "ldt-corpus.txt" and mode == "w":
            return mock_output_buffer
        # 2. ALSO intercept reads from the supplement file to isolate the test
        if filename == "corpus-supplement.txt" and mode == "r":
            return io.StringIO("")  # Return an empty in-memory file
        # 3. Allow all other file operations (like reading config and XML) to proceed normally
        return original_open(filename, mode, **kwargs)

    monkeypatch.setattr("builtins.open", mock_open_for_write)
    return mock_output_buffer


@pytest.fixture(name="lexicon_data_prep_runner")
def lexicon_data_prep_runner_fixture(tmp_path, monkeypatch):
    """Sets up a test environment and returns a runner function for lexicon data prep."""
    monkeypatch.chdir(tmp_path)

    def _run_with_macrons_data(macrons_data: str):
        (tmp_path / "macrons.txt").write_text(macrons_data, encoding="utf-8")
        create_lexicon_and_endings_data()
        endings_content = ""
        if (tmp_path / "macronized_endings.py").exists():
            endings_content = (tmp_path / "macronized_endings.py").read_text(
                encoding="utf-8"
            )
        return endings_content

    return _run_with_macrons_data


@pytest.fixture(name="corpus_test_env")
def corpus_test_env_fixture(tmp_path, monkeypatch):
    """Sets up a test environment for corpus creation tests."""
    monkeypatch.chdir(tmp_path)

    def _setup_files(ini_content, xml_files, supplement_content=""):
        (tmp_path / "test_corpus.ini").write_text(ini_content, encoding="utf-8")
        for fname, content in xml_files.items():
            full_xml = f"<root>\n{content}\n</root>"
            (tmp_path / fname).write_text(full_xml, encoding="utf-8")
        if supplement_content is not None:
            (tmp_path / "corpus-supplement.txt").write_text(
                supplement_content, encoding="utf-8"
            )

    yield _setup_files, tmp_path


@pytest.fixture(name="lemma_freq_runner")
def lemma_freq_runner_fixture(tmp_path, monkeypatch):
    """
    Sets up a temporary environment for testing create_lemma_frequency_file.
    - Changes the current directory to a temporary one.
    - Provides a helper function to create the input file (ldt-corpus.txt) and run the SUT.
    """
    monkeypatch.chdir(tmp_path)

    def _setup_and_run(corpus_content: str):
        """Helper to write the corpus, run the function, and load the output module."""
        (tmp_path / "ldt-corpus.txt").write_text(corpus_content, encoding="utf-8")
        create_lemma_frequency_file()
        output_path = tmp_path / "lemmas.py"

        if not output_path.exists():
            return None

        # Dynamically load the generated .py file as a module
        spec = importlib.util.spec_from_file_location("lemmas", output_path)

        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {output_path}")

        lemmas_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lemmas_mod)
        return lemmas_mod

    return _setup_and_run


def test_create_training_corpus_xseg_does_not_carry_over_between_files(
    mock_dependencies_buffer, monkeypatch
):
    """
    Verify that an XSEG at the end of one file does not bleed into the start of the next file.
    """
    # GIVEN: A config that points to two valid XML files

    mock_config_get = Mock(return_value="\n".join(FAKE_XML_PATHS))
    monkeypatch.setattr("configparser.ConfigParser.get", mock_config_get)

    # WHEN: The corpus creation is run
    create_training_corpus("fake_config.ini")

    # THEN: The output should be correct for both files without cross-contamination
    written_content = mock_dependencies_buffer.getvalue()
    expected_output = """WordOne\tn.-.s.-.-.-.m.n.-\tword
.\tu.-.-.-.-.-.-.-.-\tPERIOD1

WordTwo\tn.-.s.-.-.-.f.n.-\tword
.\tu.-.-.-.-.-.-.-.-\tPERIOD1

"""
    assert written_content == expected_output


def test_create_training_corpus_handles_missing_config_key(
    mock_dependencies_buffer, monkeypatch, capsys
):
    """
    Verify the script warns the user and creates an empty corpus if the config is empty.
    """
    # GIVEN: The config parser returns an empty string for the file list
    mock_config_get = Mock(return_value="")
    monkeypatch.setattr("configparser.ConfigParser.get", mock_config_get)

    # WHEN: The corpus creation is run
    create_training_corpus("fake_config.ini")

    # THEN: A warning should be printed to the console
    captured = capsys.readouterr()
    assert "[!] WARNING: No files listed" in captured.out

    # AND: The output file should be empty
    written_content = mock_dependencies_buffer.getvalue()
    assert written_content == ""


def test_create_training_corpus_skips_nonexistent_file(
    mock_dependencies_buffer, monkeypatch, capsys
):
    """
    Verify the script warns about, skips, and continues after a missing file.
    """
    # GIVEN: A config that lists one valid file and one non-existent file
    paths = ["treebank/file2.xml", "treebank/nonexistent.xml"]
    mock_config_get = Mock(return_value="\n".join(paths))
    monkeypatch.setattr("configparser.ConfigParser.get", mock_config_get)

    # WHEN: The corpus creation is run
    create_training_corpus("fake_config.ini")

    # THEN: A warning about the missing file should be printed
    captured = capsys.readouterr()
    assert (
        "[!] WARNING: File not found, skipping: treebank/nonexistent.xml"
        in captured.out
    )

    # AND: The output file should contain content ONLY from the valid file
    written_content = mock_dependencies_buffer.getvalue()
    expected_output = expected_output = (
        expected_output
    ) = """WordTwo\tn.-.s.-.-.-.f.n.-\tword
.\tu.-.-.-.-.-.-.-.-\tPERIOD1

"""
    assert written_content == expected_output


def test_create_training_corpus_postag_is_correctly_dotted(
    mock_dependencies_buffer, monkeypatch
):
    """
    Regression test to ensure the postag string is correctly formatted
    with dots between each character. This test verifies the assignment of
    `postag`.
    """
    # GIVEN: A config that points to an XML file with a standard postag
    test_file_path = "treebank/postag_test.xml"
    mock_config_get = Mock(return_value=test_file_path)
    monkeypatch.setattr("configparser.ConfigParser.get", mock_config_get)

    # WHEN: The corpus creation is run
    create_training_corpus("fake_config.ini")

    # THEN: The postag in the output file should have dots inserted between characters
    written_content = mock_dependencies_buffer.getvalue()
    # The postag "v1spia---" should become "v.1.s.p.i.a.-.-.-"
    expected_output = """amo\tv.1.s.p.i.a.-.-.-\tamo
.\tu.-.-.-.-.-.-.-.-\tPERIOD1

"""
    assert written_content == expected_output


def test_create_training_corpus_with_relative_path(
    tmp_path, monkeypatch, mock_output_file_only
):
    """
    Verify that a relative path in corpus.ini is correctly resolved
    from the current working directory.
    """
    # GIVEN: A temporary directory structure with a config and an XML file
    # tmp_path is a pytest fixture providing a temporary directory Path object
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    xml_file = data_dir / "test.xml"
    xml_file.write_text(XML_FOR_PATH_TESTS, encoding="utf-8")

    config_file = tmp_path / "test.ini"
    relative_path_str = os.path.join("data", "test.xml")  # "data/test.xml"
    config_file.write_text(f"[source_files]\nxml_files = {relative_path_str}")

    # GIVEN: We change the current directory to the temporary location
    monkeypatch.chdir(tmp_path)

    # WHEN: The corpus creation is run (it will read real temp files)
    create_training_corpus("test.ini")

    # THEN: The output file should contain the content from the XML file
    written_content = mock_output_file_only.getvalue()
    expected_output = """PathTest\tn.-.s.-.-.-.n.n.-\tpath
.\tu.-.-.-.-.-.-.-.-\tPERIOD1

"""
    assert written_content == expected_output


def test_create_training_corpus_with_absolute_path(tmp_path, mock_output_file_only):
    """
    Verify that an absolute path in corpus.ini is correctly resolved.
    """
    # GIVEN: A temporary directory structure with a config and an XML file
    xml_file = tmp_path / "test.xml"
    xml_file.write_text(XML_FOR_PATH_TESTS, encoding="utf-8")

    config_file = tmp_path / "test.ini"
    # Use the full, absolute path to the temporary XML file
    absolute_path_str = str(xml_file.resolve())
    config_file.write_text(f"[source_files]\nxml_files = {absolute_path_str}")

    # WHEN: The corpus creation is run from any directory
    create_training_corpus(str(config_file.resolve()))

    # THEN: The output file should contain the content from the XML file
    written_content = mock_output_file_only.getvalue()
    expected_output = """PathTest\tn.-.s.-.-.-.n.n.-\tpath
.\tu.-.-.-.-.-.-.-.-\tPERIOD1

"""
    assert written_content == expected_output


def test_create_lexicon_and_endings_data_correctly_rejects_short_common_endings(
    lexicon_data_prep_runner,
):
    """
    Confirms the script's correct, high-precision behavior.
    It should REJECT common endings like 'ōrum' and 'ī' because they are too
    short to meet the strict `-3` stem requirement.
    """
    macrons_data = """virorum\tn-p---mg-\tvir\tvi^ro_rum
viri\tn-p---mg-\tvir\tvi^ri_
"""
    endings = lexicon_data_prep_runner(macrons_data)
    assert "'n-p---mg-': []" in endings


def test_create_lexicon_and_endings_data_accepts_a_long_unambiguous_ending(
    lexicon_data_prep_runner,
):
    """
    Confirms that the script accepts a long ending that meets the criteria.
    The test uses 'coniuramentorum', which produces the accented form
    'coniūrāmentōrum'. The ending 'āmentōrum' is identified because it's
    long enough to leave a stem of more than 3 characters.
    """
    macrons_data = (
        "coniuramentorum\tn-p---ng-\tconiuramentum\tconiu_ra_mento_rum\n"
        "coniuramentorum\tn-p---ng-\tconiuramentum\tconiu_ra_mento_rum\n"
    )
    endings = lexicon_data_prep_runner(macrons_data)
    assert "'a_mento_rum'" in endings


def test_create_lexicon_and_endings_data_ending_is_irrelevant_due_to_frequency_tie(
    lexicon_data_prep_runner,
):
    """Tests the frequency tie-breaker: an ending is only relevant if its macronized
    frequency is strictly greater than its unmacronized counterpart."""
    macrons_data = (
        "legio\tn-s---fn-\tlegio\tle^gi^o_\n"  # -> legiō (x1)
        "legio\tn-s---fn-\tlegio\tlegio\n"  # -> legio (x1, fake form for the test)
    )
    endings = lexicon_data_prep_runner(macrons_data)
    assert "'o_'" not in endings


def test_create_lexicon_and_endings_data_ending_is_irrelevant_due_to_short_word_length(
    lexicon_data_prep_runner,
):
    """Words with 4 or fewer chars should not produce any endings."""
    macrons_data = "aqua\tn-s---fn-\taqua\ta^qua\n"
    endings = lexicon_data_prep_runner(macrons_data)
    assert "'n-s---fn-': []" in endings


def test_create_training_corpus_warns_on_missing_xml(corpus_test_env, capsys):
    setup_files, _ = corpus_test_env
    ini_content = "[source_files]\nxml_files = missing_file.xml\n"
    setup_files(ini_content, {})
    create_training_corpus("test_corpus.ini")
    captured = capsys.readouterr()
    assert "[!] WARNING: File not found, skipping: missing_file.xml" in captured.out


def test_create_training_corpus_warns_on_malformed_xml(corpus_test_env, capsys):
    setup_files, _ = corpus_test_env
    ini_content = "[source_files]\nxml_files = bad.xml\n"
    xml_files = {"bad.xml": "<sentence><word>unclosed</sentence>"}
    setup_files(ini_content, xml_files)
    create_training_corpus("test_corpus.ini")
    captured = capsys.readouterr()
    assert "[!] WARNING: XML could not be parsed, skipping: bad.xml" in captured.out


def test_create_training_corpus_no_xml_files_produces_warning_and_no_file(
    corpus_test_env, capsys
):
    """
    Verifies that when no XML files are specified in the config, the script
    prints a warning and does not create an output file.
    """
    # GIVEN: A config file with an empty list of XML files
    setup_files, tmp_path = corpus_test_env
    ini_content = "[source_files]\nxml_files =\n"
    setup_files(ini_content, {})

    # WHEN: The corpus creation function is called
    create_training_corpus("test_corpus.ini")

    # THEN: A warning should be printed to standard output
    captured = capsys.readouterr()
    assert "[!] WARNING: No files listed" in captured.out
    assert "No corpus will be generated" in captured.out

    # AND: The output file should not have been created
    output_file = tmp_path / "ldt-corpus.txt"
    assert not output_file.exists()


def test_create_training_corpus_handles_missing_supplement(corpus_test_env):
    """If supplement is missing, script should not crash and produce an empty corpus."""
    setup_files, tmp_path = corpus_test_env
    ini_content = "[source_files]\nxml_files =\n"
    setup_files(ini_content, {}, supplement_content=None)

    # If there are no XML files, the supplement open() is never reached
    # and the script finishes gracefully.
    create_training_corpus("test_corpus.ini")

    output_file = tmp_path / "ldt-corpus.txt"
    assert not output_file.exists()  # The file isn't even created if there's no data


def test_create_training_corpus_cleans_lemmas_with_hashes_and_numbers(corpus_test_env):
    setup_files, tmp_path = corpus_test_env
    xml_content = """
    <sentence id="1">
        <word id="34" form="bonorum" lemma="bonum#1" postag="n-p---ng-" head="33" relation="ATR" />
    </sentence>
    """
    ini_content = "[source_files]\nxml_files = test.xml\n"
    setup_files(ini_content, {"test.xml": xml_content})
    create_training_corpus("test_corpus.ini")
    output = (tmp_path / "ldt-corpus.txt").read_text(encoding="utf-8")
    assert "\tbonum\n" in output


def test_create_lemma_frequency_file_counts_unique_entries(lemma_freq_runner):
    """
    Tests that basic counts for unique entries are calculated correctly.
    """
    # GIVEN: A corpus with a few unique lines.
    corpus_data = (
        "bonorum\tn.-.p.-.-.-.n.g.-\tbonum\n"
        "interfectus\tt.-.s.r.p.p.m.n.-\tinterficio\n"
    )

    # WHEN: The lemma frequency file is created
    lemmas_module = lemma_freq_runner(corpus_data)

    # THEN: All three dictionaries should reflect the simple 1-to-1 counts.
    assert lemmas_module.lemma_frequency == {"bonum": 1, "interficio": 1}
    assert lemmas_module.word_lemma_freq == {
        ("bonorum", "bonum"): 1,
        ("interfectus", "interficio"): 1,
    }
    assert lemmas_module.wordform_to_corpus_lemmas == {
        "bonorum": ["bonum"],
        "interfectus": ["interficio"],
    }


def test_create_lemma_frequency_file_handles_ambiguous_wordform(lemma_freq_runner):
    """
    Tests that a single wordform mapping to multiple lemmas is handled correctly.
    """
    # GIVEN: The wordform 'est' maps to two different lemmas.
    corpus_data = """est\tv.3.s.p.i.a.-.-.-\tedo
est\tv.3.s.p.i.a.-.-.-\tsum
"""

    # WHEN: The lemma frequency file is created
    lemmas_module = lemma_freq_runner(corpus_data)

    # THEN: The wordform_to_corpus_lemmas should contain a list with both lemmas.
    # The other dicts should have separate entries for each combination.
    assert lemmas_module.lemma_frequency == {"edo": 1, "sum": 1}
    assert lemmas_module.word_lemma_freq == {("est", "edo"): 1, ("est", "sum"): 1}

    # Sort for deterministic comparison
    lemmas_for_est = sorted(lemmas_module.wordform_to_corpus_lemmas["est"])
    assert lemmas_for_est == ["edo", "sum"]


def test_create_lemma_frequency_file_increments_repeated_pair(lemma_freq_runner):
    """
    Tests that counters are correctly incremented for a repeated (word, lemma) pair.
    """
    # GIVEN: The exact same line appears twice.
    corpus_data = """bonorum\ta.-.p.-.-.-.m.g.-\tbonus
bonorum\ta.-.p.-.-.-.m.g.-\tbonus
"""

    # WHEN: The lemma frequency file is created
    lemmas_module = lemma_freq_runner(corpus_data)

    # THEN: The frequency counts should be 2, and the lemma list should not have duplicates.
    assert lemmas_module.lemma_frequency == {"bonus": 2}
    assert lemmas_module.word_lemma_freq == {("bonorum", "bonus"): 2}
    assert lemmas_module.wordform_to_corpus_lemmas == {
        "bonorum": ["bonus"]
    }  # Uniqueness is preserved


def test_create_lemma_frequency_file_ignores_blank_lines(lemma_freq_runner):
    """
    Tests that blank lines in the input corpus are safely ignored.
    """
    # GIVEN: A corpus file containing blank lines.
    corpus_data = (
        "habet\tv.3.s.p.i.a.-.-.-\thabeo\n"
        "\n"
        "\n"
        "tunicam\tn.-.s.-.-.-.f.a.-\ttunica\n"
    )

    # WHEN: The lemma frequency file is created
    lemmas_module = lemma_freq_runner(corpus_data)

    # THEN: The output should be the same as if the blank lines were not present.
    assert lemmas_module.lemma_frequency == {"habeo": 1, "tunica": 1}
    assert (
        len(lemmas_module.lemma_frequency) == 2
    )  # Confirms only two entries were processed


def test_create_lemma_frequency_file_empty_input_file(lemma_freq_runner):
    """
    Tests that an empty input corpus results in empty dictionaries.
    """
    # GIVEN: An empty ldt-corpus.txt file
    corpus_data = ""

    # WHEN: The lemma frequency file is created
    lemmas_module = lemma_freq_runner(corpus_data)

    # THEN: The resulting file should define three empty dictionaries
    assert lemmas_module.lemma_frequency == {}
    assert lemmas_module.word_lemma_freq == {}
    assert lemmas_module.wordform_to_corpus_lemmas == {}


def test_create_training_corpus_xseg_carries_over_skipped_token(corpus_test_env):
    """
    Tests that an xsegment "waits" for the next valid token if an
    intermediate token is skipped.
    """
    # GIVEN: An XML structure with a prefix, a skippable token, and a main word.
    setup_files, tmp_path = corpus_test_env
    xml_content = """
    <sentence id="1">
        <word id="1" form="ve" lemma="other" postag="---------" head="2" relation="XSEG" />
        <word id="2" form="|" lemma="punc1" postag="_" head="1" relation="AuxX" />
        <word id="3" form="rum" lemma="verum1" postag="c--------" head="0" relation="COORD" />
    </sentence>
    """
    ini_content = "[source_files]\nxml_files = test.xml\n"
    setup_files(ini_content, {"test.xml": xml_content})

    # WHEN: The corpus creation is run
    create_training_corpus("test_corpus.ini")
    output = (tmp_path / "ldt-corpus.txt").read_text(encoding="utf-8")

    # THEN: The prefix should correctly attach to the main word, and no line
    # should be generated for the skipped token.
    expected_line = "verum\tc.-.-.-.-.-.-.-.-\tverum\n"
    assert expected_line in output
    assert "ve|\t" not in output  # Verifies incorrect combination didn't happen
    assert "|\t" not in output  # Verifies skipped token produced no output


def test_create_training_corpus_explicitly_skips_token(corpus_test_env):
    """
    Confirms that tokens with form='|' or postag='_' are ignored and
    do not generate any output.
    """
    # GIVEN: An XML structure with various tokens that should be skipped.
    setup_files, tmp_path = corpus_test_env
    xml_content = """
    <sentence id="1">
        <word id="1" form="GoodWord1" lemma="good" postag="a-s---fn-" head="0" relation="PRED" />
        <word id="2" form="|" lemma="punc1" postag="u--------" head="1" relation="AuxX" />
        <word id="3" form="GoodWord2" lemma="good" postag="a-s---mn-" head="0" relation="PRED" />
        <word id="4" form="BadWord" lemma="bad" postag="_" head="3" relation="ATR" />
        <word id="5" form="GoodWord3" lemma="good" postag="a-s---nn-" head="0" relation="PRED" />
    </sentence>
    """
    ini_content = "[source_files]\nxml_files = test.xml\n"
    setup_files(ini_content, {"test.xml": xml_content})

    # WHEN: The corpus creation is run
    create_training_corpus("test_corpus.ini")
    output = (tmp_path / "ldt-corpus.txt").read_text(encoding="utf-8")

    # THEN: Only the "GoodWord" lines should be present in the output.
    assert "GoodWord1" in output
    assert "GoodWord2" in output
    assert "GoodWord3" in output
    assert "|" not in output
    assert "BadWord" not in output


def test_create_training_corpus_handles_space_in_lemma(corpus_test_env):
    """
    Tests lemma cleaning for spaces, e.g., 'res publica'.
    NOTE: A manual search of the target Latin treebank data has confirmed that
    this specific pattern (a lemma attribute containing a space) does not
    currently exist.
    """
    # GIVEN: An XML structure with a multi-word lemma.
    setup_files, tmp_path = corpus_test_env
    xml_content = """
    <sentence id="1">
        <word id="1" form="reipublicae" lemma="res publica" postag="n-s---fg-" head="0" relation="ATR"/>
    </sentence>
    """
    ini_content = "[source_files]\nxml_files = test.xml\n"
    setup_files(ini_content, {"test.xml": xml_content})

    # WHEN: The corpus creation is run
    create_training_corpus("test_corpus.ini")
    output = (tmp_path / "ldt-corpus.txt").read_text(encoding="utf-8")

    # THEN: The space in the lemma should be replaced with a '+'.
    expected_line = "reipublicae\tn.-.s.-.-.-.f.g.-\tres+publica\n"
    assert expected_line in output


def test_create_training_corpus_xseg_state_does_not_leak_between_sentences(
    corpus_test_env,
):
    """
    Confirms that the state (`xsegment`) is correctly reset between sentences,
    preventing a prefix from one sentence from attaching to a word in the next.
    """
    setup_files, tmp_path = corpus_test_env
    # GIVEN: An XML file with two sentences.
    # The first sentence ENDS with a dangling prefix.
    # The second sentence begins with a normal word.
    xml_content = """
    <sentence id="201">
        <word id="1" form="some_word" lemma="some_lemma" postag="n-s---mn-" head="0" relation="PRED"/>
        <word id="2" form="com" lemma="other" postag="---------" head="3" relation="XSEG"/>
    </sentence>
    <sentence id="202">
        <word id="1" form="parandos" lemma="comparo1" postag="t-pfppma-" head="0" relation="OBJ"/>
    </sentence>
    """
    ini_content = "[source_files]\nxml_files = test.xml\n"
    setup_files(ini_content, {"test.xml": xml_content})

    # WHEN: The corpus creation is run
    create_training_corpus("test_corpus.ini")
    output = (tmp_path / "ldt-corpus.txt").read_text(encoding="utf-8")

    # THEN: The output should be clean. The dangling "com" should be discarded,
    # and "parandos" should appear on its own without any prefix.

    # 1. The normal word from the second sentence MUST be present and correct.
    expected_correct_line = "parandos\tt.-.p.f.p.p.m.a.-\tcomparo\n"
    assert expected_correct_line in output

    # 2. The combined word MUST NOT be present.
    buggy_line = "comparandos\tt.-.p.f.p.p.m.a.-\tcomparo\n"
    assert buggy_line not in output

    # 3. The dangling prefix from the first sentence should not be printed by itself.
    assert "com\t" not in output


def test_create_training_corpus_handles_que_enclitic_correctly(corpus_test_env):
    """
    Tests the scenario: lemma="que1", relation="XSEG", and head == id + 1.
    The enclitic 'que' should be correctly appended to the following word.
    """
    setup_files, tmp_path = corpus_test_env
    xml_content = """
    <sentence id="1">
        <word id="1" form="sua" lemma="suus1" postag="a-p---nn-" head="5" relation="SBJ"/>
        <word id="2" form="que" lemma="que1" postag="c--------" head="3" relation="XSEG"/>
        <word id="3" form="quoi" lemma="qui1" postag="p-s---nd-" head="5" relation="OBJ"/>
    </sentence>
    """
    ini_content = "[source_files]\nxml_files = test.xml\n"
    setup_files(ini_content, {"test.xml": xml_content})
    create_training_corpus("test_corpus.ini")
    output = (tmp_path / "ldt-corpus.txt").read_text(encoding="utf-8")

    # The 'que' should attach to 'quoi', not be a separate word
    assert "sua\ta.-.p.-.-.-.n.n.-\tsuus\n" in output
    assert "quoique\tp.-.s.-.-.-.n.d.-\tqui\n" in output
    assert "\nque\t" not in output
    assert "\nquoi\t" not in output


def test_create_training_corpus_ignores_que_enclitic_with_wrong_head(corpus_test_env):
    """
    Tests the scenario: lemma="que1", relation="XSEG", but head != id + 1.
    The 'que' token should be treated as a normal word, not an enclitic.
    NOTE There seem to be no cases in the treebank
    """
    setup_files, tmp_path = corpus_test_env
    xml_content = """
    <sentence id="1">
        <word id="1" form="sua" lemma="suus1" postag="a-p---nn-" head="5" relation="SBJ"/>
        <word id="2" form="que" lemma="que1" postag="c--------" head="1" relation="XSEG"/>
        <word id="3" form="quoi" lemma="qui1" postag="p-s---nd-" head="5" relation="OBJ"/>
    </sentence>
    """
    ini_content = "[source_files]\nxml_files = test.xml\n"
    setup_files(ini_content, {"test.xml": xml_content})
    create_training_corpus("test_corpus.ini")
    output = (tmp_path / "ldt-corpus.txt").read_text(encoding="utf-8")

    # All three words should be processed independently
    assert "sua\ta.-.p.-.-.-.n.n.-\tsuus\n" in output
    assert "que\tc.-.-.-.-.-.-.-.-\tque\n" in output
    assert "quoi\tp.-.s.-.-.-.n.d.-\tqui\n" in output
    assert "quoique" not in output


def test_create_training_corpus_ignores_que_with_non_xseg_relation(corpus_test_env):
    """
    Tests the scenario: lemma="que1", but relation is not "XSEG".
    The 'que' token should be treated as a normal word.
    """
    setup_files, tmp_path = corpus_test_env
    xml_content = """
    <sentence id="1">
        <word id="30" form="ora" lemma="os1" postag="n-p---nn-" head="31" relation="SBJ_CO"/>
        <word id="31" form="que" lemma="que1" postag="c--------" head="27" relation="COORD"/>
        <word id="32" form="voltus" lemma="vultus1" postag="n-s---mn-" head="31" relation="SBJ_CO"/>
    </sentence>
    """
    ini_content = "[source_files]\nxml_files = test.xml\n"
    setup_files(ini_content, {"test.xml": xml_content})
    create_training_corpus("test_corpus.ini")
    output = (tmp_path / "ldt-corpus.txt").read_text(encoding="utf-8")

    # All three words should be processed independently
    assert "ora\tn.-.p.-.-.-.n.n.-\tos\n" in output
    assert "que\tc.-.-.-.-.-.-.-.-\tque\n" in output
    assert "voltus\tn.-.s.-.-.-.m.n.-\tvultus\n" in output


def test_create_training_corpus_handles_ne_enclitic_abbreviation_correctly(
    corpus_test_env,
):
    """
    Tests the scenario: lemma="ne1", relation="XSEG", and head == id + 1.
    The enclitic 'ne' should be correctly appended to the following word.
    """
    setup_files, tmp_path = corpus_test_env
    xml_content = """
    <sentence id="1">
        <word id="1" form="Habebamus" lemma="habeo1" postag="v1piia---" head="0" relation="PRED"/>
        <word id="2" form="nc" lemma="ne1" postag="---------" head="3" relation="XSEG"/>
        <word id="3" form="tu" lemma="tunc1" postag="d--------" head="1" relation="ADV"/>
        <word id="4" form="hominem" lemma="homo1" postag="n-s---ma-" head="1" relation="OBJ"/>
    </sentence>
    """
    ini_content = "[source_files]\nxml_files = test.xml\n"
    setup_files(ini_content, {"test.xml": xml_content})
    create_training_corpus("test_corpus.ini")
    output = (tmp_path / "ldt-corpus.txt").read_text(encoding="utf-8")
    
    assert "Habebamus\tv.1.p.i.i.a.-.-.-\thabeo\n" in output
    # Crucially, verify that 'tu' + 'nc' were combined into 'tunc'
    assert "tunc\td.-.-.-.-.-.-.-.-\ttunc\n" in output
    assert "hominem\tn.-.s.-.-.-.m.a.-\thomo\n" in output
    # Verify that the intermediate forms were not processed as separate words.
    assert "\nnc\t" not in output
    assert "\ntu\t" not in output


def test_create_training_corpus_ignores_ne_enclitic_with_wrong_head(corpus_test_env):
    """
    Tests the scenario: lemma="ne1", relation="XSEG", but head != id + 1.
    The 'ne' token should be treated as a normal word, not an enclitic.
    NOTE There seem to be no cases in the treebank
    """
    setup_files, tmp_path = corpus_test_env
    xml_content = """
    <sentence id="1">
        <word id="2" form="nc" lemma="ne1" postag="---------" head="1" relation="XSEG"/>
        <word id="3" form="tu" lemma="tunc1" postag="d--------" head="1" relation="ADV"/>
    </sentence>
    """
    ini_content = "[source_files]\nxml_files = test.xml\n"
    setup_files(ini_content, {"test.xml": xml_content})
    create_training_corpus("test_corpus.ini")
    output = (tmp_path / "ldt-corpus.txt").read_text(encoding="utf-8")

    assert "nc\t-.-.-.-.-.-.-.-.-\tne\n" in output
    assert "tu\td.-.-.-.-.-.-.-.-\ttunc\n" in output
    assert "tunc\t" not in output


def test_create_training_corpus_ignores_ne_with_non_xseg_relation(corpus_test_env):
    """
    Tests the scenario: lemma="ne1", but relation is not "XSEG".
    The 'ne' token should be treated as a normal word.
    """
    setup_files, tmp_path = corpus_test_env
    xml_content = """
    <sentence id="1">
        <word id="18" form="que" lemma="que1" postag="c--------" head="14" relation="COORD"/>
        <word id="19" form="ne" lemma="ne1" postag="d--------" head="20" relation="AuxZ"/>
        <word id="20" form="auctoritas" lemma="auctoritas1" postag="n-s---fn-" head="18" relation="SBJ_CO"/>
    </sentence>
    """
    ini_content = "[source_files]\nxml_files = test.xml\n"
    setup_files(ini_content, {"test.xml": xml_content})
    create_training_corpus("test_corpus.ini")
    output = (tmp_path / "ldt-corpus.txt").read_text(encoding="utf-8")

    assert "que\tc.-.-.-.-.-.-.-.-\tque\n" in output
    assert "ne\td.-.-.-.-.-.-.-.-\tne\n" in output
    assert "auctoritas\tn.-.s.-.-.-.f.n.-\tauctoritas\n" in output


def test_create_training_corpus_handles_other_prefix_correctly(corpus_test_env):
    """
    Tests the scenario: lemma="other", relation="XSEG", and head == id + 1.
    The prefix should be correctly prepended to the following word.
    """
    setup_files, tmp_path = corpus_test_env
    xml_content = """
    <sentence id="1">
        <word id="4" form="ob" lemma="other" postag="p--------" head="5" relation="XSEG"/>
        <word id="5" form="tulerat" lemma="offertor1" postag="v3slia---" head="2" relation="ADV"/>
    </sentence>
    """
    ini_content = "[source_files]\nxml_files = test.xml\n"
    setup_files(ini_content, {"test.xml": xml_content})
    create_training_corpus("test_corpus.ini")
    output = (tmp_path / "ldt-corpus.txt").read_text(encoding="utf-8")

    assert "obtulerat\tv.3.s.l.i.a.-.-.-\toffertor\n" in output
    assert "\nob\t" not in output


def test_create_training_corpus_ignores_other_prefix_with_wrong_head(corpus_test_env):
    """
    Tests the scenario: lemma="other", relation="XSEG", but head != id + 1.
    The 'cohor' token should be treated as a normal word, not a prefix.
    """
    setup_files, tmp_path = corpus_test_env
    xml_content = """
    <sentence id="1">
        <word id="7" form="cohor" lemma="other" postag="v--------" head="9" relation="XSEG"/>
        <word id="8" form="-" lemma="hyphen1" postag="u--------" head="9" relation="XSEG"/>
        <word id="9" form="tatus" lemma="cohortor1" postag="v-srppmn-" head="18" relation="ADV"/>
    </sentence>
    """
    ini_content = "[source_files]\nxml_files = test.xml\n"
    setup_files(ini_content, {"test.xml": xml_content})
    create_training_corpus("test_corpus.ini")
    output = (tmp_path / "ldt-corpus.txt").read_text(encoding="utf-8")

    # Both words should be processed independently
    assert "cohor\tv.-.-.-.-.-.-.-.-\tother\n" in output
    assert "tatus\tv.-.s.r.p.p.m.n.-\tcohortor\n" in output
    assert "cohortatus" not in output


def test_create_training_corpus_ignores_other_with_non_xseg_relation(corpus_test_env):
    """
    Tests the scenario: lemma="other", but relation is not "XSEG".
    The token should be treated as a normal word.
    """
    setup_files, tmp_path = corpus_test_env
    xml_content = """
    <sentence id="516">
        <word id="1" form="ma" lemma="other" postag="n--------" head="3" relation="AuxZ"/>
        <word id="2" form="deia" lemma="other" postag="n--------" head="0" relation="AuxY_ExD_OBJ"/>
    </sentence>
    """
    ini_content = "[source_files]\nxml_files = test.xml\n"
    setup_files(ini_content, {"test.xml": xml_content})
    create_training_corpus("test_corpus.ini")
    output = (tmp_path / "ldt-corpus.txt").read_text(encoding="utf-8")

    assert "ma\tn.-.-.-.-.-.-.-.-\tother\n" in output
    assert "deia\tn.-.-.-.-.-.-.-.-\tother\n" in output
    assert "madeia" not in output
