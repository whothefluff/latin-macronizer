import builtins
import io
import os
import xml.etree.ElementTree as ET
from unittest.mock import Mock

import pytest

from extractlexicon import create_training_corpus

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
