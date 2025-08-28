import sys

import pytest

import macronize


@pytest.fixture(name="temp_config_file")
def temp_config_file_fixture(tmp_path):
    """
    A fixture that creates a temporary config file with specified content
    and returns its absolute path.
    """

    def _create_config(content: str, filename="test.ini"):
        config_path = tmp_path / filename
        config_path.write_text(content, encoding="utf-8")
        return str(config_path.resolve())

    return _create_config


def test_main_cli_uses_default_config_path_when_unspecified(mocker, monkeypatch):
    """
    Verifies that with no --config arg, the default "config.ini" is passed.
    """
    # Arrange
    # Use --test to prevent reading from stdin and simplify the execution path
    monkeypatch.setattr(sys, "argv", ["macronize.py", "--test"])

    mock_macronizer = mocker.patch("macronize.Macronizer")

    mock_instance = mocker.MagicMock()
    mock_instance.gettext.return_value = "mocked output text"
    mock_macronizer.return_value = mock_instance

    # Act
    macronize.main_cli()

    # Assert
    mock_macronizer.assert_called_once_with("config.ini")


def test_main_cli_uses_custom_config_path_with_relative_path(
    mocker, monkeypatch, temp_config_file, tmp_path
):
    """
    Verifies that a custom, RELATIVE path provided via --config is used.
    """
    # Arrange
    ini_content = "[paths]\nrftagger_dir = /relative/works"
    relative_path = "my_config.ini"
    temp_config_file(ini_content, filename=relative_path)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["macronize.py", "--config", relative_path, "--test"]
    )

    mock_macronizer = mocker.patch("macronize.Macronizer")
    mock_instance = mocker.MagicMock()
    mock_instance.gettext.return_value = "mocked output text"
    mock_macronizer.return_value = mock_instance

    # Act
    macronize.main_cli()

    # Assert
    mock_macronizer.assert_called_once_with(relative_path)


def test_main_cli_uses_custom_config_path_with_absolute_path(
    mocker, monkeypatch, temp_config_file
):
    """
    Verifies that a custom, ABSOLUTE path provided via --config is used.
    """
    # Arrange
    ini_content = "[paths]\nrftagger_dir = /absolute/works"
    absolute_path = temp_config_file(ini_content)

    monkeypatch.setattr(
        sys, "argv", ["macronize.py", "--config", absolute_path, "--test"]
    )

    mock_macronizer = mocker.patch("macronize.Macronizer")
    mock_instance = mocker.MagicMock()
    mock_instance.gettext.return_value = "mocked output text"
    mock_macronizer.return_value = mock_instance

    # Act
    macronize.main_cli()

    # Assert
    mock_macronizer.assert_called_once_with(absolute_path)


def test_main_cli_initialization_does_not_error_on_missing_config_file(
    mocker, monkeypatch, tmp_path
):
    """
    Verifies that if the --config path does NOT exist, the script does NOT crash.
    The front-end should just pass the path along.
    """
    # Arrange
    non_existent_path = str(tmp_path / "ghost.ini")

    monkeypatch.setattr(
        sys, "argv", ["macronize.py", "--config", non_existent_path, "--test"]
    )

    mock_macronizer = mocker.patch("macronize.Macronizer")
    mock_instance = mocker.MagicMock()
    mock_instance.gettext.return_value = "mocked output text"
    mock_macronizer.return_value = mock_instance

    # Act
    macronize.main_cli()

    # Assert
    mock_macronizer.assert_called_once_with(non_existent_path)


def test_main_cgi_initializes_macronizer_with_default_config(mocker, monkeypatch):
    """
    Verifies that main_cgi() instantiates Macronizer with the default path arg.
    """
    # Arrange
    # Mock the necessary CGI environment variable for the function to run.
    monkeypatch.setenv("REQUEST_URI", "/test.cgi")
    # Mock cgi.FieldStorage.
    mock_field_storage = mocker.MagicMock()
    mock_field_storage.getvalue.side_effect = lambda key, default="": {
        "textcontent": "test text",
        "doevaluate": "on",
    }.get(key, default)
    mocker.patch("cgi.FieldStorage", return_value=mock_field_storage)
    # Patch the Macronizer class.
    mock_macronizer = mocker.patch("macronize.Macronizer")
    # Configure the mock instance to prevent downstream errors.
    mock_instance = mocker.MagicMock()
    # The 'evaluate' function needs a string from gettext().
    mock_instance.gettext.return_value = "test text"
    # The HTML rendering needs a string from detokenize().
    mock_instance.tokenization.detokenize.return_value = "mocked output"
    mock_macronizer.return_value = mock_instance
    # Mock print to swallow the HTTP header output.
    mocker.patch("builtins.print")

    # Act
    macronize.main_cgi()

    # Assert
    mock_macronizer.assert_called_once_with()
