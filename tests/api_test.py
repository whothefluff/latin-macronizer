# test_api.py
import pytest
from fastapi.testclient import TestClient

from api import sanitize

# A test double for the result of get_structured_output
STRUCTURED_OUTPUT_STUB = [
    {
        "word": "arma",
        "is_word": True,
        "macronized": "arma",
        "uncertainty_mask": 2,
        "candidates": ["arma"],
    },
    {
        "word": " ",
        "is_word": False,
        "macronized": " ",
        "uncertainty_mask": 0,
        "candidates": [],
    },
    {
        "word": "virumque",
        "is_word": True,
        "macronized": "virumque",
        "uncertainty_mask": 0,
        "candidates": [],
    },
]


@pytest.fixture(name="mock_macronizer")
def mock_macronizer_fixture(mocker):
    """
    Fixture to create a mock of the Macronizer instance.
    This mock will be returned by the patched Macronizer class.
    """
    # Create the nested mock structure needed by the API
    mock_instance = mocker.MagicMock()
    mock_instance.tokenization.get_structured_output.return_value = (
        STRUCTURED_OUTPUT_STUB
    )
    return mock_instance


@pytest.fixture(name="client")
def client_fixture(mocker, mock_macronizer):
    """
    A pytest fixture that provides a configured FastAPI TestClient.

    It works by patching the Macronizer class *before* the `api` module is
    imported. This ensures that when FastAPI builds the app, it uses our
    mocked class instead of the real one.
    """
    # The patched class will return our mock_macronizer instance when called.
    mocker.patch("api.Macronizer", return_value=mock_macronizer)

    # Now that the patch is active, we can import the app.
    from api import app  # pylint: disable=import-outside-toplevel

    # Yield the test client for the test function to use.
    with TestClient(app) as test_client:
        yield test_client


class TestMacronizeEndpoint:

    def test_get_structured_output_is_called_with_request_params(
        self, client, mock_macronizer, mocker
    ):
        """
        Ensures request parameters are correctly forwarded to the
        `get_structured_output` method.
        """
        # Arrange
        request_data = {
            "text": "test",
            "domacronize": False,
            "alsomaius": True,
            "performutov": True,
            "performitoj": False,
        }
        spy = mocker.spy(mock_macronizer.tokenization, "get_structured_output")

        # Act
        response = client.post("/macronize-text", json=request_data)

        # Assert
        assert response.status_code == 200
        spy.assert_called_once_with(
            False,  # domacronize
            True,  # alsomaius
            True,  # performutov
            False,  # performitoj
        )
        assert response.json() == {"results": STRUCTURED_OUTPUT_STUB}

    def test_extract_text_returns_user_input_when_clean_is_false(
        self, client, mock_macronizer
    ):
        """
        Ensures that if `clean=False`, the raw user input is passed directly
        to the macronizer's `settext` method.
        """
        # Arrange
        raw_text = "Tĭbĕrĭī, 😂"

        # Act
        client.post("/macronize-text", json={"text": raw_text, "clean": False})

        # Assert
        mock_macronizer.settext.assert_called_once_with(raw_text)

    def test_extract_text_returns_unsanitized_input_by_default(
        self, client, mock_macronizer, mocker
    ):
        """
        Ensures that if `clean` is not specified, the raw user input is passed
        to the macronizer and the sanitize function is NOT called
        """
        # Arrange
        user_text = "some dirty text"
        # Stub the sanitize function to control its output
        sanitize_stub = mocker.patch("api.sanitize")

        # Act
        client.post("/macronize-text", json={"text": user_text})

        # Assert
        sanitize_stub.assert_not_called()
        mock_macronizer.settext.assert_called_once_with(user_text)


class TestSanitizeFunction:

    def test_preserves_valid_latin_text(self):
        """
        Tests that sanitize doesn't alter a clean string containing a wide range
        of valid Latin characters, punctuation, and ligatures.
        """
        text = (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
            "abcdefghijklmnopqrstuvwxyz 0123456789 .,;:?!'\"-()[]—"
        )
        assert sanitize(text) == text

    @pytest.mark.parametrize(
        "dirty_text, expected_clean_text",
        [
            # Test case 1: Macrons and breves
            ("Mārcus tŭllĭus Cĭcĕrō dīxērunt.", "Marcus tullius Cicero dixerunt."),
            # Test case 2: Other diacritics
            ("Äneas fugit Trōiā.", "Aneas fugit Troia."),
            # Test case 3: Non-sensical unicode characters (emojis, Cyrillic)
            ("Latin text with 😂 and some русский.", "Latin text with  and some ."),
            # Test case 4: A mix of everything
            ("Vēnit, vīdit, vīcit! ❤️", "Venit, vidit, vicit! "),
        ],
    )
    def test_removes_various_unwanted_chars(
        self, dirty_text, expected_clean_text
    ):
        """
        Test multiple cleaning scenarios, including macrons, breves,
        other diacritics, and non-Latin characters.
        """
        assert sanitize(dirty_text) == expected_clean_text

    def test_handles_empty_string(self):
        """Tests that an empty string remains an empty string."""
        assert sanitize("") == ""
