import sqlite3
import string
import unicodedata
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from macronizer import DB_NAME as DB_PATH
from macronizer import SCANSIONS, Macronizer, WordResult

ALLOWED_CHARS = (
    string.ascii_letters
    + string.digits
    + string.whitespace
    + ".,;:?!'\"-()[]—"
    + "æÆœŒ"
)
WHITELIST = set(ALLOWED_CHARS)


class MacronizationRequest(BaseModel):
    text: str
    scan_option: int = 0
    domacronize: bool = True
    alsomaius: bool = False
    performutov: bool = False
    performitoj: bool = False
    clean: bool = False


class MacronizationResponse(BaseModel):
    results: List[WordResult]


@asynccontextmanager
async def macronizer_manager(a: FastAPI) -> AsyncGenerator[Any, Any]:
    """Manages the Macronizer instance and its DB connection for the app's life."""

    db_connection = sqlite3.connect(DB_PATH)
    macronizer_instance = Macronizer(db_connection)
    a.state.macronizer = macronizer_instance
    print("Macronizer initialized with shared DB connection.")

    # Run app
    yield

    # On application shutdown
    db_connection.close()
    print("Shared DB connection closed.")


app = FastAPI(
    title="Latin Macronizer API",
    description="An API for macronizing Latin text with vowel-level ambiguity analysis.",
    lifespan=macronizer_manager,
)


@app.post("/macronize-text", response_model=MacronizationResponse)
async def macronize_text(request: MacronizationRequest) -> MacronizationResponse:
    """
    Macronizes an arbitrary string of Latin text
    """

    try:

        macronizer: Macronizer = app.state.macronizer

        text = extract_text(request)

        macronizer.settext(text)

        if 0 < request.scan_option < len(SCANSIONS):
            automata = SCANSIONS[request.scan_option][1]
            if automata:
                macronizer.scan(automata)

        structured_results = macronizer.tokenization.get_structured_output(
            request.domacronize,
            request.alsomaius,
            request.performutov,
            request.performitoj,
        )

        return MacronizationResponse(results=structured_results)

    except Exception as e:

        raise HTTPException(status_code=500, detail="An internal error occurred") from e


def extract_text(request):
    return sanitize(request.text) if request.clean else request.text


def sanitize(text: str) -> str:
    """
    A comprehensive sanitization pipeline for user-submitted text. It ensures
    the output is clean, valid Latin script suitable for the macronizer.
    """

    # Strip all diacritical marks ('Tĭbĕrĭī' -> 'Tiberii')
    nfd_form = unicodedata.normalize("NFD", text)
    text_no_diacritics = "".join(
        char for char in nfd_form if unicodedata.category(char)[0] != "M"
    )
    # Filter against the whitelist to remove non-Latin characters (removes 'का', '😂', etc.)
    return "".join(char for char in text_no_diacritics if char in WHITELIST)
