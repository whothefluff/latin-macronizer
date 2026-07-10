#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2015-2021 Johan Winge
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import configparser
import os
import re
import sqlite3
import subprocess
from collections import defaultdict
from html import escape
from tempfile import NamedTemporaryFile
from typing import List, Tuple, TypedDict

import postags
from lemmas import lemma_frequency, word_lemma_freq, wordform_to_corpus_lemmas
from macronized_endings import tag_to_endings

ScansionRules = dict[tuple[int, str], tuple[int, str, int]]

DACTYLICH_EXAMETER: ScansionRules = {
    (0, "L"): (1, "", 0),
    (0, "S"): (-1, "", 0),
    (1, "L"): (3, "S", 0),
    (1, "S"): (2, "", 0),
    (2, "L"): (-1, "", 0),
    (2, "S"): (3, "D", 0),
    (3, "L"): (4, "", 0),
    (3, "S"): (-1, "", 0),
    (4, "L"): (6, "S", 0),
    (4, "S"): (5, "", 0),
    (5, "L"): (-1, "", 0),
    (5, "S"): (6, "D", 0),
    (6, "L"): (7, "", 0),
    (6, "S"): (-1, "", 0),
    (7, "L"): (9, "S", 0),
    (7, "S"): (8, "", 0),
    (8, "L"): (-1, "", 0),
    (8, "S"): (9, "D", 0),
    (9, "L"): (10, "", 0),
    (9, "S"): (-1, "", 0),
    (10, "L"): (12, "S", 0),
    (10, "S"): (11, "", 0),
    (11, "L"): (-1, "", 0),
    (11, "S"): (12, "D", 0),
    (12, "L"): (13, "", 0),
    (12, "S"): (-1, "", 0),
    (13, "L"): (15, "S", 0),
    (13, "S"): (14, "", 0),
    (14, "L"): (-1, "", 0),
    (14, "S"): (15, "D", 0),
    (15, "L"): (16, "", 0),
    (15, "S"): (-1, "", 0),
    (16, "L"): (0, "S", 0),
    (16, "S"): (0, "T", 0),
}

DACTYLIC_PENTAMETER: ScansionRules = {
    (0, "L"): (1, "", 0),
    (0, "S"): (-1, "", 0),
    (1, "L"): (3, "S", 0),
    (1, "S"): (2, "", 0),
    (2, "L"): (-1, "", 0),
    (2, "S"): (3, "D", 0),
    (3, "L"): (4, "", 0),
    (3, "S"): (-1, "", 0),
    (4, "L"): (6, "S", 0),
    (4, "S"): (5, "", 0),
    (5, "L"): (-1, "", 0),
    (5, "S"): (6, "D", 0),
    (6, "L"): (7, "-", 0),
    (6, "S"): (-1, "", 0),
    (7, "L"): (8, "", 0),
    (7, "S"): (-1, "", 0),
    (8, "L"): (-1, "", 0),
    (8, "S"): (9, "", 0),
    (9, "L"): (-1, "", 0),
    (9, "S"): (10, "D", 0),
    (10, "L"): (11, "", 0),
    (10, "S"): (-1, "", 0),
    (11, "L"): (-1, "", 0),
    (11, "S"): (12, "", 0),
    (12, "L"): (-1, "", 0),
    (12, "S"): (13, "D", 0),
    (13, "L"): (0, "-", 0),
    (13, "S"): (0, "-", 0),
}

HENDECASYLLABLE: ScansionRules = {
    (0, "L"): (1, "-", 0),
    (0, "S"): (1, "u", 0),
    (1, "L"): (2, "-", 0),
    (1, "S"): (2, "u", 0),
    (2, "L"): (3, "-", 0),
    (2, "S"): (-1, "", 0),
    (3, "L"): (-1, "", 0),
    (3, "S"): (4, "u", 0),
    (4, "L"): (-1, "", 0),
    (4, "S"): (5, "u", 0),
    (5, "L"): (6, "-", 0),
    (5, "S"): (-1, "", 0),
    (6, "L"): (-1, "", 0),
    (6, "S"): (7, "u", 0),
    (7, "L"): (8, "-", 0),
    (7, "S"): (-1, "", 0),
    (8, "L"): (-1, "", 0),
    (8, "S"): (9, "u", 0),
    (9, "L"): (10, "-", 0),
    (9, "S"): (-1, "", 0),
    (10, "L"): (0, "-", 0),
    (10, "S"): (0, "u", 0),
}

IAMBIC_TRIMETER: ScansionRules = {
    (0, "L"): (3, "-", 0),
    (0, "S"): (1, "u", 0),
    (1, "L"): (5, "-|", 0),
    (1, "S"): (2, "u", 0),
    (2, "L"): (5, "-|", 0),
    (2, "S"): (5, "u|", 0),
    (3, "L"): (5, "-|", 0),
    (3, "S"): (4, "u", 0),
    (4, "L"): (-1, "", 0),
    (4, "S"): (5, "u|", 0),
    (5, "L"): (-1, "", 0),
    (5, "S"): (6, "u", 0),
    (6, "L"): (7, "-|", 0),
    (6, "S"): (21, "u", 1),
    (21, "L"): (-1, "", 0),
    (21, "S"): (7, "u|", 0),
    (7, "L"): (10, "-", 0),
    (7, "S"): (8, "u", 0),
    (8, "L"): (12, "-|", 0),
    (8, "S"): (9, "u", 0),
    (9, "L"): (12, "-|", 0),
    (9, "S"): (12, "u|", 0),
    (10, "L"): (12, "-|", 0),
    (10, "S"): (11, "u", 0),
    (11, "L"): (-1, "", 0),
    (11, "S"): (12, "u|", 0),
    (12, "L"): (-1, "", 0),
    (12, "S"): (13, "u", 0),
    (13, "L"): (14, "-|", 0),
    (13, "S"): (-1, "", 0),
    (14, "L"): (17, "-", 0),
    (14, "S"): (15, "u", 0),
    (15, "L"): (19, "-|", 0),
    (15, "S"): (16, "u", 0),
    (16, "L"): (19, "-|", 0),
    (16, "S"): (19, "u|", 0),
    (17, "L"): (19, "-|", 0),
    (17, "S"): (18, "u", 0),
    (18, "L"): (-1, "", 0),
    (18, "S"): (19, "u|", 0),
    (19, "L"): (-1, "", 0),
    (19, "S"): (20, "u", 0),
    (20, "L"): (0, "-", 0),
    (20, "S"): (0, "u", 0),
}

IAMBIC_DIMETER: ScansionRules = {
    (0, "L"): (3, "-", 0),
    (0, "S"): (1, "u", 0),
    (1, "L"): (5, "-|", 0),
    (1, "S"): (2, "u", 0),
    (2, "L"): (5, "-|", 0),
    (2, "S"): (5, "u|", 0),
    (3, "L"): (5, "-|", 0),
    (3, "S"): (4, "u", 0),
    (4, "L"): (-1, "", 0),
    (4, "S"): (5, "u|", 0),
    (5, "L"): (-1, "", 0),
    (5, "S"): (6, "u", 0),
    (6, "L"): (7, "-|", 0),
    (6, "S"): (14, "u", 1),
    (14, "L"): (-1, "", 0),
    (14, "S"): (7, "u|", 0),
    (7, "L"): (10, "-", 0),
    (7, "S"): (8, "u", 0),
    (8, "L"): (12, "-|", 0),
    (8, "S"): (9, "u", 0),
    (9, "L"): (12, "-|", 0),
    (9, "S"): (12, "u|", 0),
    (10, "L"): (12, "-|", 0),
    (10, "S"): (11, "u", 0),
    (11, "L"): (-1, "", 0),
    (11, "S"): (12, "u|", 0),
    (12, "L"): (-1, "", 0),
    (12, "S"): (13, "u", 0),
    (13, "L"): (0, "-", 0),
    (13, "S"): (0, "u", 0),
}

SCANSIONS: List[Tuple[str, List[ScansionRules]]] = [
    ("prose (no scansion)", []),
    ("dactylic hexameters", [DACTYLICH_EXAMETER]),
    ("elegiac distichs", [DACTYLICH_EXAMETER, DACTYLIC_PENTAMETER]),
    ("hendecasyllables", [HENDECASYLLABLE]),
    (
        "iambic trimeter + dimeter",
        [IAMBIC_TRIMETER, IAMBIC_DIMETER],
    ),
]

USE_DB = True
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(SCRIPT_DIR, "macronizer.db")
MORPHEUS_DIR = os.path.join(SCRIPT_DIR, "morpheus")
MACRONS_FILE = os.path.join(SCRIPT_DIR, "macrons.txt")


class MacronizerError(Exception):
    """Base class for exceptions in this module."""


class ExternalDependencyError(MacronizerError):
    """Raised when an external tool (Morpheus, RFTagger) fails."""


class DatabaseError(MacronizerError):
    """Raised for database-related errors (missing table, etc.)."""


class ParsingError(MacronizerError):
    """Raised when converting data from a source fails."""


class InvalidArgumentError(MacronizerError):
    """Raised when a function receives an argument with an invalid value."""


def toascii(txt):
    for source, replacement in [
        ("æ", "ae"),
        ("Æ", "Ae"),
        ("œ", "oe"),
        ("Œ", "Oe"),
        ("ä", "a"),
        ("ë", "e"),
        ("ï", "i"),
        ("ö", "o"),
        ("ü", "u"),
        ("ÿ", "y"),
    ]:
        txt = txt.replace(source, replacement)
    return txt


def touiorthography(txt):
    for source, replacement in [("v", "u"), ("U", "V"), ("j", "i"), ("J", "I")]:
        txt = txt.replace(source, replacement)
    return txt


def clean_lemma(lemma):
    return (
        lemma.replace("#", "")
        .replace("1", "")
        .replace(" ", "+")
        .replace("-", "")
        .replace("^", "")
        .replace("_", "")
    )


class Wordlist:
    def __init__(self, db_conn: sqlite3.Connection):
        self.unknownwords: set[str] = set()  # Unknown to Morpheus
        self.formtolemmas: defaultdict[str, list] = defaultdict(list)
        self.formtoaccenteds: defaultdict[str, list] = defaultdict(list)
        self.formtotaglemmaaccents: defaultdict[str, list] = defaultdict(list)
        if USE_DB:
            self.dbconn = db_conn
            self.dbcursor = self.dbconn.cursor()
        else:
            self.loadwordsfromfile(MACRONS_FILE)

    # enddef

    def reinitializedatabase(self):
        self.dbcursor.execute("DROP TABLE IF EXISTS morpheus")
        self.dbcursor.execute(
            """
            CREATE TABLE morpheus(
                id INTEGER PRIMARY KEY, 
                wordform TEXT NOT NULL, 
                morphtag TEXT, 
                lemma TEXT, 
                accented TEXT, 
                was_crunched INTEGER DEFAULT FALSE,
                UNIQUE(wordform, morphtag, lemma, accented)
            )
        """
        )
        self.loadwordsfromfile(MACRONS_FILE, storeindb=True)
        self.dbcursor.execute(
            "CREATE INDEX morpheus_wordform_index ON morpheus (wordform)"
        )
        self.dbconn.commit()

    # enddef

    def loadwordsfromfile(self, filename, storeindb=False):
        with open(filename, "r", encoding="utf-8") as plaindbfile:
            for line in plaindbfile:
                if line.startswith("#"):
                    continue
                [wordform, morphtag, lemma, accented] = line.split()
                self.addwordparse(wordform, morphtag, lemma, accented)
                if USE_DB and storeindb:
                    self.dbcursor.execute(
                        "INSERT OR IGNORE INTO morpheus (wordform, morphtag, lemma, accented) VALUES (?, ?, ?, ?)",
                        (wordform, morphtag, lemma, accented),
                    )

    # enddef

    def loadwords(self, words):  # Expects a set of lowercase words
        unseenwords = set()
        for word in words:
            if word in self.formtotaglemmaaccents:  # Word is already loaded
                continue
            if not self.loadwordfromdb(word):  # Could not find word in database
                unseenwords.add(word)
        if len(unseenwords) > 0:
            self.crunchwords(
                unseenwords
            )  # Try to parse unseen words with Morpheus, and add result to the database
            for word in unseenwords:
                if not self.loadwordfromdb(word):
                    raise DatabaseError(f"Could not store {word} in the database.")

    # enddef

    def loadwordfromdb(self, word):
        if USE_DB:
            try:
                self.dbcursor.execute(
                    "SELECT wordform, morphtag, lemma, accented FROM morpheus WHERE wordform = ?",
                    (word,),
                )
            except sqlite3.Error as exc:
                raise DatabaseError(
                    f"Query failed '{exc}'. If the database is missing, reset it using --initialize."
                ) from exc
            rows = self.dbcursor.fetchall()
            if len(rows) == 0:
                return False
            for [wordform, morphtag, lemma, accented] in rows:
                self.addwordparse(wordform, morphtag, lemma, accented)
        else:
            self.addwordparse(word, None, None, None)
        return True

    # enddef

    def addwordparse(self, wordform, morphtag, lemma, accented):
        if accented is None:
            self.unknownwords.add(wordform)
        else:
            self.formtolemmas[wordform].append(lemma)
            self.formtoaccenteds[wordform].append(accented.lower())
            self.formtotaglemmaaccents[wordform].append((morphtag, lemma, accented))

    # enddef

    def crunchwords(self, words):
        morphinp = NamedTemporaryFile(
            "w", encoding="utf-8", delete=False, suffix=".txt"
        )
        crunched = NamedTemporaryFile("wb", delete=False, suffix=".txt")
        morphinpfname = morphinp.name
        crunchedfname = crunched.name
        morphinp.close()
        crunched.close()
        try:
            # Write to the input file for Morpheus cruncher
            with open(morphinpfname, "w", encoding="utf-8") as morphinpfile:
                for w in words:
                    word = w.strip()
                    morphinpfile.write(word.lower() + "\n")
                    morphinpfile.write(word.capitalize() + "\n")
            # Resolve Morpheus cruncher path
            cruncher = os.path.join(MORPHEUS_DIR, "bin", "cruncher")
            if not (os.path.isfile(cruncher) and os.access(cruncher, os.X_OK)):
                raise ExternalDependencyError(
                    f"Morpheus cruncher not found or not executable: {cruncher}"
                )
            cmd = [cruncher, "-L"]
            env = os.environ.copy()
            env["MORPHLIB"] = os.path.join(MORPHEUS_DIR, "stemlib")
            # Run cruncher: stdin <- morphinpfname, stdout -> crunchedfname
            with open(morphinpfname, "rb") as fin, open(crunchedfname, "wb") as fout:
                run_external(
                    cmd,
                    stdin=fin,
                    stdout=fout,
                    env=env,
                    timeout=120,
                    tool_name="morpheus cruncher",
                )
            # Read output
            with open(crunchedfname, "r", encoding="utf-8") as crunchedfile:
                morpheus = crunchedfile.read()
            crunchedwordforms = {}
            knownwords = set()
            lines = morpheus.splitlines()
            it = iter(lines)
            try:
                for raw in it:
                    wordform = raw.strip().lower()
                    nls = next(it).strip()
                    crunchedwordforms[wordform] = (
                        crunchedwordforms.get(wordform, "") + nls
                    )
            except StopIteration as e:
                raise ParsingError("Morpheus output parsing failed.") from e
            for wordform, nls in crunchedwordforms.items():
                parses = []
                for nl in nls.split("<NL>"):
                    nl = nl.replace("</NL>", "")
                    nlparts = nl.split()
                    if len(nlparts) > 0:
                        parses += postags.morpheus_to_parses(wordform, nl)
                lemmatagtoaccenteds = defaultdict(list)
                for parse in parses:
                    lemma = clean_lemma(parse[postags.LEMMA])
                    parse[postags.LEMMA] = lemma
                    accented = parse[postags.ACCENTEDFORM]
                    # Work around shortcoming in Morpheus, adding _ in tradu_co_, etc.:
                    if parse[postags.LEMMA].startswith("trans") and accented[3] != "_":
                        accented = accented[:3] + "_" + accented[3:]
                    parse[postags.ACCENTEDFORM] = accented
                    tag = postags.parse_to_ldt(parse)
                    lemmatagtoaccenteds[(lemma, tag)].append(accented)
                if len(lemmatagtoaccenteds) == 0:
                    continue
                knownwords.add(wordform)
                for (lemma, tag), accenteds in lemmatagtoaccenteds.items():
                    # Sometimes there are multiple accented forms; prefer 'volvit' to 'voluit', 'Ju_lius' to 'Iu_lius' etc.:
                    bestaccented = sorted(
                        accenteds,
                        key=lambda x: x.count("v") + x.count("j") + x.count("J"),
                    )[-1]
                    lemmatagtoaccenteds[(lemma, tag)] = bestaccented
                for (lemma, tag), accented in lemmatagtoaccenteds.items():
                    self.dbcursor.execute(
                        "INSERT OR IGNORE INTO morpheus (wordform, morphtag, lemma, accented, was_crunched) VALUES (?, ?, ?, ?, ?)",
                        (wordform, tag, lemma, accented, True),
                    )
            # The remaining were unknown to Morpheus:
            for wordform in words - knownwords:
                self.dbcursor.execute(
                    "INSERT OR IGNORE INTO morpheus (wordform, was_crunched) VALUES (?, ?)",
                    (wordform, True),
                )

            self.dbconn.commit()
        finally:
            # Clean up temp files (even on failure)
            for fn in (morphinpfname, crunchedfname):
                try:
                    os.remove(fn)
                except OSError:
                    pass


prefixeswithshortj = (
    "bij",
    "fidej",
    "Foroj",
    "foroj",
    "ju_rej",
    "multij",
    "praej",
    "quadrij",
    "rej",
    "retroj",
    "se_mij",
    "sesquij",
    "u_nij",
    "introj",
)


class WordResult(TypedDict):
    word: str
    is_word: bool
    macronized: str
    uncertainty_mask: int
    candidates: list[str]


class Token:
    def __init__(self, text):
        self.tag = ""
        self.lemma = ""
        self.accented = [""]
        self.macronized = ""
        self.text = postags.removemacrons(text)
        self.isword = bool(re.match(r"[^\W\d_]", text, flags=re.UNICODE))
        self.isspace = bool(re.match(r"\s", text, flags=re.UNICODE))
        self.hasenclitic = False
        self.isenclitic = False
        self.is_context_start = False
        self.endssentence = False
        self.isunknown = False

    # enddef

    def _apply_case_from_plain(self, candidate_text: str) -> str:
        """
        Projects the casing from the original token text (`self.text`) onto a
        macronized candidate string. Assumes skeletons match.
        Example: self.text="UNUS", candidate_text="ūnus" -> "ŪNUS"
        """
        plain_text = self.text
        # Fallback for safety
        if (
            plain_text.lower()
            != candidate_text.replace("_", "").replace("^", "").lower()
        ):
            return candidate_text

        result_chars = []
        plain_idx = 0
        for char_in_candidate in candidate_text:
            if char_in_candidate in "_^":
                result_chars.append(char_in_candidate)
            else:
                # This is a letter, so we take the corresponding letter
                # from the original plain_text to preserve its case.
                if plain_idx < len(plain_text):
                    result_chars.append(plain_text[plain_idx])
                    plain_idx += 1
                else:
                    # Should not happen if skeletons match, but a safe fallback.
                    result_chars.append(char_in_candidate)
        return "".join(result_chars)

    def split(self, pos, enclitic):
        newtokena = Token(self.text[:-pos])
        newtokenb = Token(self.text[-pos:])
        newtokena.is_context_start = self.is_context_start
        if enclitic:
            newtokena.hasenclitic = True
            newtokenb.isenclitic = True
        return [newtokena, newtokenb]

    # enddef

    def show(self):
        print(
            "\t".join(
                [self.text, self.tag, self.lemma, self.accented[0]],
            ).expandtabs(16)
        )

    # enddef

    def get_macronized(
        self, domacronize: bool, alsomaius: bool, performutov: bool, performitoj: bool
    ) -> str:
        plain = self.text
        if not self.isword:
            return plain
        accented = self.accented[0]
        accented = accented.replace("_^", "").replace("^", "")
        while "__" in accented:
            accented = accented.replace("__", "_")
        # Mark long before consonantal j if requested (excluding known short-j prefixes)
        if domacronize and alsomaius and "j" in accented:
            if not accented.startswith(prefixeswithshortj):
                accented = re.sub("([aeiouy])(j[aeiouy])", r"\1_\2", accented)
        # If we're not adding macrons (no underscores) and not doing u→v or i→j, just return original
        if (
            (not domacronize or "_" not in accented)
            and not performutov
            and not performitoj
        ):
            return plain
        # Enclitic tokens are not macronized (except "ue" when converting u→v)
        if self.isenclitic and not (plain.lower() == "ue" and performutov):
            return plain
        # Normalize both to lowercase for the alignment, then re-apply the original casing at the end.
        plain_original_case = plain
        plain = plain.lower()
        accented_for_align = accented.lower()

        # Skeleton check: compare after removing underscores and normalizing to UI orthography + ASCII
        s_plain = touiorthography(toascii(plain)).lower()
        s_acc = touiorthography(toascii(accented_for_align.replace("_", ""))).lower()
        if s_plain != s_acc:
            # Not the same word skeleton; avoid forcing a dubious alignment
            return plain_original_case

        def inscost(a):
            return 0 if a == "_" else 2

        def subcost(p, a):
            if a == "_":
                return 100  # don't "substitute" underscores
            if (a in "IJij" and p in "IJij") or (a in "UVuv" and p in "UVuv"):
                return 1
            return 2

        def delcost(_):
            return 2

        # Build DP table using the now-consistent lowercase strings
        n = len(plain) + 1
        m = len(accented_for_align) + 1
        distance = [[0 for _ in range(m)] for _ in range(n)]
        for i in range(1, n):
            distance[i][0] = distance[i - 1][0] + delcost(plain[i - 1])
        for j in range(1, m):
            distance[0][j] = distance[0][j - 1] + inscost(accented_for_align[j - 1])
        for i in range(1, n):
            for j in range(1, m):
                if toascii(plain[i - 1]) == toascii(accented_for_align[j - 1]): # Simpler comparison now
                    distance[i][j] = distance[i - 1][j - 1]
                else:
                    rghtcost = distance[i - 1][j] + delcost(plain[i - 1])
                    downcost = distance[i][j - 1] + inscost(accented_for_align[j - 1])
                    diagcost = distance[i - 1][j - 1] + subcost(
                        plain[i - 1], accented_for_align[j - 1]
                    )
                    distance[i][j] = min(rghtcost, diagcost, downcost)
        # Backtrace with explicit flush of remainders
        i = n - 1
        j = m - 1
        result = ""
        while i > 0 and j > 0:
            # Prefer diagonal when ties occur to keep alignment stable
            same = toascii(plain[i - 1]) == toascii(accented_for_align[j - 1])
            diag_needed = distance[i][j] == distance[i - 1][j - 1] + (
                0 if same else subcost(plain[i - 1], accented_for_align[j - 1])
            )
            up_needed = distance[i][j] == distance[i][j - 1] + inscost(accented_for_align[j - 1])
            if diag_needed:
                i -= 1
                j -= 1
                if performutov and accented[j].lower() == "v" and plain_original_case[i] == "u":
                    result = "v" + result
                elif performutov and accented[j].lower() == "v" and plain_original_case[i] == "U":
                    result = "V" + result
                elif performitoj and accented[j].lower() == "j" and plain_original_case[i] == "i":
                    result = "j" + result
                elif performitoj and accented[j].lower() == "j" and plain_original_case[i] == "I":
                    result = "J" + result
                else:
                    # Take character from original text to preserve its case
                    result = plain_original_case[i] + result
            elif up_needed:
                j -= 1
                if domacronize and accented_for_align[j] == "_":
                    result = "_" + result
            else:  # Left move
                i -= 1
                result = plain_original_case[i] + result
        # Flush any remaining insertions (underscores) from accented
        while j > 0:
            j -= 1
            if domacronize and accented_for_align[j] == "_":
                result = "_" + result
        # Flush any remaining deletions (characters) from plain
        while i > 0:
            i -= 1
            result = plain_original_case[i] + result
        # Some strange morpheus output (e.g. de_e_recti_) may give an additional _ in the result:
        result = result.replace("__", "_")
        return result

    def macronize(self, domacronize, alsomaius, performutov, performitoj) -> None:
        self.macronized = self.get_macronized(
            domacronize, alsomaius, performutov, performitoj
        )

    def get_structured_output(self, macronized: str) -> WordResult:
        """
        Reads the token's state and generates a structured dictionary.
        """

        if not self.isword:
            return {
                "word": self.text,
                "is_word": self.isword,
                "macronized": macronized,
                "uncertainty_mask": 0,
                "candidates": [],
            }

        final_macronized_text = postags.unicodeaccents(macronized)

        if self.isunknown:
            word_len = len(final_macronized_text)
            return {
                "word": self.text,
                "is_word": self.isword,
                "macronized": final_macronized_text,
                "uncertainty_mask": (1 << word_len) - 1 if word_len > 0 else 0,
                "candidates": [],
            }

        uncertainty_mask = 0
        # Create the unique list for display, preserving the original order of first appearance
        # and correctly cased
        unique_candidates = list(
            dict.fromkeys(
                postags.unicodeaccents(self._apply_case_from_plain(c.replace("^", "")))
                for c in self.accented
            )
        )
        # Only enter if there is a real, mappable ambiguity to calculate.
        if len(unique_candidates) > 1:
            best_guess = unique_candidates[0]
            base_skeleton = postags.removemacrons(best_guess).lower()
            comparable_candidates = [
                cand
                for cand in unique_candidates
                if postags.removemacrons(cand).lower() == base_skeleton
            ]
            # The core logic: only calculate a non-zero mask if there is more
            # than one candidate with the *exact same word skeleton*.
            if len(comparable_candidates) > 1:
                for char_index, char in enumerate(base_skeleton):
                    # Only check for ambiguity on vowels.
                    if char in "aeiouy":
                        # Collect all vowel states (e.g., {'i', 'ī'}) from candidates at this specific character index.
                        macron_states_for_vowel = {
                            cand[char_index] for cand in comparable_candidates
                        }
                        if len(macron_states_for_vowel) > 1:
                            uncertainty_mask |= 1 << char_index

        return {
            "word": self.text,
            "is_word": True,
            "macronized": final_macronized_text,
            "uncertainty_mask": uncertainty_mask,
            "candidates": unique_candidates[1:],
        }


class Tokenization:

    REPRIORITIZE_PENALTY = 1
    MUTA_CUM_LIQUIDA_PENALTY = 1
    DIAERESIS_PENALTY = 2
    NO_SYNEZIS_PENALTY = 2  # in the context s or ng + u + vowel
    SYNEZIS_PENALTY = 3
    HIATUS_PENALTY = 3
    ENCLITIC_SPECIAL_FORMS = [
        "nec",
        "neque",
        "necnon",
        "seque",
        "seseque",
        "quique",
        "mecumque",
        "tecumque",
        "secumque",
    ]
    # Enclitic suffixes that splittokens knows how to split off.
    SPLITTABLE_ENCLITIC_SUFFIXES_3 = ("que",)
    SPLITTABLE_ENCLITIC_SUFFIXES_2 = ("ve", "ue", "ne", "st")

    def __init__(self, text):
        self.tokens = []
        possiblesentenceend = False
        # This combination correctly identifies a new verse in a poem, a new paragraph, or the start of a text
        sentencehasended = True  # Tracks sentence-ending punctuation (.;:?!).
        follows_newline = (
            True  # Tracks if the last non-space token contained a newline.
        )

        for chunk in re.findall(r"[^\W\d_]+|\s+|[^\w\s]+|[\d_]+", text, re.UNICODE):
            token = Token(chunk)
            if token.isword:

                if sentencehasended or follows_newline:
                    token.is_context_start = True

                # Reset flags after processing the word
                sentencehasended = False
                follows_newline = False
                possiblesentenceend = len(token.text) > 1

            else:  # Token is not a word (whitespace or punctuation)
                if possiblesentenceend and any(i in token.text for i in ".;:?!"):
                    token.endssentence = True
                    possiblesentenceend = False
                    sentencehasended = True
                # If the current non-word chunk contains a newline, set the flag for the *next* word.
                if "\n" in token.text:
                    follows_newline = True

            self.tokens.append(token)
        self.scannedfeet = []

    # enddef

    def allwordforms(self):
        words = set()
        for token in self.tokens:
            if token.isword:
                words.add(toascii(token.text).lower())
        return words

    # enddef

    dividenda = {
        "nequid": 4,
        "attamen": 5,
        "unusquisque": 7,
        "unaquaeque": 7,
        "unumquodque": 7,
        "uniuscuiusque": 8,
        "uniuscujusque": 8,
        "unicuique": 6,
        "unumquemque": 7,
        "unamquamque": 7,
        "unoquoque": 6,
        "unaquaque": 6,
        "cuiusmodi": 4,
        "cujusmodi": 4,
        "quojusmodi": 4,
        "eiusmodi": 4,
        "ejusmodi": 4,
        "huiuscemodi": 4,
        "hujuscemodi": 4,
        "huiusmodi": 4,
        "hujusmodi": 4,
        "istiusmodi": 4,
        "nullomodo": 4,
        "quodammodo": 4,
        "nudiustertius": 7,
        "nonnisi": 4,
        "plusquam": 4,
        "proculdubio": 5,
        "quamplures": 6,
        "quamprimum": 6,
        "quinetiam": 5,
        "uerumetiam": 5,
        "verumetiam": 5,
        "verumtamen": 5,
        "uerumtamen": 5,
        "paterfamilias": 8,
        "patrisfamilias": 8,
        "patremfamilias": 8,
        "patrifamilias": 8,
        "patrefamilias": 8,
        "patresfamilias": 8,
        "patrumfamilias": 8,
        "patribusfamilias": 8,
        "materfamilias": 8,
        "matrisfamilias": 8,
        "matremfamilias": 8,
        "matrifamilias": 8,
        "matrefamilias": 8,
        "matresfamilias": 8,
        "matrumfamilias": 8,
        "matribusfamilias": 8,
        "respublica": 7,
        "reipublicae": 8,
        "rempublicam": 8,
        "senatusconsultum": 9,
        "senatusconsulto": 8,
        "senatusconsulti": 8,
        "usufructu": 6,
        "usumfructum": 7,
        "ususfructus": 7,
        "supradicti": 5,
        "supradictum": 6,
        "supradictus": 6,
        "supradicto": 5,
        "seipse": 4,
        "seipsa": 4,
        "seipsum": 5,
        "seipsam": 5,
        "seipso": 4,
        "seipsos": 5,
        "seipsas": 5,
        "seipsis": 5,
        "semetipse": 4,
        "semetipsa": 4,
        "semetipsum": 5,
        "semetipsam": 5,
        "semetipso": 4,
        "semetipsos": 5,
        "semetipsas": 5,
        "semetipsis": 5,
        "teipsum": 5,
        "temetipsum": 5,
        "vosmetipsos": 5,
        "idipsum": 5,
    }
    # satisdare, satisdetur, etc

    @staticmethod
    def _skeleton(text: str) -> str:
        """Length-normalized comparable form: no macrons/breves, ui-orthography, ascii, lowercase"""
        return touiorthography(toascii(text.replace("_", "").replace("^", ""))).lower()

    def _morpheus_returned_only_stem(
        self, asciiword: str, stem: str, wordlist: Wordlist
    ) -> bool:
        """Return whether every Morpheus accented form matches the expected stem."""
        accenteds: list[str] | None = wordlist.formtoaccenteds.get(asciiword)
        if not accenteds:
            return False
        stem_skeleton = Tokenization._skeleton(stem)
        return all(
            Tokenization._skeleton(accented) == stem_skeleton for accented in accenteds
        )

    def splittokens(self, wordlist: Wordlist) -> set[str]:
        newwords: set[str] = set()
        newtokens: list[Token] = []
        for oldtoken in self.tokens:

            tobeadded: list[Token] = []

            oldlc: str = oldtoken.text.lower()
            if oldtoken.isword and oldlc != "que":
                # Pre-calculate how and where we might split this word
                split_pos: int = 0
                is_enclitic_split = True
                if oldlc == "nec":
                    split_pos = 1
                elif oldlc == "necnon":
                    pass  # Handled specially below
                elif oldlc in Tokenization.dividenda:
                    split_pos = Tokenization.dividenda[oldlc]
                    is_enclitic_split = False
                elif len(oldlc) > 3 and oldlc.endswith(
                    Tokenization.SPLITTABLE_ENCLITIC_SUFFIXES_3
                ):
                    split_pos = 3
                elif len(oldlc) > 2 and oldlc.endswith(
                    Tokenization.SPLITTABLE_ENCLITIC_SUFFIXES_2
                ):
                    split_pos = 2

                is_unknown_to_morpheus = oldlc in wordlist.unknownwords
                is_forced_split = oldlc in Tokenization.ENCLITIC_SPECIAL_FORMS
                has_only_stem_parses = False
                if (
                    not is_unknown_to_morpheus
                    and not is_forced_split
                    and is_enclitic_split
                    and split_pos > 0
                ):
                    stem = oldlc[:-split_pos]
                    has_only_stem_parses = self._morpheus_returned_only_stem(
                        toascii(oldlc), toascii(stem), wordlist
                    )
                if is_unknown_to_morpheus or is_forced_split or has_only_stem_parses:
                    if oldlc == "necnon":
                        [tempa, tempb] = oldtoken.split(3, False)
                        tobeadded = tempa.split(1, True) + [tempb]
                    elif split_pos > 0:
                        tobeadded = oldtoken.split(split_pos, is_enclitic_split)

            if len(tobeadded) == 0:
                newtokens.append(oldtoken)
            else:
                for part in tobeadded:
                    newwords.add(toascii(part.text).lower())
                    newtokens.append(part)

        self.tokens = newtokens
        return newwords

    def show(self):
        for token in self.tokens[:500]:
            if token.isword:
                token.show()
            if token.endssentence:
                print()
        if len(self.tokens) > 500:
            print("... (truncated) ...")

    # enddef

    def addtags(self, rftagger_dir: str) -> None:
        with NamedTemporaryFile(
            "w+", encoding="utf-8", delete=True
        ) as totaggerfile, NamedTemporaryFile(
            "w+", encoding="utf-8", delete=True
        ) as fromtaggerfile:
            # Write the input data for RFTagger
            savedencliticbearer = None
            for token in self.tokens:
                if not token.isspace:
                    tokentext = token.text
                    if tokentext == tokentext.upper():
                        tokentext = tokentext.lower()
                    if token.hasenclitic:
                        savedencliticbearer = toascii(tokentext)
                        continue
                    totaggerfile.write(toascii(tokentext) + "\n")
                    if token.isenclitic:
                        assert savedencliticbearer is not None
                        totaggerfile.write(savedencliticbearer + "\n")
                        savedencliticbearer = None
                if token.endssentence:
                    totaggerfile.write("\n")

            # Ensure all data is written to disk before the external program tries to read it.
            totaggerfile.flush()

            rftagger_model = os.path.join(
                os.path.dirname(__file__), "rftagger-ldt.model"
            )
            # Resolve rft-annotate path
            rft_annotate = os.path.join(rftagger_dir, "rft-annotate")
            if not (os.path.isfile(rft_annotate) and os.access(rft_annotate, os.X_OK)):
                raise ExternalDependencyError(
                    f"RFTagger 'rft-annotate' not found or not executable: {rft_annotate}"
                )

            cmd = [
                rft_annotate,
                "-s",
                "-q",
                rftagger_model,
                totaggerfile.name,
                fromtaggerfile.name,
            ]
            # Run rft-annotate
            run_external(cmd, tool_name="RFTagger")
            # After the external tool writes to the file, we need to go back to the start of it to read.
            fromtaggerfile.seek(0)

            (taggedenclititoken, enclitictag) = (None, None)
            line = None
            for token in self.tokens:
                if not token.isspace:
                    try:
                        if token.hasenclitic:
                            line = fromtaggerfile.readline().strip()
                            assert line and line.count("\t") == 1
                            (taggedenclititoken, enclitictag) = line.split("\t")
                        if token.isenclitic:
                            assert (
                                taggedenclititoken is not None
                                and enclitictag is not None
                            )
                            (taggedtoken, tag) = (taggedenclititoken, enclitictag)
                        else:
                            line = fromtaggerfile.readline().strip()
                            assert line and line.count("\t") == 1
                            (taggedtoken, tag) = line.split("\t")
                        if token.text == token.text.upper():
                            assert taggedtoken == toascii(token.text.lower())
                        else:
                            assert taggedtoken == toascii(token.text)
                    except AssertionError as exc:
                        raise ParsingError(
                            f"Error: Could not handle tagging data from RFTagger:\n"
                            f"'{'Premature End Of File.' if not line else line}'"
                        ) from exc
                    token.tag = tag.replace(".", "")
                if token.endssentence:
                    line = fromtaggerfile.readline()

    def addlemmas(self, wordlist):

        for token in self.tokens:
            wordform = toascii(token.text)
            best_lemma = "-"
            max_freq = -1
            if wordform in wordform_to_corpus_lemmas:
                for corpus_lemma in wordform_to_corpus_lemmas[wordform]:
                    if word_lemma_freq[(wordform, corpus_lemma)] > max_freq:
                        max_freq = word_lemma_freq[(wordform, corpus_lemma)]
                        best_lemma = corpus_lemma
            elif wordform.lower() in wordlist.formtolemmas:
                for lex_lemma in wordlist.formtolemmas[wordform.lower()]:
                    if lemma_frequency.get(lex_lemma, 0) > max_freq:
                        max_freq = lemma_frequency.get(lex_lemma, 0)
                        best_lemma = lex_lemma
            # endif
            token.lemma = best_lemma

    @staticmethod
    def levenshtein(s1, s2):
        if len(s1) < len(s2):
            # pylint: disable=arguments-out-of-order
            return Tokenization.levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def getaccents(self, wordlist):
        prev_word_was_all_caps = False
        prev_word_was_capitalized = False
        for token in self.tokens:
            if not token.isword:
                continue

            wordform_original_case = toascii(token.text)
            is_all_caps = (
                wordform_original_case.isupper() and len(wordform_original_case) > 1
            )
            is_capitalized = False  # Initialize here for the finally block

            try:
                is_capitalized = wordform_original_case.istitle() or is_all_caps
                wordform = wordform_original_case.lower()

                tag = token.tag
                lemma = token.lemma
                if token.isenclitic:
                    token.accented = (
                        ["ve"] if token.text.lower() == "ue" else [token.text.lower()]
                    )
                    continue
                if token.text.lower() == "ne" and token.hasenclitic:  # Not nēque...
                    token.accented = ["ne"]
                    continue

                if wordform in wordlist.formtotaglemmaaccents:
                    # ALL CAPS mid-sentence is an error/unknown, unless it's part of a sequence.
                    if (
                        is_all_caps
                        and not token.is_context_start
                        and not prev_word_was_all_caps
                    ):
                        token.accented = [token.text]
                        token.isunknown = True
                        continue

                    raw = []
                    for lextag, lexlemma, accented in wordlist.formtotaglemmaaccents[
                        wordform
                    ]:
                        # Asymmetrical case filter:
                        # If the input token is lowercase, do NOT consider Titlecase lemmas (no proper noun from lowercase).
                        if not is_capitalized and lexlemma.istitle():
                            continue

                        # Rank by case; forgive capitalized words at context start (text/verse/sentence) or in sequence.
                        is_case_mismatch = is_capitalized != lexlemma.istitle()
                        is_forgivable = (token.is_context_start and is_capitalized) or (
                            is_capitalized and prev_word_was_capitalized
                        )
                        casedist = 0 if (not is_case_mismatch or is_forgivable) else 1

                        # Tie-breaker to prefer proper nouns (score 0) over common nouns (score 1) when casedist is tied
                        case_preference_penalty = 0 if lexlemma.istitle() else 1

                        tagdist = postags.tag_distance(tag, lextag)
                        lemdist = Tokenization.levenshtein(lemma, lexlemma)
                        raw.append((casedist, case_preference_penalty, tagdist, lemdist, accented, lexlemma))

                    # Mid-context Titlecase token with no Titlecase lemma candidates:
                    # very likely an editorial capitalization -> mark as unknown.
                    if (
                        token.text.istitle()
                        and not token.is_context_start
                        and not prev_word_was_capitalized
                        and not any(lexlemma.istitle() for _, _, _, _, _, lexlemma in raw)
                    ):
                        token.accented = [token.text]
                        token.isunknown = True
                        continue

                    raw.sort()
                    token.accented = []
                    if raw:
                        best_casedist = raw[0][0]
                        for casedist, _, _, _, accented, _ in raw:
                            if (
                                casedist == best_casedist
                                and accented not in token.accented
                            ):
                                token.accented.append(accented)
                    else:
                        token.accented = [token.text]
                        token.isunknown = True
                        continue

                else:
                    # Unknown word, but attempt to mark vowels in ending:
                    # To-do: Better support for different capitalization and orthography
                    token.accented = [token.text]
                    if any(i in token.text for i in "aeiouyAEIOUY"):
                        for accented_ending in tag_to_endings.get(tag, []):
                            plain_ending = accented_ending.replace("_", "").replace(
                                "^", ""
                            )
                            if wordform.endswith(plain_ending):
                                token.accented = [
                                    wordform[: -len(plain_ending)] + accented_ending
                                ]
                                break
                        token.isunknown = True
            finally:
                prev_word_was_all_caps = is_all_caps
                prev_word_was_capitalized = is_capitalized

    def scanverses(self, meterautomatons):
        """Try to scan the text according to meterautomatons. This function will, for each token,
        reconsider the order of the accented forms given by the getaccents function, by finding
        a likely combination of accented forms that make the verses scan."""

        def allvowelsambiguous(accented):
            """Generate accented forms for unknown words"""
            accented = re.sub("([aeiouy])", "\\1_^", accented)
            accented = accented.replace("qu_^", "qu")
            accented = re.sub(r"_\^(ns|nf|nct)", "_\\1", accented)
            accented = re.sub(r"_\^([bcdfgjklmnpqrstv]{2,}|[xz])", "\\1", accented)
            accented = re.sub(r"_\^m$", "m", accented)
            return accented

        # enddef

        def separate_ambiguous_vowels(accenteds):
            """
            If a vowel is ambiguous (_^), generate separate accented forms, one for each possible combination.
            Input: ['ba_^ce_^]
            Output: ['bace', 'ba_ce', 'bace_', 'ba_ce_']
            """
            accented_modifications = {
                "nescio_": "nescio_^",
                "u_ni_us": "u_ni_^us",
                "illi_us": "illi_^us",
                "ipsi_us": "ipsi_^us",
                "alteri_us": "alteri_^us",
            }
            new_accenteds = []
            for accented in accenteds:
                accented = accented_modifications.get(accented, accented)
                parts = accented.split("_^")
                for variant in range(1 << len(parts) - 1):
                    new_accented = []
                    for bit_pos, part in enumerate(parts):
                        new_accented.append(part)
                        if 1 << bit_pos & variant:
                            new_accented.append("_")
                    new_accenteds.append("".join(new_accented))
            return new_accenteds

        # enddef

        def segmentaccented(accented):
            """Split an accented form into a list of individual vowel phonemes and consonant clusters"""
            if accented == "hoc":  # Ad hoc fix. (Haha!)
                return ["o", "cc"]
            text = (
                accented.lower()
                .replace("qu", "q")
                .replace("x", "cs")
                .replace("z", "ds")
                .replace("+", "^")
                + "#"
            )
            segments = []
            segmentstart = 0
            pos = 0
            while True:
                if (
                    text[pos : pos + 2] in ["ae", "au", "ei", "eu", "oe"]
                    and text[pos + 2] not in "_^+"
                ):
                    pos += 2
                elif text[pos] in "aeiouy":
                    pos += 1
                    while text[pos] in "_^+":
                        pos += 1
                else:
                    while text[pos] not in "aeiouy#":
                        pos += 1
                segment = text[segmentstart:pos].replace("h", "")
                if segment != "":
                    segments.append(segment)
                if text[pos] == "#":
                    break
                segmentstart = pos
            return segments

        # enddef

        def possiblescans(accentedcandidates, followingsegment):
            """
            A form with marked vowel lengths can be scanned differently, considering
            muta cum liquida, diphthong vs. diaeresis, elision, etc.

            input: followingsegment is one of ["V", "C", "CC", "#"]
            returns: [(penalty, scansion, accented), ...]
            """
            scans = []
            # Iterate over top-level candidates in the priority order given by getaccents
            for cand_idx, base_accented in enumerate(accentedcandidates):
                # Expand only this candidate’s ambiguous vowels (_^) into per-variant accenteds
                variants = separate_ambiguous_vowels([base_accented])
                # Apply reprioritization penalty per candidate, not per variant
                basepenalty = 0 if cand_idx == 0 else self.REPRIORITIZE_PENALTY

                for accented in variants:
                    segments = segmentaccented(accented)
                    segments.append(followingsegment)

                    temps = [(basepenalty, "")]
                    for i, thisseg in enumerate(segments):
                        prevseg = "#" if i == 0 else segments[i - 1]
                        nextseg = "#" if i == len(segments) - 1 else segments[i + 1]
                        if i == 0 and not thisseg[0] in "aeiouy":
                            # Skip leading consonant clusters at word start
                            continue
                        news = []
                        for penaltysofar, scansofar in temps:
                            if "_" in thisseg:
                                news.append((penaltysofar, scansofar + "L"))
                            elif thisseg in ["ae", "au", "ei", "oe", "eu"]:
                                news.append((penaltysofar, scansofar + "L"))
                                news.append(
                                    (
                                        penaltysofar + self.DIAERESIS_PENALTY,
                                        scansofar + "VV",
                                    )
                                )
                            elif (
                                (prevseg.endswith("s") or prevseg.endswith("ng"))
                                and thisseg == "u"
                                and nextseg[0] in "aeiouy"
                            ):
                                news.append((penaltysofar, scansofar + "C"))
                                news.append(
                                    (
                                        penaltysofar + self.NO_SYNEZIS_PENALTY,
                                        scansofar + "V",
                                    )
                                )
                            elif thisseg[0] in "ui" and (
                                nextseg[0] in "aeiouy" or prevseg[0] in "aeiouy"
                            ):
                                news.append((penaltysofar, scansofar + "V"))
                                news.append(
                                    (
                                        penaltysofar + self.SYNEZIS_PENALTY,
                                        scansofar + "C",
                                    )
                                )
                            elif thisseg[0] in "aeiouy":
                                news.append((penaltysofar, scansofar + "V"))
                            elif thisseg == "m" and nextseg in ["V", "C", "CC", "#"]:
                                news.append((penaltysofar, scansofar + "M"))
                            elif thisseg == "j" and prevseg != "#":
                                if accented.startswith(prefixeswithshortj):
                                    news.append((penaltysofar, scansofar + "C"))
                                else:
                                    news.append((penaltysofar, scansofar + "CC"))
                            elif thisseg == "V":  # next word begins with vowel
                                if scansofar.endswith("V") or scansofar.endswith("L"):
                                    news.append((penaltysofar, scansofar[:-1]))
                                    news.append(
                                        (penaltysofar + self.HIATUS_PENALTY, scansofar)
                                    )
                                elif scansofar.endswith("M"):
                                    news.append((penaltysofar, scansofar[:-2]))
                                    news.append(
                                        (penaltysofar + self.HIATUS_PENALTY, scansofar)
                                    )
                                else:
                                    news.append((penaltysofar, scansofar))
                            elif thisseg == "#":
                                news.append((penaltysofar, scansofar))
                            elif len(thisseg) == 1:
                                news.append((penaltysofar, scansofar + "C"))
                            elif (
                                len(thisseg) == 2
                                and thisseg[0] in "tpcdbgf"
                                and thisseg[1] in "rl"
                            ):
                                news.append((penaltysofar, scansofar + "C"))
                                news.append(
                                    (
                                        penaltysofar + self.MUTA_CUM_LIQUIDA_PENALTY,
                                        scansofar + "CC",
                                    )
                                )
                            else:
                                news.append((penaltysofar, scansofar + "CC"))
                        temps = news
                    for penalty, scansion in temps:
                        scansion = re.sub("VMC*|VCCC*|LM?C*", "L", scansion)
                        scansion = re.sub("VC?", "S", scansion)
                        scansion = re.sub("^C*", "", scansion)
                        scans.append((penalty, scansion, accented))
            # De-duplicate by scansion, preferring lower penalty (then lexicographically)
            filteredscans = []
            foundscansions = set()
            for penalty, scansion, accented in sorted(scans):
                if scansion not in foundscansions:
                    filteredscans.append((penalty, scansion, accented))
                    foundscansions.add(scansion)
            return filteredscans

        def scanverse(verse, automaton):
            """Input: The "verse" is a complicated list of the format
            [(tokenindex, [(penalty, scansion, accented), (penalty, scansion, accented), ...]), ...]
            For example: [(0, [(0, 'L', 'in')]), (2, [(0, 'SL', 'no^va_'), (1, 'SS', 'no^va')]), ...]
            It returns a tuple such as ([(0, 'in'), (2, 'no^va'), (4, 'fe^rt'), ...], 'DDSSDS')
            """

            def scanverserecurse(verse, wordindex, automaton, oldnodeindex):
                if wordindex == len(verse):
                    return [], [], 0
                (tokenindex, wordscansions) = verse[wordindex]
                besttail = []
                besttailfeet = []
                besttailpenalty = float("inf")
                for scanpenalty, scansion, accented in wordscansions:
                    nodeindex = oldnodeindex
                    feet = []
                    finished = False
                    meterpenalty = 0
                    for syllable in scansion:
                        (nodeindex, foot, meterpenaltypart) = automaton.get(
                            (nodeindex, syllable), (-1, "", 0)
                        )
                        meterpenalty += meterpenaltypart
                        if nodeindex == 0:
                            finished = True
                        feet.append(foot)
                    if (
                        nodeindex == -1
                        or finished
                        and (nodeindex != 0 or wordindex != len(verse) - 1)
                    ):
                        continue
                    tail, tailfeet, tailpenalty = scanverserecurse(
                        verse, wordindex + 1, automaton, nodeindex
                    )
                    if scanpenalty + meterpenalty + tailpenalty < besttailpenalty:
                        besttail = [(tokenindex, accented)] + tail
                        besttailfeet = feet + tailfeet
                        besttailpenalty = scanpenalty + meterpenalty + tailpenalty
                return besttail, besttailfeet, besttailpenalty

            # enddef
            indexaccentedpairs, feet, _ = scanverserecurse(verse, 0, automaton, 0)
            return indexaccentedpairs, "".join(feet)

        # enddef

        self.scannedfeet = []
        verse = []
        automatonindex = 0
        for index, token in enumerate(self.tokens):
            if token.isword:
                followingtext = ""
                nextindex = index
                while True:
                    nextindex += 1
                    if (
                        nextindex == len(self.tokens)
                        or "\n" in self.tokens[nextindex].text
                    ):
                        break
                    if self.tokens[nextindex].isspace:
                        followingtext += " "
                    elif self.tokens[nextindex].isword:
                        followingtext += self.tokens[nextindex].accented[0]
                        if any(ch in "aeiouy" for ch in followingtext):
                            break
                followingtext = followingtext.lower().replace("h", "")
                if followingtext == "":
                    followingsegment = "#"
                elif re.match(" *[aeiouy]", followingtext):
                    followingsegment = "V"
                elif re.match(
                    " *([bcdfgjklmnpqrstv] *|[tpcdbgf][lr])[aeiouy]", followingtext
                ):
                    followingsegment = "C"
                else:
                    followingsegment = "CC"
                if token.isunknown:
                    token.accented.append(allvowelsambiguous(token.text.lower()))
                verse.append((index, possiblescans(token.accented, followingsegment)))
            if "\n" in token.text or index == len(self.tokens) - 1:
                (accentcorrections, feet) = scanverse(
                    verse, meterautomatons[automatonindex]
                )
                self.scannedfeet.append(feet)
                self.scannedfeet += [""] * (token.text.count("\n") - 1)
                for tokenindex, newaccented in accentcorrections:
                    try:
                        self.tokens[tokenindex].accented.remove(newaccented)
                    except ValueError:
                        pass
                    self.tokens[tokenindex].accented.insert(0, newaccented)
                verse = []
                automatonindex += 1
                if automatonindex == len(meterautomatons):
                    automatonindex = 0

    # enddef

    def macronize(self, domacronize, alsomaius, performutov, performitoj):
        for token in self.tokens:
            token.macronize(domacronize, alsomaius, performutov, performitoj)

    # enddef

    def detokenize(self, markambiguous):
        result = []
        for token in self.tokens:
            if token.isword:
                unicodetext = postags.unicodeaccents(token.macronized)
                if markambiguous:
                    unicodetext = re.sub(
                        r"([āēīōūȳĀĒĪŌŪȲaeiouyAEIOUY])", "<span>\\1</span>", unicodetext
                    )
                    if token.isunknown:
                        unicodetext = f'<span class="unknown">{unicodetext}</span>'
                    elif len(set([x.replace("^", "") for x in token.accented])) > 1:
                        unicodetext = f'<span class="ambig">{unicodetext}</span>'
                    else:
                        unicodetext = f'<span class="auto">{unicodetext}</span>'
                result.append(unicodetext)
            else:
                if markambiguous:
                    result.append(escape(token.macronized))
                else:
                    result.append(token.macronized)
        return "".join(result)

    def get_structured_output(
        self, domacronize: bool, alsomaius: bool, performutov: bool, performitoj: bool
    ) -> list[WordResult]:
        """
        Generates a list of structured data for each token
        """
        result = []
        for token in self.tokens:
            macr = token.get_macronized(
                domacronize, alsomaius, performutov, performitoj
            )
            result.append(token.get_structured_output(macr))
        return result


class Macronizer:
    def __init__(
        self,
        db_conn: sqlite3.Connection,
        config_path: str = os.path.join(SCRIPT_DIR, "config.ini"),
    ):
        config = configparser.ConfigParser()
        config.read(config_path)
        self.rftagger_dir = config.get("paths", "rftagger_dir", fallback="")

        self.wordlist = Wordlist(db_conn)
        self.tokenization = Tokenization("")

    def settext(self, text):
        self.tokenization = Tokenization(text)
        self.wordlist.loadwords(self.tokenization.allwordforms())
        newwordforms = self.tokenization.splittokens(self.wordlist)
        self.wordlist.loadwords(newwordforms)
        self.tokenization.addtags(self.rftagger_dir)
        self.tokenization.addlemmas(self.wordlist)
        self.tokenization.getaccents(self.wordlist)

    # enddef

    def scan(self, automatons):
        self.tokenization.scanverses(automatons)

    # enddef

    def gettext(
        self,
        domacronize=True,
        alsomaius=False,
        performutov=False,
        performitoj=False,
        markambigs=False,
    ):
        self.tokenization.macronize(domacronize, alsomaius, performutov, performitoj)
        return self.tokenization.detokenize(markambigs)

    # enddef

    def macronize(
        self,
        text,
        domacronize=True,
        alsomaius=False,
        performutov=False,
        performitoj=False,
        markambigs=False,
    ):
        self.settext(text)
        return self.gettext(
            domacronize, alsomaius, performutov, performitoj, markambigs
        )

    # enddef


# endclass


def evaluate(goldstandard, macronizedtext):
    if len(goldstandard) != len(macronizedtext):
        raise InvalidArgumentError(
            f"Error: Text mismatch. Gold standard length ({len(goldstandard)}) "
            f"does not match macronized text length ({len(macronizedtext)})."
        )
    vowelcount = 0
    lengthcorrect = 0
    outtext = []
    for a, b in zip(list(goldstandard), list(macronizedtext)):
        plaina = postags.removemacrons(a)
        plainb = postags.removemacrons(b)
        if touiorthography(toascii(plaina)) != touiorthography(toascii(plainb)):
            raise InvalidArgumentError("Error: Text mismatch.")
        if plaina in "AEIOUYaeiouy":
            vowelcount += 1
            if a == b:
                lengthcorrect += 1
        if toascii(touiorthography(a)) == toascii(touiorthography(b)):
            outtext.append(escape(b))
        else:
            outtext.append(f'<span class="wrong">{escape(b)}</span>')
    # If there are no vowels, and the texts matched, accuracy is arguably 100%
    return lengthcorrect / float(vowelcount) if vowelcount else 1.0, "".join(outtext)


def run_external(
    cmd, *, stdin=None, stdout=None, env=None, timeout=120, tool_name=None
):
    """
    Run an external command safely, capturing stderr and converting common failures into ExternalDependencyError.
    """
    try:
        completed = subprocess.run(
            cmd,
            stdin=stdin,
            stdout=stdout,
            stderr=subprocess.PIPE,  # capture for diagnostics
            env=env,
            check=True,
            timeout=timeout,
        )
        return completed
    except FileNotFoundError as exc:
        name = tool_name or (cmd[0] if isinstance(cmd, (list, tuple)) else str(cmd))
        raise ExternalDependencyError(
            f"Required external tool not found: {name}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        name = tool_name or (cmd[0] if isinstance(cmd, (list, tuple)) else str(cmd))
        raise ExternalDependencyError(f"External tool timed out: {name}") from exc
    except subprocess.CalledProcessError as exc:
        name = tool_name or (cmd[0] if isinstance(cmd, (list, tuple)) else str(cmd))
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
        raise ExternalDependencyError(
            f"External tool failed (exit {exc.returncode}): {name}\nStderr:\n{stderr}"
        ) from exc


if __name__ == "__main__":
    print(
        """Library for marking Latin texts with macrons. Copyright 2015-2017 Johan Winge.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Minimal example of usage:
    from macronizer import Macronizer
    macronizer = Macronizer()
    macronizedtext = macronizer.macronize("Iam primum omnium satis constat Troia capta in ceteros saevitum esse Troianos")

Initializing Macronizer() may take a couple of seconds, so if you want
to mark macrons in several strings, you are better off reusing the
same Macronizer object.

The macronizer function takes a couple of optional parameters, which
control in what way the input string is transformed:
    domacronize: mark long vowels; default True
    alsomaius: also mark vowels before consonantic i; default False
    performutov: change consonantic u to v; default False
    performitoj: similarly change i to j; default False
    markambigs: mark up the text in various ways with HTML tags; default False

If you want to transform the same text in different ways, you should use
the separate gettext and settext functions, instead of macronize:
    from macronizer import Macronizer
    macronizer = Macronizer()
    macronizer.settext("Iam primum omnium")
    print(macronizer.gettext())
    print(macronizer.gettext(domacronize=False, performitoj=True))

NOTE: If you are not a developer, you probably want to call the front end
macronize.py instead.
"""
    )
