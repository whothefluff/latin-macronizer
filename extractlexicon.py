#!/usr/bin/python
# -*- coding: utf-8 -*-

import pprint
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import DefaultDict, List, Tuple

import postags

PP = pprint.PrettyPrinter()


def create_lexicon_and_endings_data() -> None:
    """
    - Reads macrons.txt to create rftagger-lexicon.txt.
    - Gathers data on accented forms for each tag.
    - Uses that data to create macronized_endings.py.
    """
    print("Generating lexicon and macronized endings...")
    tag_to_accents = defaultdict(list)

    # First pass: Create the lexicon and gather accent data
    with open("macrons.txt", "r", encoding="utf-8") as macrons_file, open(
        "rftagger-lexicon.txt", "w", encoding="utf-8"
    ) as lexicon_file:
        for line in macrons_file:
            [wordform, tag, lemma, accented] = line.split()
            accented_clean = accented.replace("_^", "").replace("^", "")
            tag_to_accents[tag].append(postags.unicodeaccents(accented_clean))
            if accented[0].isupper():
                wordform = wordform.title()
            tag = ".".join(list(tag))
            lexicon_file.write(f"{wordform}\t{tag}\t{lemma}\n")

    # Second pass: Use the gathered accent data to create the endings file
    with open("macronized_endings.py", "w", encoding="utf-8") as endings_file:
        endings_file.write("tag_to_endings = {\n")
        for tag in sorted(tag_to_accents):
            ending_freqs: DefaultDict[str, int] = defaultdict(int)
            for accented in tag_to_accents[tag]:
                for i in range(1, min(len(accented) - 3, 12)):
                    ending = accented[-i:]
                    ending_freqs[ending] += 1
            relevant_endings = []
            for ending in ending_freqs:
                ending_without_macrons = postags.removemacrons(ending)
                if ending[0] != ending_without_macrons[0] and ending_freqs[
                    ending
                ] > ending_freqs.get(ending_without_macrons, 1):
                    relevant_endings.append(ending)
            cleaned_list = [
                str(postags.escape_macrons(ending))
                for ending in sorted(relevant_endings, key=lambda x: (-len(x), x))
            ]
            endings_file.write(f"  '{tag}': {cleaned_list},\n")
        endings_file.write("}\n")


def create_training_corpus() -> None:
    """
    - Parses a hard-coded list of XML treebank files.
    - Processes the tokens and writes them to ldt-corpus.txt.
    """
    print("Creating training corpus from treebank data...")
    with open("ldt-corpus.txt", "w", encoding="utf-8") as pos_corpus_file:
        xsegment = ""
        xsegmentbehind = ""
        for f in [
            "1999.02.0010",
            "2008.01.0002",
            "2007.01.0001",
            "1999.02.0060",
            "phi0448.phi001.perseus-lat1",
            "phi0620.phi001.perseus-lat1",
            "phi0959.phi006.perseus-lat1",
            "phi0690.phi003.perseus-lat1",
        ]:
            bank = ET.parse(f"treebank_data/v1.6/latin/data/{f}.tb.xml")
            for sentence in bank.getroot():
                for token in sentence.findall("word"):
                    idnum = int(token.get("id", "_"))
                    head = int(token.get("head", "_"))
                    relation = token.get("relation", "_")
                    form = token.get("form", "_")
                    lemma = token.get("lemma", form)
                    postag = token.get("postag", "_")
                    if form != "|" and postag != "" and postag != "_":
                        if (
                            lemma == "other"
                            and relation == "XSEG"
                            and head == idnum + 1
                        ):
                            xsegment = form
                            continue
                        if (
                            (lemma == "que1" or lemma == "ne1")
                            and relation == "XSEG"
                            and head == idnum + 1
                        ):
                            xsegmentbehind = form
                            continue
                        postag = ".".join(list(postag))
                        lemma = (
                            lemma.replace("#", "").replace("1", "").replace(" ", "+")
                        )
                        word = xsegment + form + xsegmentbehind
                        pos_corpus_file.write(f"{word}\t{postag}\t{lemma}\n")
                        xsegment = ""
                        xsegmentbehind = ""
                pos_corpus_file.write(".\tu.-.-.-.-.-.-.-.-\tPERIOD1\n")
                pos_corpus_file.write("\n")
        with open("corpus-supplement.txt", "r", encoding="utf-8") as supplement:
            for line in supplement:
                pos_corpus_file.write(line)


def create_lemma_frequency_file() -> None:
    """
    - Reads the generated ldt-corpus.txt.
    - Calculates lemma and word-form frequencies.
    - Writes the frequency data to lemmas.py.
    """
    print("Creating lemma frequency file...")
    lemma_frequency: DefaultDict[str, int] = defaultdict(int)
    word_lemma_freq: DefaultDict[Tuple[str, str], int] = defaultdict(int)
    wordform_to_corpus_lemmas: DefaultDict[str, List[str]] = defaultdict(list)
    with open("ldt-corpus.txt", "r", encoding="utf-8") as pos_corpus_file:
        for line in pos_corpus_file:
            if "\t" in line:
                [wordform, _, lemma] = line.strip().split("\t")
                lemma_frequency[lemma] += 1
                word_lemma_freq[(wordform, lemma)] += 1
                if lemma not in wordform_to_corpus_lemmas[wordform]:
                    wordform_to_corpus_lemmas[wordform].append(lemma)
    with open("lemmas.py", "w", encoding="utf-8") as lemma_file:
        lemma_file.write(f"lemma_frequency = {PP.pformat(dict(lemma_frequency))}\n")
        lemma_file.write(f"word_lemma_freq = {PP.pformat(dict(word_lemma_freq))}\n")
        lemma_file.write(
            f"wordform_to_corpus_lemmas = {PP.pformat(dict(wordform_to_corpus_lemmas))}\n"
        )


def main() -> None:
    """
    Orchestrates the process of generating all necessary data files
    for the RFTagger training.
    """
    print("Step 1:")
    create_lexicon_and_endings_data()
    print("Step 2:")
    create_training_corpus()
    print("Step 3:")
    create_lemma_frequency_file()
    print("\nAll tasks complete. Required files have been generated.")


if __name__ == "__main__":
    main()
