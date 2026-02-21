#!/bin/env python

import sys
import re
import fitz  # PyMuPDF
import colorsys
import ollama
import json
import random
import argparse

import nltk
from nltk.corpus import words
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('words')
nltk.download('punkt_tab')
english_vocab = set(words.words())

from wordfreq import zipf_frequency

from colors import color, red, blue, green, yellow

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Set, TypedDict

from ollama import GenerateResponse
from pydantic import BaseModel

OLLAMA_URL: str = "http://localhost:11434/api/generate"
OLLAMA_MODEL: str = "qwen3:8b"

def is_natural_language(text:str, verbose:bool = False) -> bool:
    """
    Returns True if text looks like natural language,
    False if it looks like a math equation.
    Assumes input is a single line string.
    """

    tokens = word_tokenize(text)
    if not tokens:
        return False

    word_tokens = [t for t in tokens if t.isalpha()]
    word_count = len(word_tokens)
    english_word_tokens = [w for w in word_tokens if zipf_frequency(w, 'en') > 0]
    english_word_count = len(english_word_tokens)

    if not word_tokens:
        return False

    english_ratio = english_word_count / word_count
    english_pseudocount, non_english_pseudocount = 1,1
    english_ratio_bayesian = \
        (english_pseudocount + english_word_count) / \
        (english_pseudocount + non_english_pseudocount + word_count)

    if verbose:
        print(f"-----------------------------------------")
        print(f"input: {text}")
        print(f"  word_tokens: {' '.join(word_tokens)}")
        print(f"  english_word_tokens: {' '.join(english_word_tokens)}")
        print(f"  word_count: {word_count}")
        print(f"  english_word_count: {english_word_count}")
        print(f"  frequentist: {english_ratio}")
        print(f"  bayesian: {english_ratio_bayesian}")
        print(f"")

    return english_ratio_bayesian > 0.7


assert is_natural_language("ing, in-context retrieval, length extrapolation, and long-context understanding. Building on these",True)
assert is_natural_language("results, we also develop hybrid architectures that strategically combine Gated DeltaNet layers with",True)
assert is_natural_language("sliding window attention or Mamba2 layers, further enhancing both training efﬁciency and model",True)

assert not is_natural_language("St = St−1 + vtk⊺",True)
assert not is_natural_language("t ∈Rdv×dk,",True)
assert not is_natural_language("ot = Stqt ∈Rdv",True)
assert not is_natural_language("i=1",True)
# assert not is_natural_language("vi",True)
assert not is_natural_language("[t]ki⊺",True)
assert not is_natural_language("[t] ∈Rdv×dk,",True)


def extract_text(pdf_path: str) -> List[str]:
    doc: fitz.Document = fitz.open(pdf_path)
    text_parts: List[str] = []

    for page in doc:
        text_parts.append(page.get_text())

    doc.close()
    return text_parts


class Concept(BaseModel):
    synonyms: List[str]

class Concepts(BaseModel):
    concepts: List[Concept]

MAX_CONCEPTS = 48

def extract_concepts(text: str, page:int = 0, charlimit:int = sys.maxsize) -> List[Set[str]]:
    print("input text:")
    print(text[:charlimit])
    prompt: str = (
        "Extract all concepts and their acronyms, such as the name of methods, models, "
        "algorithms, datasets, theorems, systems, etc. that are the key topics of the paper, "
        "in the decreasing order of importance.\n"
        "Synonyms must be grouped together as a single concept. "
        "For example, Recurrent Neural Network and RNN are synonyms. \n"
        "Return them in the required JSON schema.\n"
        f"\n\n{text[:charlimit]}"
    )

    response: GenerateResponse = ollama.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        format=Concepts.model_json_schema(),
        stream=False,
    )

    parsed: Concepts = Concepts.model_validate_json(response.response)

    def remove_substrings(synonyms:set[str]) -> set[str]:
        for s1 in synonyms:
            if any([ (s1.lower() in s2.lower()) for s2 in synonyms ]):
                return remove_substrings(synonyms - set([s1]))
        return synonyms

    concepts = [remove_substrings(set(concept.synonyms)) for concept in parsed.concepts]

    print(yellow(f"*** extracted concepts in page {page} ***"))
    for i, concept in enumerate(concepts):
        if i < 48:
            print(" "+", ".join(sorted(concept)))
        else:
            print(" "+color(", ".join(sorted(concept)), fg="gray"))

    return concepts[:MAX_CONCEPTS]


def merge_concepts(concepts_over_pages: List[List[Set[str]]]) -> List[Set[str]]:
    merged: List[Set[str]] = []

    for concepts_per_page in concepts_over_pages:
        for concept in concepts_per_page:
            for merged_concept in merged:
                if concept & merged_concept:
                    merged_concept |= concept
                    break
            else:
                merged.append(concept)

    return merged


def extract_concepts_many(texts: List[str], workers: int = 4) -> List[Set[str]]:
    print(f"{OLLAMA_MODEL} is thinking in parallel ...")

    conceptss: List[List[Set[str]]] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(extract_concepts, text, page) for page, text in enumerate(texts)]

        for future in as_completed(futures):
            conceptss.append(future.result())

    print("done!")

    concepts: List[Set[str]] = merge_concepts(conceptss)

    concepts: List[Set[str]] = ask_merge_concepts(conceptss)

    print(red(f"*** extracted concepts ***"))
    for i, concept in enumerate(concepts):
        if i < 48:
            print(" "+", ".join(sorted(concept)))
        else:
            print(" "+color(", ".join(sorted(concept)), fg="gray"))

    return concepts


Color = Tuple[float, float, float]


def gen_colors(concepts: List[Set[str]]) -> Dict[str, Color]:
    max_per_ring: int = min(12, len(concepts))
    variations: List[Tuple[float, float]] = [
        (4/5, 1.0),   # full saturation, full value
        (3/5, 1.0),   # lighter (half saturation)
        (2/5, 1.0),   # lighter (half saturation)
        (1/5, 1.0),   # lighter (half saturation)
    ]

    colors: Dict[str, Color] = {}
    n: int = len(concepts)

    for idx, concept in enumerate(concepts):
        ring: int = idx // max_per_ring
        pos: int = idx % max_per_ring

        if ring >= len(variations):
            raise ValueError(f"Too many concepts (max supported: {MAX_CONCEPTS})")

        s, v = variations[ring]

        # Even hue spacing starting at cyan (H=0.5)
        hue_step: float = 1.0 / max_per_ring
        h: float = (0.5 + pos * hue_step) % 1.0

        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        for synonym in concept:
            colors[synonym] = (r, g, b)

    return colors



def rect_overlap(a: fitz.Rect, b: fitz.Rect) -> float:
    inter: fitz.Rect = a & b
    if inter.is_empty:
        return 0.0
    return inter.get_area() / min(a.get_area(), b.get_area())


def highlight_pdf(
    in_path: str,
    out_path: str,
    colors: Dict[str, Color]
) -> None:
    doc: fitz.Document = fitz.open(in_path)

    for page in doc:
        used: List[fitz.Rect] = []

        for noun, color in colors.items():
            areas: List[fitz.Rect] = page.search_for(noun)

            for rect in areas:
                rect = rect + (-1, -1, 1, 1)  # slight padding

                # skip if heavy overlap (>40%) with existing highlight
                if any(rect_overlap(rect, u) > 0.4 for u in used):
                    continue

                annot: fitz.Annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=color)
                # annot.set_opacity(0.35)  # reduce stacking darkness
                annot.update()

                used.append(rect)

    doc.save(out_path)
    doc.close()


def parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Highlight detected proper nouns in a PDF paper."
    )

    parser.add_argument("input_pdf", type=str, help="Path to input PDF file")

    parser.add_argument("output_pdf", type=str, help="Path to output highlighted PDF file")

    return parser.parse_args()


def main() -> None:
    args: argparse.Namespace = parse_args()
    texts: List[str] = extract_text(args.input_pdf)
    # concepts: List[Set[str]] = extract_concepts_many(texts)
    text:str = "\n".join(texts)
    text = re.sub(r'\b(acknowledgment|references)\b.*', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = "\n".join([ line for line in text.split("\n") if is_natural_language(line) ])
    print(f"input length: {len(text)}")
    concepts: List[Set[str]] = extract_concepts(text)
    colors: Dict[str, Color] = gen_colors(concepts)
    highlight_pdf(args.input_pdf, args.output_pdf, colors)


if __name__ == "__main__":
    main()
