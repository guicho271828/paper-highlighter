#!/bin/env python

import fitz  # PyMuPDF
import colorsys
import ollama
import json
import random
import argparse

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Set, TypedDict

from ollama import GenerateResponse
from pydantic import BaseModel

OLLAMA_URL: str = "http://localhost:11434/api/generate"
OLLAMA_MODEL: str = "qwen3:8b"


def extract_text(pdf_path: str) -> List[str]:
    doc: fitz.Document = fitz.open(pdf_path)
    text_parts: List[str] = []

    for page in doc:
        text_parts.append(page.get_text())

    doc.close()
    return text_parts


class Concepts(BaseModel):
    concepts: List[List[str]]


def ask_page(page:int, text: str) -> List[Set[str]]:
    prompt: str = (
        "Extract all proper nouns and their acronyms, such as the name of methods, models, "
        "algorithms, datasets, theorems, systems, etc. in the paper.\n"
        "Synonyms must be grouped together as a single concept.\n"
        "Return them in the required JSON schema."
        f"\n\n{text[:12000]}"
    )

    response: GenerateResponse = ollama.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        format=Concepts.model_json_schema(),
        stream=False,
    )

    parsed: Concepts = Concepts.model_validate_json(response.response)

    concepts = [set(concept) for concept in parsed.concepts]

    print(f"extracted concepts in page {page}:")
    for concept in concepts:
        print(", ".join(sorted(concept)))

    return concepts



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


def ask_ollama(texts: List[str], workers: int = 4) -> List[Set[str]]:
    print(f"{OLLAMA_MODEL} is thinking in parallel ...")

    results: List[List[Set[str]]] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(ask_page, page, text) for page, text in enumerate(texts)]

        for future in as_completed(futures):
            results.append(future.result())

    print("done!")

    concepts: List[Set[str]] = merge_concepts(results)

    print("extracted concepts:")
    for concept in concepts:
        print(", ".join(sorted(concept)))

    return concepts


Color = Tuple[float, float, float]


def gen_colors(concepts: List[Set[str]]) -> Dict[str, Color]:
    max_per_ring: int = min(12, len(concepts))
    variations: List[Tuple[float, float]] = [
        (1.0, 1.0),   # full saturation, full value
        (0.5, 1.0),   # lighter (half saturation)
        (1.0, 0.5),   # darker (half value)
        (0.5, 0.5),   # muted dark
    ]

    colors: Dict[str, Color] = {}
    n: int = len(concepts)

    for idx, concept in enumerate(concepts):
        ring: int = idx // max_per_ring
        pos: int = idx % max_per_ring

        if ring >= len(variations):
            raise ValueError("Too many concepts (max supported: 48)")

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


def highlight_paper(in_path: str, out_path: str) -> None:
    texts: List[str] = extract_text(in_path)
    concepts: List[Set[str]] = ask_ollama(texts)
    colors: Dict[str, Color] = gen_colors(concepts)
    highlight_pdf(in_path, out_path, colors)


def parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Highlight detected proper nouns in a PDF paper."
    )

    parser.add_argument("input_pdf", type=str, help="Path to input PDF file")

    parser.add_argument("output_pdf", type=str, help="Path to output highlighted PDF file")

    return parser.parse_args()


def main() -> None:
    args: argparse.Namespace = parse_args()
    highlight_paper(args.input_pdf, args.output_pdf)


if __name__ == "__main__":
    main()
