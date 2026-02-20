import fitz  # PyMuPDF
import colorsys
import ollama
import json
import random
from typing import List, Dict, Tuple, Set, TypedDict

from ollama import GenerateResponse
from pydantic import BaseModel

OLLAMA_URL: str = "http://localhost:11434/api/generate"
OLLAMA_MODEL: str = "qwen3:8b"


def extract_text(pdf_path: str) -> str:
    doc: fitz.Document = fitz.open(pdf_path)
    text_parts: List[str] = []

    for page in doc:
        text_parts.append(page.get_text())

    doc.close()
    return "\n".join(text_parts)


class NounList(BaseModel):
    nouns: List[str]


def ask_ollama(text: str) -> List[str]:
    prompt: str = (
        "Extract all proper nouns referring to method names, model names, "
        "algorithm names, dataset names, or named systems in the paper.\n"
        "Return them in the required JSON schema."
        f"\n\n{text[:12000]}"
    )

    response: GenerateResponse = ollama.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        format=NounList.model_json_schema(),
        stream=False,
    )

    parsed: NounList = NounList.model_validate_json(response.response)
    nouns: List[str] = list(set(parsed.nouns))

    return nouns


Color = Tuple[float, float, float]


def gen_colors(nouns: List[str]) -> Dict[str, Color]:
    max_per_ring: int = min(12, len(nouns))
    variations: List[Tuple[float, float]] = [
        (1.0, 1.0),   # full saturation, full value
        (0.5, 1.0),   # lighter (half saturation)
        (1.0, 0.5),   # darker (half value)
        (0.5, 0.5),   # muted dark
    ]

    colors: Dict[str, Color] = {}
    n: int = len(nouns)

    for idx, noun in enumerate(nouns):
        ring: int = idx // max_per_ring
        pos: int = idx % max_per_ring

        if ring >= len(variations):
            raise ValueError("Too many nouns (max supported: 48)")

        s, v = variations[ring]

        # Even hue spacing starting at cyan (H=0.5)
        hue_step: float = 1.0 / max_per_ring
        h: float = (0.5 + pos * hue_step) % 1.0

        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors[noun] = (r, g, b)

    return colors


def highlight_pdf(
    in_path: str,
    out_path: str,
    colors: Dict[str, Color]
) -> None:
    doc: fitz.Document = fitz.open(in_path)

    for page in doc:
        for noun, color in colors.items():
            areas: List[fitz.Rect] = page.search_for(noun)

            for rect in areas:
                annot: fitz.Annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=color)
                annot.update()

    doc.save(out_path)
    doc.close()


def highlight_paper(in_path: str, out_path: str) -> None:
    text: str = extract_text(in_path)
    nouns: List[str] = ask_ollama(text)
    colors: Dict[str, Color] = gen_colors(nouns)
    highlight_pdf(in_path, out_path, colors)


if __name__ == "__main__":
    input_pdf: str = "paper.pdf"
    output_pdf: str = "paper_highlighted.pdf"
    highlight_paper(input_pdf, output_pdf)
