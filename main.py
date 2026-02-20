import fitz  # PyMuPDF
import requests
import json
import random
from typing import List, Dict, Tuple, Set


OLLAMA_URL: str = "http://localhost:11434/api/generate"
OLLAMA_MODEL: str = "llama3"


def extract_text(pdf_path: str) -> str:
    doc: fitz.Document = fitz.open(pdf_path)
    text_parts: List[str] = []

    for page in doc:
        text_parts.append(page.get_text())

    doc.close()
    return "\n".join(text_parts)


def ask_ollama(text: str) -> List[str]:
    prompt: str = (
        "Extract all proper nouns referring to method names, model names, "
        "algorithm names, dataset names, or named systems in the following paper.\n"
        "Return them as a JSON list of unique strings.\n\n"
        f"{text[:12000]}"
    )

    payload: Dict[str, object] = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    response: requests.Response = requests.post(OLLAMA_URL, json=payload)
    result: Dict[str, object] = response.json()

    raw: str = result["response"]  # type: ignore
    try:
        nouns: List[str] = json.loads(raw)
    except Exception:
        nouns = []

    return list(set(nouns))


def gen_colors(nouns: List[str]) -> Dict[str, Tuple[float, float, float]]:
    colors: Dict[str, Tuple[float, float, float]] = {}
    used: Set[Tuple[float, float, float]] = set()

    for noun in nouns:
        while True:
            color: Tuple[float, float, float] = (
                random.random(),
                random.random(),
                random.random()
            )
            if color not in used:
                used.add(color)
                colors[noun] = color
                break

    return colors


def highlight_pdf(
    in_path: str,
    out_path: str,
    colors: Dict[str, Tuple[float, float, float]]
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
    colors: Dict[str, Tuple[float, float, float]] = gen_colors(nouns)
    highlight_pdf(in_path, out_path, colors)


if __name__ == "__main__":
    input_pdf: str = "paper.pdf"
    output_pdf: str = "paper_highlighted.pdf"
    highlight_paper(input_pdf, output_pdf)
