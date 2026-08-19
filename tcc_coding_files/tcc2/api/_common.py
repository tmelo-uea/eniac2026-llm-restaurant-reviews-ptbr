from __future__ import annotations

import os
from dataclasses import dataclass

TEMPERATURE = 0.7
TOP_P = 0.9
MAX_OUTPUT_TOKENS = 512

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "prompts")


@dataclass
class GenParams:
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    max_output_tokens: int = MAX_OUTPUT_TOKENS


def load_system_prompt(model_dir: str, persona: str, shot: str) -> str:
    path = os.path.join(PROMPTS_DIR, model_dir, f"{persona}_{shot}.txt")
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt nao encontrado: {path}")
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def build_user_message(comment: str) -> str:
    return f'Comentario do cliente:\n"{comment.strip()}"\n\nResposta:'
