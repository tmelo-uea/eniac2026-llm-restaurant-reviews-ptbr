"""
gemini_api.py -- Geracao de respostas com Gemini 3.1 (Google) via API oficial.

Acesso: SDK google-generativeai (declarado em requirements.txt). Chave lida de
GEMINI_API_KEY (nunca embutir segredo no codigo).

    pip install google-generativeai
    export GEMINI_API_KEY="..."
    python gemini_api.py --persona persona_2 --shot few_shot --comment "..."
"""
from __future__ import annotations

import os
import sys
import argparse

from _common import GenParams, load_system_prompt, build_user_message

MODEL_ID = "gemini-3.1-pro"
PROMPT_DIR = "gemini"


def generate(system_prompt: str, comment: str, params: GenParams | None = None) -> str:
    params = params or GenParams()
    try:
        import google.generativeai as genai
    except ImportError:
        sys.exit("Instale o SDK:  pip install google-generativeai")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("Defina GEMINI_API_KEY no ambiente.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_ID, system_instruction=system_prompt)
    resp = model.generate_content(
        build_user_message(comment),
        generation_config=genai.types.GenerationConfig(
            temperature=params.temperature,
            top_p=params.top_p,
            max_output_tokens=params.max_output_tokens,
        ),
    )
    return (resp.text or "").strip()


def main():
    ap = argparse.ArgumentParser(description="Gemini 3.1 via API oficial.")
    ap.add_argument("--persona", required=True, choices=["persona_1", "persona_2"])
    ap.add_argument("--shot", required=True, choices=["zero_shot", "few_shot"])
    ap.add_argument("--comment", required=True)
    args = ap.parse_args()

    system_prompt = load_system_prompt(PROMPT_DIR, args.persona, args.shot)
    print(generate(system_prompt, args.comment))


if __name__ == "__main__":
    main()
