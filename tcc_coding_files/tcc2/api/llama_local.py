from __future__ import annotations

import sys
import argparse

from _common import GenParams, load_system_prompt, build_user_message

MODEL_TAG = "llama4:maverick"
PROMPT_DIR = "llama"
OLLAMA_HOST = ""


def generate_sdk(system_prompt: str, comment: str, params: GenParams | None = None) -> str:
    params = params or GenParams()
    try:
        import ollama
    except ImportError:
        sys.exit("Instale o SDK:  pip install ollama")

    resp = ollama.chat(
        model=MODEL_TAG,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_message(comment)},
        ],
        options={
            "temperature": params.temperature,
            "top_p": params.top_p,
            "num_predict": params.max_output_tokens,  
        },
    )
    return (resp["message"]["content"] or "").strip()


def main():
    ap = argparse.ArgumentParser(description="Llama 4 Maverick local via Ollama.")
    ap.add_argument("--persona", required=True, choices=["persona_1", "persona_2"])
    ap.add_argument("--shot", required=True, choices=["zero_shot", "few_shot"])
    ap.add_argument("--comment", required=True)
    ap.add_argument("--http", action="store_true", help="Usar HTTP direto em vez do SDK.")
    args = ap.parse_args()

    system_prompt = load_system_prompt(PROMPT_DIR, args.persona, args.shot)

    print(generate_sdk(system_prompt, args.comment))


if __name__ == "__main__":
    main()
