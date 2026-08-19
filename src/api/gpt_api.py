from __future__ import annotations

import os
import sys
import argparse

from _common import GenParams, load_system_prompt, build_user_message

MODEL_ID = "gpt-5.5"
PROMPT_DIR = "gpt"


def generate(system_prompt: str, comment: str, params: GenParams | None = None) -> str:
    params = params or GenParams()
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("Instale o SDK:  pip install openai")

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Defina OPENAI_API_KEY no ambiente.")

    client = OpenAI()
    resp = client.responses.create(
        model=MODEL_ID,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_message(comment)},
        ],
        temperature=params.temperature,
        top_p=params.top_p,
        max_output_tokens=params.max_output_tokens,
    )
    return (resp.output_text or "").strip()


def main():
    ap = argparse.ArgumentParser(description="GPT-5.5 via OpenAI API.")
    ap.add_argument("--persona", required=True, choices=["persona_1", "persona_2"])
    ap.add_argument("--shot", required=True, choices=["zero_shot", "few_shot"])
    ap.add_argument("--comment", required=True)
    args = ap.parse_args()

    system_prompt = load_system_prompt(PROMPT_DIR, args.persona, args.shot)
    print(generate(system_prompt, args.comment))


if __name__ == "__main__":
    main()
