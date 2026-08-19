import argparse
from typing import Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "bdotloh/roberta-base-empathy"


def build_input_text(comment: str, response: str) -> str:
    """
    Constrói um texto único a partir de comentário + resposta.
    Isso é uma adaptação de uso para o domínio do TCC.
    """
    comment = "" if pd.isna(comment) else str(comment).strip()
    response = "" if pd.isna(response) else str(response).strip()

    return (
        "[COMMENT]\n"
        f"{comment}\n\n"
        "[RESPONSE]\n"
        f"{response}"
    )


def load_model(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def predict_raw_scores(
    tokenizer,
    model,
    comment: str,
    response: str,
    max_length: int = 256,
) -> Tuple[float, float]:
    """
    Retorna os scores brutos do checkpoint:
    - índice 0 = Empathy
    - índice 1 = Distress

    O mapeamento Empathy/Distress vem do config.json do próprio modelo.
    """
    text = build_input_text(comment, response)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits.squeeze()

    # Garante vetor com 2 saídas
    if logits.ndim == 0:
        raise ValueError("Saída inesperada do modelo: logits escalar.")
    if logits.shape[-1] < 2:
        raise ValueError(
            f"Saída inesperada do modelo: esperado >=2 dimensões, recebido {tuple(logits.shape)}"
        )

    empathy_raw = float(logits[0].item())
    distress_raw = float(logits[1].item())

    return empathy_raw, distress_raw


def main():
    parser = argparse.ArgumentParser(
        description="Gera scores automáticos brutos de empatia/distress com checkpoint WASSA."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="CSV de entrada com colunas comment e response",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="responses_with_empathy_scores.csv",
        help="CSV de saída com os scores gerados",
    )
    parser.add_argument(
        "--comment_col",
        type=str,
        default="comment",
        help="Nome da coluna de comentário",
    )
    parser.add_argument(
        "--response_col",
        type=str,
        default="response",
        help="Nome da coluna de resposta",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    if args.comment_col not in df.columns:
        raise ValueError(f"Coluna de comentário não encontrada: {args.comment_col}")
    if args.response_col not in df.columns:
        raise ValueError(f"Coluna de resposta não encontrada: {args.response_col}")

    tokenizer, model = load_model(MODEL_NAME)

    empathy_scores = []
    distress_scores = []

    for _, row in df.iterrows():
        empathy_raw, distress_raw = predict_raw_scores(
            tokenizer=tokenizer,
            model=model,
            comment=row[args.comment_col],
            response=row[args.response_col],
        )
        empathy_scores.append(empathy_raw)
        distress_scores.append(distress_raw)

    df = df.copy()
    df["empathy_raw_score"] = empathy_scores
    df["distress_raw_score"] = distress_scores

    df.to_csv(args.output_csv, index=False)
    print(f"Arquivo salvo em: {args.output_csv}")


if __name__ == "__main__":
    main()