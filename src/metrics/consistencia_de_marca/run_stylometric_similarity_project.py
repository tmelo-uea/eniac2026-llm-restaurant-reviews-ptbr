from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from stylometric_similarity_writeprints import WriteprintsBrandSimilarity


BEFORE_BASE_CANDIDATES = [
    "data/expanded/before_llm_release_comments_expanded.csv",
    "before_llm_release_comments_expanded.csv",
    "before_llm_release_comments.csv",
    "before_llm_release.csv",
]

AFTER_BASE_CANDIDATES = [
    "data/expanded/after_llm_release_comments_expanded.csv",
    "after_llm_release_comments_expanded.csv",
    "after_llm_release_comments.csv",
    "after_llm_release.csv",
]


def read_csv_flexible(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    seps = [",", ";"]

    last_error = None
    for enc in encodings:
        for sep in seps:
            try:
                return pd.read_csv(path, encoding=enc, sep=sep)
            except Exception as e:
                last_error = e

    raise RuntimeError(f"Não foi possível ler {path}. Último erro: {last_error}")


def normalize_colname(name: str) -> str:
    return str(name).strip().lower()


def find_existing_file(root: Path, candidates: List[str]) -> Optional[Path]:
    for name in candidates:
        p = root / name
        if p.exists():
            return p
    return None


def detect_text_column(df: pd.DataFrame, preferred_keywords: List[str]) -> Optional[str]:
    cols = list(df.columns)
    norm_map = {c: normalize_colname(c) for c in cols}

    for k in preferred_keywords:
        for c in cols:
            if k in norm_map[c]:
                return c

    for c in cols:
        series = df[c]
        if series.dtype == "object":
            non_null = series.dropna().astype(str).str.strip()
            if not non_null.empty and non_null.str.len().mean() > 10:
                return c

    return None


def find_generated_response_columns(df: pd.DataFrame) -> List[str]:
    cols = list(df.columns)
    result = []

    positive_patterns = [
        "generated",
        "response",
        "reply",
        "resposta",
        "output",
    ]

    negative_patterns = [
        "time",
        "tempo",
        "latency",
        "prompt",
        "comentario_original",
        "original_comment",
        "comment",
        "comentário",
        "comentario",
        "review",
    ]

    for c in cols:
        n = normalize_colname(c)

        if any(neg in n for neg in negative_patterns):
            continue

        if any(pos in n for pos in positive_patterns):
            result.append(c)

    if not result:
        for c in cols:
            n = normalize_colname(c)
            if any(neg in n for neg in negative_patterns):
                continue

            series = df[c]
            if series.dtype == "object":
                non_null = series.dropna().astype(str).str.strip()
                if not non_null.empty and non_null.str.len().mean() > 10:
                    result.append(c)

    seen = set()
    unique_result = []
    for c in result:
        if c not in seen:
            seen.add(c)
            unique_result.append(c)

    return unique_result


def collect_brand_texts(base_csv: Path) -> Tuple[List[str], str]:
    df = read_csv_flexible(base_csv)

    text_col = detect_text_column(
        df,
        preferred_keywords=[
            "resposta",
            "response",
            "comment",
            "coment",
            "review",
            "text",
            "texto",
        ],
    )

    if text_col is None:
        raise RuntimeError(
            f"Não foi possível detectar a coluna textual no corpus base: {base_csv.name}"
        )

    texts = (
        df[text_col]
        .dropna()
        .astype(str)
        .str.strip()
    )
    texts = [t for t in texts if t]

    if not texts:
        raise RuntimeError(f"O corpus base {base_csv.name} não possui textos válidos")

    n_before = len(texts)
    texts = list(dict.fromkeys(texts))
    n_after = len(texts)
    if n_before != n_after:
        print(f"[INFO] Deduplicação do corpus {base_csv.name}: {n_before} → {n_after} textos únicos")

    return texts, text_col


def discover_generated_csvs(project_root: Path) -> List[Path]:
    generated_dir = project_root / "data" / "generated_responses"
    legacy_dir = project_root / "generated_comments"

    files = []
    if generated_dir.exists() and generated_dir.is_dir():
        files.extend(sorted(generated_dir.glob("*.csv")))
    elif legacy_dir.exists() and legacy_dir.is_dir():
        files.extend(sorted(legacy_dir.glob("*.csv")))

    for p in sorted(project_root.glob("*.csv")):
        if "generated" in p.name.lower():
            files.append(p)

    seen = set()
    unique_files = []
    for f in files:
        key = str(f.resolve())
        if key not in seen:
            seen.add(key)
            unique_files.append(f)

    return unique_files


def choose_brand_base_for_generated_file(
    generated_csv: Path,
    before_base: Path,
    after_base: Path,
) -> Path:
    name = generated_csv.name.lower()

    if "before" in name:
        return before_base
    if "after" in name:
        return after_base

    raise RuntimeError(
        f"Não foi possível inferir se o arquivo {generated_csv.name} é before ou after"
    )


def safe_text_list_with_row_ids(series: pd.Series, file_stem: str) -> Tuple[List[str], List[str]]:
    texts = []
    ids = []

    for idx, value in series.items():
        if pd.isna(value):
            continue
        txt = str(value).strip()
        if not txt:
            continue
        texts.append(txt)
        ids.append(f"{file_stem}__row_{idx}")

    return texts, ids


def parse_metadata_from_filename(filename: str) -> dict:
    name = filename.lower()

    release_period = "before" if "before" in name else "after" if "after" in name else "unknown"
    persona = (
        "persona_1" if "persona_1" in name
        else "persona_2" if "persona_2" in name
        else "persona_1" if "persona1" in name
        else "persona_2" if "persona2" in name
        else "unknown"
    )
    prompting = "few_shot" if "few_shot" in name else "zero_shot" if "zero_shot" in name else "unknown"

    return {
        "release_period": release_period,
        "persona": persona,
        "prompting_strategy": prompting,
    }


def run(project_root: Path) -> None:
    results_dir = project_root / "results" / "stylometric_similarity"
    results_dir.mkdir(parents=True, exist_ok=True)

    before_base = find_existing_file(project_root, BEFORE_BASE_CANDIDATES)
    after_base = find_existing_file(project_root, AFTER_BASE_CANDIDATES)

    if before_base is None:
        raise FileNotFoundError(
            "Arquivo base BEFORE não encontrado. Esperado um destes nomes:\n"
            + "\n".join(BEFORE_BASE_CANDIDATES)
        )

    if after_base is None:
        raise FileNotFoundError(
            "Arquivo base AFTER não encontrado. Esperado um destes nomes:\n"
            + "\n".join(AFTER_BASE_CANDIDATES)
        )

    before_brand_texts, before_text_col = collect_brand_texts(before_base)
    after_brand_texts, after_text_col = collect_brand_texts(after_base)

    print(f"[INFO] BEFORE base: {before_base.name} | coluna textual: {before_text_col} | n={len(before_brand_texts)}")
    print(f"[INFO] AFTER base:  {after_base.name} | coluna textual: {after_text_col} | n={len(after_brand_texts)}")

    before_metric = WriteprintsBrandSimilarity(
        top_k_mfw=150,
        window_chars=1500,
        step_chars=750,
        explained_variance_threshold=0.95,
        disruption_z_threshold=2.0,
    )
    before_metric.fit(before_brand_texts)

    after_metric = WriteprintsBrandSimilarity(
        top_k_mfw=150,
        window_chars=1500,
        step_chars=750,
        explained_variance_threshold=0.95,
        disruption_z_threshold=2.0,
    )
    after_metric.fit(after_brand_texts)

    generated_csvs = discover_generated_csvs(project_root)
    if not generated_csvs:
        raise FileNotFoundError(
            "Nenhum CSV de respostas geradas encontrado em generated_comments/ ou na raiz."
        )

    detailed_rows = []
    summary_rows = []

    for gen_csv in generated_csvs:
        print(f"[INFO] Processando: {gen_csv.name}")

        df = read_csv_flexible(gen_csv)
        response_cols = find_generated_response_columns(df)

        if not response_cols:
            print(f"[WARN] Nenhuma coluna de resposta gerada detectada em {gen_csv.name}")
            continue

        chosen_base = choose_brand_base_for_generated_file(gen_csv, before_base, after_base)
        metric = before_metric if chosen_base == before_base else after_metric
        base_label = "before" if chosen_base == before_base else "after"

        file_meta = parse_metadata_from_filename(gen_csv.name)

        for response_col in response_cols:
            texts, text_ids = safe_text_list_with_row_ids(df[response_col], gen_csv.stem)

            if not texts:
                print(f"[WARN] Coluna vazia ignorada: {response_col} em {gen_csv.name}")
                continue

            scores_df = metric.score_many(texts, text_ids=text_ids)

            scores_df["source_file"] = gen_csv.name
            scores_df["response_column"] = response_col
            scores_df["brand_base_file"] = chosen_base.name
            scores_df["brand_base_label"] = base_label

            for k, v in file_meta.items():
                scores_df[k] = v

            detailed_rows.append(scores_df)

            summary_rows.append({
                "source_file": gen_csv.name,
                "response_column": response_col,
                "brand_base_file": chosen_base.name,
                "brand_base_label": base_label,
                "release_period": file_meta["release_period"],
                "persona": file_meta["persona"],
                "prompting_strategy": file_meta["prompting_strategy"],
                "n_texts": len(scores_df),
                "mean_similarity_score_0_1": float(scores_df["similarity_score_0_1"].mean()),
                "std_similarity_score_0_1": float(scores_df["similarity_score_0_1"].std(ddof=0)),
                "mean_nn_euclidean_distance": float(scores_df["mean_nn_euclidean_distance"].mean()),
                "std_nn_euclidean_distance": float(scores_df["mean_nn_euclidean_distance"].std(ddof=0)),
                "mean_best_cosine_similarity": float(scores_df["mean_best_cosine_similarity"].mean()),
                "std_best_cosine_similarity": float(scores_df["mean_best_cosine_similarity"].std(ddof=0)),
                "mean_pattern_disruption_rate": float(scores_df["pattern_disruption_rate"].mean()),
                "std_pattern_disruption_rate": float(scores_df["pattern_disruption_rate"].std(ddof=0)),
            })

    if not detailed_rows:
        raise RuntimeError("Nenhum resultado foi gerado. Verifique colunas e arquivos.")

    detailed_df = pd.concat(detailed_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    detailed_out = results_dir / "stylometric_similarity_results_detailed.csv"
    summary_out = results_dir / "stylometric_similarity_results_summary.csv"

    detailed_df.to_csv(detailed_out, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_out, index=False, encoding="utf-8-sig")

    print(f"[OK] Arquivo detalhado salvo em: {detailed_out}")
    print(f"[OK] Arquivo resumo salvo em:    {summary_out}")


def main():
    parser = argparse.ArgumentParser(
        description="Executa Stylometric Similarity inspirada em Writeprints no projeto."
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Caminho para a raiz do projeto",
    )

    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    run(project_root)


main()