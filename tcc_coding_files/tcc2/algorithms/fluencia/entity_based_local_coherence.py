from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_TO_GENERATED = {
    "before_llm_release_comments_expanded.csv": [
        "restaurante_a_before_100_generated_responses_persona_1_few_shot.csv",
        "restaurante_a_before_100_generated_responses_persona_1_zero_shot.csv",
        "restaurante_a_before_100_generated_responses_persona_2_few_shot.csv",
        "restaurante_a_before_100_generated_responses_persona_2_zero_shot.csv",
    ],
    "after_llm_release_comments_expanded.csv": [
        "restaurante_a_after_100_generated_responses_persona_1_few_shot.csv",
        "restaurante_a_after_100_generated_responses_persona1_zero_shot.csv",
        "restaurante_a_after_100_generated_responses_persona_2_few_shot.csv",
        "restaurante_a_after_100_generated_responses_persona_2_zero_shot.csv",
    ],
}


@dataclass
class FluencyResult:
    token_count: int
    mean_nll_per_token: float          # lower is better
    perplexity: float                  # lower is better
    inverse_perplexity_0_1: float      # higher is better = exp(-mean_nll)
    valid_for_fluency: bool
    score_error: bool
    score_error_message: str


def _safe_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def build_project_paths(script_path: Path) -> Dict[str, Path]:
    fluency_dir = script_path.parent
    algorithms_dir = fluency_dir.parent
    tcc2_dir = algorithms_dir.parent

    project_root = tcc2_dir.parent

    return {
        "script": script_path,
        "fluency_dir": fluency_dir,
        "algorithms_dir": algorithms_dir,
        "tcc2_dir": tcc2_dir,
        "results_dir": project_root / "results" / "fluency",
        "generated_dir": project_root / "data" / "generated_responses",
        "before_base": project_root / "data" / "expanded" / "before_llm_release_comments_expanded.csv",
        "after_base": project_root / "data" / "expanded" / "after_llm_release_comments_expanded.csv",
        "model_cache_dir": fluency_dir / "model_cache",
    }


def _response_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for col in df.columns:
        lower = col.lower()
        if "generated response" in lower and "processing time" not in lower:
            cols.append(col)
    return cols


def select_comment_column(df: pd.DataFrame) -> str:
    candidates = ["comment", "comentario", "review", "texto"]
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    raise ValueError(
        f"Não encontrei a coluna do comentário base. Colunas disponíveis: {list(df.columns)}"
    )


def infer_metadata_from_generated_filename(filename: str) -> Dict[str, str]:
    lower = filename.lower()

    if "before" in lower:
        period = "before"
    elif "after" in lower:
        period = "after"
    else:
        period = "unknown"

    if "persona_1" in lower:
        persona = "persona_1"
    elif "persona_2" in lower:
        persona = "persona_2"
    else:
        persona = "unknown"

    if "few_shot" in lower:
        shot_type = "few_shot"
    elif "zero_shot" in lower:
        shot_type = "zero_shot"
    else:
        shot_type = "unknown"

    return {
        "period": period,
        "persona": persona,
        "shot_type": shot_type,
    }


def infer_model_from_response_column(colname: str) -> str:
    lower = colname.lower()
    if "gpt" in lower:
        return "gpt5.5"
    if "gemini" in lower:
        return "gemini 3.1"
    if "llama" in lower:
        return "llama4"
    return "unknown"


class LMFluencyScorer:
    MAX_CHARS = 4000
    MIN_TOKEN_COUNT = 2

    def __init__(
        self,
        model_name: str = "pierreguillou/gpt2-small-portuguese",
        batch_size: int = 8,
        device: str | None = None,
        cache_dir: str | None = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.cache_dir = cache_dir

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )
        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._cache: Dict[str, FluencyResult] = {}

    def _normalize_text(self, text: str) -> str:
        text = _safe_text(text)
        if len(text) > self.MAX_CHARS:
            text = text[:self.MAX_CHARS]
        return text

    def _invalid_result(self, message: str = "") -> FluencyResult:
        return FluencyResult(
            token_count=0,
            mean_nll_per_token=float("nan"),
            perplexity=float("nan"),
            inverse_perplexity_0_1=float("nan"),
            valid_for_fluency=False,
            score_error=False,
            score_error_message=message,
        )

    def _error_result(self, message: str) -> FluencyResult:
        return FluencyResult(
            token_count=0,
            mean_nll_per_token=float("nan"),
            perplexity=float("nan"),
            inverse_perplexity_0_1=float("nan"),
            valid_for_fluency=False,
            score_error=True,
            score_error_message=message,
        )

    @torch.no_grad()
    def _score_batch_uncached(self, texts: List[str]) -> List[FluencyResult]:
        norm_texts = [self._normalize_text(t) for t in texts]

        valid_positions = []
        valid_texts = []
        results: List[FluencyResult | None] = [None] * len(norm_texts)

        for i, text in enumerate(norm_texts):
            if not text:
                results[i] = self._invalid_result("empty_text")
                continue

            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) < self.MIN_TOKEN_COUNT:
                results[i] = self._invalid_result("too_few_tokens")
                continue

            valid_positions.append(i)
            valid_texts.append(text)

        if not valid_texts:
            return results  # type: ignore

        try:
            encoded = self.tokenizer(
                valid_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=min(
                    getattr(self.tokenizer, "model_max_length", 1024),
                    1024,
                ),
            )

            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()

            token_losses = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view(shift_labels.size())

            token_losses = token_losses * shift_mask

            token_counts = shift_mask.sum(dim=1).cpu().numpy()
            sum_losses = token_losses.sum(dim=1).cpu().numpy()

            for row_idx, original_pos in enumerate(valid_positions):
                count = int(token_counts[row_idx])

                if count <= 0:
                    results[original_pos] = self._invalid_result("no_scored_tokens")
                    continue

                mean_nll = float(sum_losses[row_idx] / count)
                ppl = float(math.exp(mean_nll))
                inv_ppl = float(math.exp(-mean_nll))  # = 1 / ppl

                results[original_pos] = FluencyResult(
                    token_count=count,
                    mean_nll_per_token=round(mean_nll, 6),
                    perplexity=round(ppl, 6),
                    inverse_perplexity_0_1=round(inv_ppl, 6),
                    valid_for_fluency=True,
                    score_error=False,
                    score_error_message="",
                )

        except Exception as e:
            err = self._error_result(f"{type(e).__name__}: {e}")
            for pos in valid_positions:
                results[pos] = err

        return results  # type: ignore

    def score_many(self, texts: List[str]) -> List[FluencyResult]:
        normalized = [_safe_text(t) for t in texts]
        results: List[FluencyResult | None] = [None] * len(normalized)

        uncached_positions = []
        uncached_texts = []

        for i, text in enumerate(normalized):
            if text in self._cache:
                results[i] = self._cache[text]
            else:
                uncached_positions.append(i)
                uncached_texts.append(text)

        for start in range(0, len(uncached_texts), self.batch_size):
            batch_texts = uncached_texts[start:start + self.batch_size]
            batch_positions = uncached_positions[start:start + self.batch_size]

            batch_results = self._score_batch_uncached(batch_texts)

            for pos, text, result in zip(batch_positions, batch_texts, batch_results):
                self._cache[text] = result
                results[pos] = result

        return results  # type: ignore


def load_base_dataframe(base_file: Path) -> pd.DataFrame:
    if not base_file.exists():
        raise FileNotFoundError(f"Arquivo base não encontrado: {base_file}")
    return pd.read_csv(base_file)


def load_generated_dataframe(generated_file: Path) -> pd.DataFrame:
    if not generated_file.exists():
        raise FileNotFoundError(f"Arquivo de respostas geradas não encontrado: {generated_file}")
    return pd.read_csv(generated_file)


def score_generated_file(
    scorer: LMFluencyScorer,
    base_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    generated_filename: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    comment_col = select_comment_column(base_df)
    gen_response_cols = _response_columns(generated_df)

    if not gen_response_cols:
        raise ValueError(
            f"Nenhuma coluna de resposta gerada encontrada em {generated_filename}. "
            f"Colunas disponíveis: {list(generated_df.columns)}"
        )

    meta = infer_metadata_from_generated_filename(generated_filename)

    n_base = len(base_df)
    n_gen = len(generated_df)
    n = min(n_base, n_gen)

    if n_base != n_gen:
        print(
            f"[WARN] Tamanhos diferentes em {generated_filename}: "
            f"base={n_base}, generated={n_gen}. Será usado o menor tamanho: {n}."
        )

    base_trim = base_df.iloc[:n].copy().reset_index(drop=True)
    gen_trim = generated_df.iloc[:n].copy().reset_index(drop=True)

    wide_out = pd.DataFrame()
    wide_out["row_id"] = np.arange(n)
    wide_out["comment"] = base_trim[comment_col]

    if "response" in base_trim.columns:
        wide_out["human_response_reference"] = base_trim["response"]
    if "restaurant" in base_trim.columns:
        wide_out["restaurant"] = base_trim["restaurant"]
    if "period" in base_trim.columns:
        wide_out["period"] = base_trim["period"]

    long_rows = []

    print(f"[INFO] Colunas de resposta detectadas em {generated_filename}: {gen_response_cols}")

    for response_col in gen_response_cols:
        response_values = gen_trim[response_col].fillna("").astype(str).tolist()
        model_name = infer_model_from_response_column(response_col)

        print(f"\n[INFO] Processando coluna: {response_col}")
        batch_results = scorer.score_many(response_values)

        token_counts = []
        mean_nlls = []
        perplexities = []
        inverse_ppls = []
        valid_flags = []
        error_flags = []
        error_messages = []

        for idx, (generated_response, result) in enumerate(zip(response_values, batch_results)):
            token_counts.append(result.token_count)
            mean_nlls.append(result.mean_nll_per_token)
            perplexities.append(result.perplexity)
            inverse_ppls.append(result.inverse_perplexity_0_1)
            valid_flags.append(result.valid_for_fluency)
            error_flags.append(result.score_error)
            error_messages.append(result.score_error_message)

            long_rows.append(
                {
                    "source_file": generated_filename,
                    "period": meta["period"],
                    "persona": meta["persona"],
                    "shot_type": meta["shot_type"],
                    "model": model_name,
                    "row_id": idx,
                    "comment": wide_out.loc[idx, "comment"],
                    "generated_response": generated_response,
                    "token_count": result.token_count,
                    "mean_nll_per_token": result.mean_nll_per_token,
                    "perplexity": result.perplexity,
                    "inverse_perplexity_0_1": result.inverse_perplexity_0_1,
                    "valid_for_fluency": result.valid_for_fluency,
                    "score_error": result.score_error,
                    "score_error_message": result.score_error_message,
                }
            )

        safe_prefix = (
            response_col
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace("-", "_")
        )

        wide_out[f"{safe_prefix}__generated_response"] = response_values
        wide_out[f"{safe_prefix}__token_count"] = token_counts
        wide_out[f"{safe_prefix}__mean_nll_per_token"] = mean_nlls
        wide_out[f"{safe_prefix}__perplexity"] = perplexities
        wide_out[f"{safe_prefix}__inverse_perplexity_0_1"] = inverse_ppls
        wide_out[f"{safe_prefix}__valid_for_fluency"] = valid_flags
        wide_out[f"{safe_prefix}__score_error"] = error_flags
        wide_out[f"{safe_prefix}__score_error_message"] = error_messages

        valid_perps = [x for x, ok in zip(perplexities, valid_flags) if ok and pd.notna(x)]
        mean_ppl = float(np.mean(valid_perps)) if valid_perps else float("nan")

        print(
            f"[INFO] {generated_filename} | {response_col} -> "
            f"n={len(batch_results)}, valid={sum(valid_flags)}, errors={sum(error_flags)}, "
            f"mean_ppl={mean_ppl:.4f}"
        )

    long_df = pd.DataFrame(long_rows)
    return wide_out, long_df


def build_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()

    def _valid_mean(series_name: str, df_group: pd.DataFrame) -> float:
        valid_df = df_group[df_group["valid_for_fluency"]]
        if valid_df.empty:
            return float("nan")
        return float(valid_df[series_name].mean())

    rows = []
    grouped = long_df.groupby(["period", "persona", "shot_type", "model"], dropna=False)

    for keys, group in grouped:
        period, persona, shot_type, model = keys

        total_rows = len(group)
        valid_count = int(group["valid_for_fluency"].sum())
        error_count = int(group["score_error"].sum())

        rows.append(
            {
                "period": period,
                "persona": persona,
                "shot_type": shot_type,
                "model": model,
                "total_rows": total_rows,
                "valid_count": valid_count,
                "valid_rate": round(valid_count / total_rows, 4) if total_rows else np.nan,
                "score_error_count": error_count,
                "mean_token_count": round(float(group["token_count"].mean()), 4),
                "mean_nll_all": round(float(group["mean_nll_per_token"].mean()), 6),
                "mean_nll_valid": round(_valid_mean("mean_nll_per_token", group), 6),
                "mean_perplexity_all": round(float(group["perplexity"].mean()), 6),
                "mean_perplexity_valid": round(_valid_mean("perplexity", group), 6),
                "mean_inverse_perplexity_all": round(float(group["inverse_perplexity_0_1"].mean()), 6),
                "mean_inverse_perplexity_valid": round(_valid_mean("inverse_perplexity_0_1", group), 6),
                "min_perplexity": round(float(group["perplexity"].min()), 6),
                "max_inverse_perplexity": round(float(group["inverse_perplexity_0_1"].max()), 6),
            }
        )

    return pd.DataFrame(rows)


def main():
    paths = build_project_paths(Path(__file__).resolve())

    results_dir = paths["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    paths["model_cache_dir"].mkdir(parents=True, exist_ok=True)

    scorer = LMFluencyScorer(
        model_name="pierreguillou/gpt2-small-portuguese",
        batch_size=8,
        cache_dir=str(paths["model_cache_dir"]),
    )

    base_map = {
        "before_llm_release_comments_expanded.csv": paths["before_base"],
        "after_llm_release_comments_expanded.csv": paths["after_base"],
    }

    generated_dir = paths["generated_dir"]
    all_long_dfs = []

    for base_name, generated_files in BASE_TO_GENERATED.items():
        base_file = base_map[base_name]
        base_df = load_base_dataframe(base_file)

        for generated_name in generated_files:
            generated_file = generated_dir / generated_name
            generated_df = load_generated_dataframe(generated_file)

            wide_df, long_df = score_generated_file(
                scorer=scorer,
                base_df=base_df,
                generated_df=generated_df,
                generated_filename=generated_name,
            )

            output_name = f"fluency_{Path(generated_name).stem}.csv"
            output_path = results_dir / output_name
            wide_df.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"[OK] Resultado salvo em: {output_path}")

            all_long_dfs.append(long_df)

    consolidated_df = pd.concat(all_long_dfs, ignore_index=True) if all_long_dfs else pd.DataFrame()

    consolidated_path = results_dir / "fluency_consolidated.csv"
    consolidated_df.to_csv(consolidated_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Consolidado salvo em: {consolidated_path}")

    summary_df = build_summary(consolidated_df)
    summary_path = results_dir / "fluency_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Resumo salvo em: {summary_path}")


main()