from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Dict, Optional, Tuple
from collections import Counter
import re
import math

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")

# Pontuação explícita: sinais de pontuação são marcadores estilométricos clássicos.
DEFAULT_PUNCTUATION = [".", ",", ";", ":", "!", "?", "-", "(", ")", '"', "'"]


# =========================================================
# Utilidades básicas
# =========================================================

def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    return str(text).strip()


def tokenize_words(text: str) -> List[str]:
    return _WORD_RE.findall(normalize_text(text).lower())


def split_sentences(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    parts = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return parts or [text]


def split_paragraphs(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    parts = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]
    return parts or [text]


def sliding_char_windows(
    text: str,
    window_chars: int = 1500,
    step_chars: Optional[int] = None,
) -> List[str]:
    """
    Sliding windows em caracteres.

    O artigo de Abbasi & Chen trabalha com blocos/janelas de 1.500 caracteres.
    Aqui usamos janelas sobrepostas por padrão (50%), o que é mais próximo de
    um sliding window real do que blocos totalmente não sobrepostos.
    """
    text = normalize_text(text)
    if not text:
        return []

    if len(text) <= window_chars:
        return [text]

    if step_chars is None:
        step_chars = max(1, window_chars // 2)  # 50% de overlap

    windows: List[str] = []
    for start in range(0, len(text) - window_chars + 1, step_chars):
        windows.append(text[start:start + window_chars])

    # garante cobertura do final
    if windows:
        last_start = len(text) - window_chars
        if last_start > 0 and (not windows or windows[-1] != text[last_start:last_start + window_chars]):
            windows.append(text[last_start:last_start + window_chars])

    if not windows:
        windows = [text]

    return windows


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


# =========================================================
# Features estilométricas
# =========================================================

def yules_k(words: Sequence[str]) -> float:
    if not words:
        return 0.0

    freqs = Counter(words)
    n = len(words)
    m2 = sum(f * f for f in freqs.values())
    return 10000.0 * ((m2 - n) / (n * n)) if n else 0.0


def hapax_ratio(words: Sequence[str]) -> float:
    if not words:
        return 0.0

    freqs = Counter(words)
    hapax = sum(1 for _, c in freqs.items() if c == 1)
    return hapax / len(words)


def type_token_ratio(words: Sequence[str]) -> float:
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def average_word_frequency_class(words: Sequence[str]) -> float:
    """
    Proxy simples de repetição/regularidade lexical no próprio texto.
    """
    if not words:
        return 0.0

    freqs = Counter(words)
    vals = np.asarray([freqs[w] for w in words], dtype=float)
    return float(vals.mean())


def word_length_distribution(words: Sequence[str], max_len: int = 20) -> Dict[str, float]:
    counts = Counter(min(len(w), max_len) for w in words)
    total = len(words)
    return {f"wl_{i}": _safe_div(counts.get(i, 0), total) for i in range(1, max_len + 1)}


def punctuation_profile(text: str, punctuation_marks: Sequence[str]) -> Dict[str, float]:
    total_chars = len(text)
    return {
        f"punct_{repr(p)}": _safe_div(text.count(p), total_chars)
        for p in punctuation_marks
    }


def character_class_profile(text: str) -> Dict[str, float]:
    text = normalize_text(text)
    total_chars = len(text)

    n_upper = sum(1 for ch in text if ch.isupper())
    n_digit = sum(1 for ch in text if ch.isdigit())
    n_space = sum(1 for ch in text if ch.isspace())
    n_alpha = sum(1 for ch in text if ch.isalpha())

    return {
        "char_upper_ratio": _safe_div(n_upper, total_chars),
        "char_digit_ratio": _safe_div(n_digit, total_chars),
        "char_space_ratio": _safe_div(n_space, total_chars),
        "char_alpha_ratio": _safe_div(n_alpha, total_chars),
    }


def sentence_length_features(sentences: Sequence[str]) -> Dict[str, float]:
    sent_lengths = [len(tokenize_words(s)) for s in sentences if s.strip()]
    if not sent_lengths:
        sent_lengths = [0]

    arr = np.asarray(sent_lengths, dtype=float)
    return {
        "sent_len_mean": float(arr.mean()),
        "sent_len_std": float(arr.std(ddof=0)),
        "sent_len_median": float(np.median(arr)),
        "sent_len_p90": float(np.percentile(arr, 90)),
        "n_sentences": float(len(sent_lengths)),
    }


def paragraph_features(paragraphs: Sequence[str]) -> Dict[str, float]:
    para_lengths = [len(tokenize_words(p)) for p in paragraphs if p.strip()]
    if not para_lengths:
        para_lengths = [0]

    arr = np.asarray(para_lengths, dtype=float)
    return {
        "n_paragraphs": float(len(para_lengths)),
        "para_len_mean": float(arr.mean()),
        "para_len_std": float(arr.std(ddof=0)),
    }


def word_length_stats(words: Sequence[str]) -> Dict[str, float]:
    lengths = [len(w) for w in words]
    if not lengths:
        lengths = [0]

    arr = np.asarray(lengths, dtype=float)
    return {
        "word_len_mean": float(arr.mean()),
        "word_len_std": float(arr.std(ddof=0)),
        "word_len_median": float(np.median(arr)),
    }


def lexical_richness_features(words: Sequence[str]) -> Dict[str, float]:
    return {
        "n_tokens": float(len(words)),
        "n_types": float(len(set(words))),
        "ttr": type_token_ratio(words),
        "hapax_ratio": hapax_ratio(words),
        "yules_k": yules_k(words),
        "avg_word_freq_class": average_word_frequency_class(words),
    }


def structural_features(text: str, words: Sequence[str], sentences: Sequence[str]) -> Dict[str, float]:
    n_chars = len(text)
    n_words = len(words)
    n_sentences = len(sentences) if sentences else 0

    return {
        "chars_per_word": _safe_div(n_chars, n_words),
        "words_per_sentence": _safe_div(n_words, n_sentences),
        "exclamation_per_sentence": _safe_div(text.count("!"), n_sentences),
        "question_per_sentence": _safe_div(text.count("?"), n_sentences),
    }


def extract_mfw_vocabulary(
    reference_texts: Sequence[str],
    *,
    top_k: int = 150,
    min_token_len: int = 1,
    only_alpha: bool = True,
) -> List[str]:
    """
    MFW corpus-driven: mais defensável do que usar stopwords arbitrárias.
    """
    counter: Counter[str] = Counter()

    for text in reference_texts:
        for tok in tokenize_words(text):
            if len(tok) < min_token_len:
                continue
            if only_alpha and not tok.isalpha():
                continue
            counter[tok] += 1

    return [w for w, _ in counter.most_common(top_k)]


def mfw_profile(words: Sequence[str], vocabulary: Sequence[str]) -> Dict[str, float]:
    counts = Counter(words)
    total = len(words)
    return {f"mfw_{w}": _safe_div(counts.get(w, 0), total) for w in vocabulary}


def extract_writeprints_style_features(
    text: str,
    *,
    mfw_vocabulary: Sequence[str],
    punctuation_marks: Sequence[str] = DEFAULT_PUNCTUATION,
    max_word_len_bin: int = 20,
) -> Dict[str, float]:
    text = normalize_text(text)
    words = tokenize_words(text)
    sentences = split_sentences(text)
    paragraphs = split_paragraphs(text)

    feats: Dict[str, float] = {}
    feats.update(mfw_profile(words, mfw_vocabulary))
    feats.update(punctuation_profile(text, punctuation_marks))
    feats.update(character_class_profile(text))
    feats.update(word_length_distribution(words, max_len=max_word_len_bin))
    feats.update(word_length_stats(words))
    feats.update(sentence_length_features(sentences))
    feats.update(paragraph_features(paragraphs))
    feats.update(lexical_richness_features(words))
    feats.update(structural_features(text, words, sentences))
    return feats


# =========================================================
# Saída
# =========================================================

@dataclass
class StylometricSimilarityResult:
    text_id: str
    n_windows: int

    mean_nn_euclidean_distance: float
    std_nn_euclidean_distance: float
    min_nn_euclidean_distance: float

    mean_best_cosine_similarity: float
    std_best_cosine_similarity: float
    max_best_cosine_similarity: float

    pattern_disruption_rate: float
    similarity_score_0_1: float


# =========================================================
# Modelo
# =========================================================

@dataclass
class WriteprintsBrandSimilarity:
    """
    Similaridade estilométrica inspirada em Writeprints para corpus de marca.

    Aproximações fiéis ao paper:
    - sliding windows
    - conjunto rico de features estilométricas
    - padronização no espaço da marca
    - transformação KLT-like via PCA
    - comparação candidato vs janelas reais da marca (não vs vetor zero)

    Nota:
    O 'pattern disruption' aqui é uma aproximação operacional:
    mede a fração de features do candidato fora de uma banda z do corpus de marca.
    Não é uma reprodução literal do algoritmo proprietário/descrito no paper.
    """
    top_k_mfw: int = 150
    window_chars: int = 1500
    step_chars: Optional[int] = None
    punctuation_marks: Sequence[str] = field(default_factory=lambda: list(DEFAULT_PUNCTUATION))
    max_word_len_bin: int = 20
    explained_variance_threshold: float = 0.95
    disruption_z_threshold: float = 2.0

    mfw_vocabulary_: List[str] = field(default_factory=list, init=False)
    feature_columns_: List[str] = field(default_factory=list, init=False)

    feature_mean_: Optional[pd.Series] = field(default=None, init=False)
    feature_std_: Optional[pd.Series] = field(default=None, init=False)

    brand_windows_raw_: Optional[pd.DataFrame] = field(default=None, init=False)
    brand_windows_z_: Optional[np.ndarray] = field(default=None, init=False)
    brand_windows_klt_: Optional[np.ndarray] = field(default=None, init=False)

    brand_nn_distance_mean_: Optional[float] = field(default=None, init=False)
    brand_nn_distance_std_: Optional[float] = field(default=None, init=False)

    pca_: Optional[PCA] = field(default=None, init=False)

    def _texts_to_windows(self, texts: Sequence[str]) -> List[str]:
        windows: List[str] = []
        for text in texts:
            windows.extend(
                sliding_char_windows(
                    text,
                    window_chars=self.window_chars,
                    step_chars=self.step_chars,
                )
            )
        return windows

    def _extract_feature_df(self, texts: Sequence[str]) -> pd.DataFrame:
        windows = self._texts_to_windows(texts)
        if not windows:
            raise ValueError("Nenhuma janela válida foi gerada a partir dos textos.")

        rows = []
        for idx, window in enumerate(windows):
            feats = extract_writeprints_style_features(
                window,
                mfw_vocabulary=self.mfw_vocabulary_,
                punctuation_marks=self.punctuation_marks,
                max_word_len_bin=self.max_word_len_bin,
            )
            feats["__window_id__"] = idx
            feats["__text__"] = window
            rows.append(feats)

        df = pd.DataFrame(rows).fillna(0.0)

        if self.feature_columns_:
            for col in self.feature_columns_:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[["__window_id__", "__text__"] + self.feature_columns_]

        return df

    def _zscore(self, x: pd.DataFrame) -> np.ndarray:
        z = (x - self.feature_mean_) / self.feature_std_
        return z.to_numpy(dtype=float)

    def _fit_pca_klt(self, z_brand: np.ndarray) -> None:
        max_components = min(z_brand.shape[0], z_brand.shape[1])
        if max_components < 1:
            raise ValueError("Não há componentes suficientes para PCA/KLT.")

        pca_full = PCA(n_components=max_components, svd_solver="full", random_state=42)
        pca_full.fit(z_brand)

        cumulative = np.cumsum(pca_full.explained_variance_ratio_)
        k = int(np.searchsorted(cumulative, self.explained_variance_threshold) + 1)
        k = max(1, min(k, max_components))

        self.pca_ = PCA(n_components=k, svd_solver="full", random_state=42)
        self.pca_.fit(z_brand)

    def fit(self, brand_texts: Sequence[str]) -> "WriteprintsBrandSimilarity":
        if not brand_texts:
            raise ValueError("brand_texts must not be empty")

        brand_texts = [normalize_text(t) for t in brand_texts if normalize_text(t)]
        if not brand_texts:
            raise ValueError("brand_texts must contain at least one non-empty text")

        self.mfw_vocabulary_ = extract_mfw_vocabulary(
            brand_texts,
            top_k=self.top_k_mfw,
            min_token_len=1,
            only_alpha=True,
        )

        # extração inicial para descobrir colunas
        temp_rows = []
        temp_windows = self._texts_to_windows(brand_texts)
        for idx, window in enumerate(temp_windows):
            feats = extract_writeprints_style_features(
                window,
                mfw_vocabulary=self.mfw_vocabulary_,
                punctuation_marks=self.punctuation_marks,
                max_word_len_bin=self.max_word_len_bin,
            )
            feats["__window_id__"] = idx
            feats["__text__"] = window
            temp_rows.append(feats)

        df = pd.DataFrame(temp_rows).fillna(0.0)
        self.feature_columns_ = [c for c in df.columns if not c.startswith("__")]
        self.brand_windows_raw_ = df[["__window_id__", "__text__"] + self.feature_columns_].copy()

        x = self.brand_windows_raw_[self.feature_columns_]
        self.feature_mean_ = x.mean(axis=0)
        self.feature_std_ = x.std(axis=0, ddof=0).replace(0, 1.0)

        z_brand = self._zscore(x)
        self.brand_windows_z_ = z_brand

        # KLT-like transform
        self._fit_pca_klt(z_brand)
        self.brand_windows_klt_ = self.pca_.transform(z_brand)

        # baseline interna: distância do vizinho mais próximo entre janelas da própria marca
        if self.brand_windows_klt_.shape[0] == 1:
            self.brand_nn_distance_mean_ = 1.0
            self.brand_nn_distance_std_ = 0.0
        else:
            dmat = _pairwise_euclidean(self.brand_windows_klt_, self.brand_windows_klt_)
            np.fill_diagonal(dmat, np.inf)
            nn = dmat.min(axis=1)
            self.brand_nn_distance_mean_ = float(nn.mean())
            self.brand_nn_distance_std_ = float(nn.std(ddof=0))

        return self

    def _transform_candidate_windows(self, texts: Sequence[str]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        if not self.mfw_vocabulary_ or self.pca_ is None:
            raise RuntimeError("fit() must be called before scoring.")

        df = self._extract_feature_df(texts)
        x = df[self.feature_columns_]
        z = self._zscore(x)
        klt = self.pca_.transform(z)
        return df, z, klt

    def _pattern_disruption_rate(self, z_candidate: np.ndarray) -> float:
        if z_candidate.size == 0:
            return 0.0
        disrupted = np.abs(z_candidate) > self.disruption_z_threshold
        return float(disrupted.mean())

    def score_text(self, text: str, text_id: str = "text") -> StylometricSimilarityResult:
        df, z_cand, klt_cand = self._transform_candidate_windows([text])

        # Distância e similaridade do candidato contra TODAS as janelas da marca
        dmat = _pairwise_euclidean(klt_cand, self.brand_windows_klt_)
        cmat = cosine_similarity(klt_cand, self.brand_windows_klt_)

        nn_dist = dmat.min(axis=1)
        best_cos = cmat.max(axis=1)

        mean_nn_dist = float(nn_dist.mean())
        std_nn_dist = float(nn_dist.std(ddof=0))
        min_nn_dist = float(nn_dist.min())

        mean_best_cos = float(best_cos.mean())
        std_best_cos = float(best_cos.std(ddof=0))
        max_best_cos = float(best_cos.max())

        disruption_rate = self._pattern_disruption_rate(z_cand)

        # score calibrado pela coesão interna da marca
        # se o candidato estiver na mesma ordem de distância das próprias janelas da marca,
        # o score tende a ser alto; quanto mais distante do baseline, menor.
        baseline = self.brand_nn_distance_mean_ if self.brand_nn_distance_mean_ and self.brand_nn_distance_mean_ > 0 else 1.0
        similarity_01 = float(math.exp(-mean_nn_dist / baseline))

        return StylometricSimilarityResult(
            text_id=text_id,
            n_windows=len(df),
            mean_nn_euclidean_distance=mean_nn_dist,
            std_nn_euclidean_distance=std_nn_dist,
            min_nn_euclidean_distance=min_nn_dist,
            mean_best_cosine_similarity=mean_best_cos,
            std_best_cosine_similarity=std_best_cos,
            max_best_cosine_similarity=max_best_cos,
            pattern_disruption_rate=disruption_rate,
            similarity_score_0_1=similarity_01,
        )

    def score_many(
        self,
        texts: Sequence[str],
        text_ids: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        if text_ids is None:
            text_ids = [f"text_{i}" for i in range(len(texts))]

        if len(text_ids) != len(texts):
            raise ValueError("text_ids must have the same length as texts")

        results = [self.score_text(t, tid) for t, tid in zip(texts, text_ids)]
        return pd.DataFrame([r.__dict__ for r in results])


def _pairwise_euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Distância euclidiana vetorizada.
    """
    a2 = np.sum(a * a, axis=1, keepdims=True)
    b2 = np.sum(b * b, axis=1, keepdims=True).T
    d2 = np.maximum(a2 + b2 - 2.0 * (a @ b.T), 0.0)
    return np.sqrt(d2)