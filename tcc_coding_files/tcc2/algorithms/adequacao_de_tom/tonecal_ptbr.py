from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None

try:
    from mord import LogisticIT  # type: ignore
    HAVE_MORD = True
except Exception:  # pragma: no cover
    HAVE_MORD = False
    from sklearn.linear_model import LogisticRegression

WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
SENT_RE = re.compile(r"[.!?]+")
PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""
    return str(text).strip()


def tokenize_words(text: str) -> List[str]:
    return WORD_RE.findall(normalize_text(text).lower())


def split_sentences(text: str) -> List[str]:
    t = normalize_text(text)
    parts = [p.strip() for p in SENT_RE.split(t) if p.strip()]
    return parts if parts else ([t] if t else [])


def count_patterns(text: str, patterns: Sequence[str]) -> int:
    t = normalize_text(text).lower()
    total = 0
    for pat in patterns:
        total += len(re.findall(pat, t, flags=re.IGNORECASE | re.UNICODE))
    return total


def contains_any(text: str, patterns: Sequence[str]) -> bool:
    return count_patterns(text, patterns) > 0


def compute_text_stats(text: str) -> Dict[str, float]:
    t = normalize_text(text)
    words = tokenize_words(t)
    sents = split_sentences(t)
    punct = re.findall(PUNCT_RE, t)
    unique_words = set(words)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0.0
    avg_sent_len = np.mean([len(tokenize_words(s)) for s in sents]) if sents else 0.0

    return {
        "length_chars": float(len(t)),
        "length_tokens": float(len(words)),
        "n_sentences": float(len(sents)),
        "avg_word_len": float(avg_word_len),
        "avg_sentence_len_tokens": float(avg_sent_len),
        "type_token_ratio": safe_div(len(unique_words), len(words)),
        "punct_ratio": safe_div(len(punct), max(1, len(t))),
        "exclamation_count": float(t.count("!")),
        "question_count": float(t.count("?")),
    }


class ToneLexicon:
    GREETING = [
        r"\bolá\b", r"\boi\b", r"\bopa\b", r"\bboa tarde\b",
        r"\bboa noite\b", r"\bbom dia\b", r"\bseja bem-vindo\b",
        r"\bseja bem vinda\b", r"\bseja bem vindo\b",
    ]
    GRATITUDE = [
        r"\bobrigad[oa]s?\b", r"\bagradecemos\b", r"\bmuito obrigad[oa]\b",
        r"\bagradeço\b", r"\bagradecer\b", r"\bgratid[aã]o\b",
        r"\bvalorizamos seu feedback\b", r"\bagradecemos seu feedback\b",
    ]
    APOLOGY = [
        r"\bdesculp", r"\blamentamos\b", r"\bsentimos muito\b",
        r"\bpedimos desculpas\b", r"\bnos desculpe\b", r"\bpeço desculpas\b",
    ]
    DEFERENCE = [
        r"\bsenhor\b", r"\bsenhora\b", r"\bprezado\b", r"\bprezada\b",
        r"\bgentileza\b", r"\bpor favor\b", r"\bestimado\b", r"\bestimada\b",
        r"\bcordialmente\b", r"\bficamos à disposição\b", r"\bficamos a disposição\b",
    ]
    INDIRECTNESS = [
        r"\bpoderia\b", r"\bpoderíamos\b", r"\bgostaríamos\b",
        r"\bse possível\b", r"\bcaso deseje\b", r"\bcaso queira\b",
        r"\brecomendamos\b", r"\bsugerimos\b", r"\bseria possível\b",
    ]
    POSITIVE_CLOSING = [
        r"\bvolte sempre\b", r"\bestamos à disposição\b", r"\bestamos a disposição\b",
        r"\besperamos revê-lo\b", r"\besperamos revê-la\b", r"\baté breve\b",
        r"\bserá um prazer recebê-lo\b", r"\bserá um prazer recebê-la\b",
        r"\bconte conosco\b",
    ]
    IMPOSING_PATTERNS = [
        r"\bfaça\b", r"\bdeve\b", r"\bprecisa\b", r"\btem que\b",
        r"\bmande\b", r"\benvie\b", r"\bentre em contato imediatamente\b",
    ]

    NEGATIVE_COMMENT = [
        r"\bruim\b", r"\bpéssim", r"\bpessim", r"\bhorrível\b", r"\bhorrivel\b",
        r"\bdemor", r"\bfrio\b", r"\berrado\b", r"\bdecepcion", r"\bnunca mais\b",
        r"\breclama", r"\binsatisfeit", r"\bproblema\b", r"\bterrível\b", r"\bterrivel\b",
    ]
    POSITIVE_COMMENT = [
        r"\bexcelente\b", r"\bótim", r"\botim", r"\bmaravilh", r"\bperfeit",
        r"\bamamos\b", r"\badorei\b", r"\badoramos\b", r"\bmuito bom\b",
        r"\bdelicioso\b", r"\bincrível\b", r"\bincrivel\b",
    ]
    QUESTION_COMMENT = [
        r"\?+", r"\bqual\b", r"\bcomo\b", r"\btem\b", r"\bfaz reserva\b",
        r"\bfunciona\b", r"\bhorário\b", r"\bhorario\b", r"\bvalor\b",
    ]
    ACTIONABILITY = [
        r"\bestamos à disposição\b", r"\bestamos a disposição\b", r"\bentre em contato\b",
        r"\bfale conosco\b", r"\bnos chame\b", r"\bnos procure\b",
        r"\bvamos verificar\b", r"\bvamos averiguar\b", r"\bqueremos resolver\b",
        r"\bpodemos ajudar\b", r"\bficaremos felizes em ajudar\b",
    ]
    PERSONA_FORMAL = [
        r"\bprezad", r"\bsenhor", r"\bsenhora", r"\bcordialmente\b",
        r"\bficamos à disposição\b", r"\bagradecemos\b",
    ]
    PERSONA_INFORMAL = [
        r"\boi\b", r"\bolá\b", r"\bopa\b", r"\bvolte sempre\b",
        r"\bconte com a gente\b", r"\bficamos felizes\b",
    ]
    REPAIR_ACTION = [
        r"\bvamos resolver\b", r"\bvamos verificar\b", r"\bqueremos entender\b",
        r"\bpor favor.*entre em contato\b", r"\bchame.*no direct\b",
        r"\bnos envie\b", r"\bnos informe\b",
    ]


def extract_politeness_rubric_features(response: str) -> Dict[str, float]:
    text = normalize_text(response).lower()
    greeting = min(count_patterns(text, ToneLexicon.GREETING), 2) / 2.0
    gratitude = min(count_patterns(text, ToneLexicon.GRATITUDE), 2) / 2.0
    apology = min(count_patterns(text, ToneLexicon.APOLOGY), 1) / 1.0
    deference = min(count_patterns(text, ToneLexicon.DEFERENCE), 2) / 2.0
    indirectness = min(count_patterns(text, ToneLexicon.INDIRECTNESS), 2) / 2.0
    positive_closing = min(count_patterns(text, ToneLexicon.POSITIVE_CLOSING), 2) / 2.0

    imposing = min(count_patterns(text, ToneLexicon.IMPOSING_PATTERNS), 2)
    imposition_control = (2 - imposing) / 2.0

    return {
        "greeting": clamp01(greeting),
        "gratitude": clamp01(gratitude),
        "apology": clamp01(apology),
        "deference": clamp01(deference),
        "indirectness": clamp01(indirectness),
        "positive_closing": clamp01(positive_closing),
        "imposition_control": clamp01(imposition_control),
    }


def detect_comment_type(comment: str) -> Dict[str, float]:
    t = normalize_text(comment).lower()
    neg = contains_any(t, ToneLexicon.NEGATIVE_COMMENT)
    pos = contains_any(t, ToneLexicon.POSITIVE_COMMENT)
    qst = contains_any(t, ToneLexicon.QUESTION_COMMENT)
    mixed = float(neg and pos)
    return {
        "comment_is_negative": float(neg),
        "comment_is_positive": float(pos),
        "comment_is_question": float(qst),
        "comment_is_mixed": mixed,
    }


def score_apology_appropriateness(comment: str, response: str) -> float:
    c = detect_comment_type(comment)
    has_apology = contains_any(response, ToneLexicon.APOLOGY)
    if c["comment_is_negative"]:
        return 1.0 if has_apology else 0.0
    if c["comment_is_positive"]:
        return 0.2 if has_apology else 1.0
    return 0.6 if has_apology else 0.5


def score_gratitude_appropriateness(comment: str, response: str) -> float:
    c = detect_comment_type(comment)
    has_gratitude = contains_any(response, ToneLexicon.GRATITUDE)
    if c["comment_is_positive"]:
        return 1.0 if has_gratitude else 0.2
    if c["comment_is_negative"]:
        return 0.8 if has_gratitude else 0.3
    return 0.9 if has_gratitude else 0.4


def score_sentiment_alignment(comment: str, response: str) -> float:
    c = detect_comment_type(comment)
    has_apology = contains_any(response, ToneLexicon.APOLOGY)
    has_gratitude = contains_any(response, ToneLexicon.GRATITUDE)
    has_action = contains_any(response, ToneLexicon.REPAIR_ACTION)
    if c["comment_is_negative"]:
        score = 0.0
        score += 0.5 if has_apology else 0.0
        score += 0.3 if has_action else 0.0
        score += 0.2 if has_gratitude else 0.0
        return clamp01(score)
    if c["comment_is_positive"]:
        return 1.0 if has_gratitude else 0.3
    if c["comment_is_question"]:
        return 0.9 if len(normalize_text(response)) > 20 else 0.2
    return 0.5 + 0.2 * float(has_gratitude)


def score_actionability(comment: str, response: str) -> float:
    c = detect_comment_type(comment)
    has_action = contains_any(response, ToneLexicon.ACTIONABILITY) or contains_any(response, ToneLexicon.REPAIR_ACTION)
    if c["comment_is_negative"] or c["comment_is_question"]:
        return 1.0 if has_action else 0.2
    return 0.7 if has_action else 0.5


def score_respectfulness(response: str) -> float:
    rub = extract_politeness_rubric_features(response)
    score = (
        0.30 * rub["deference"] +
        0.25 * rub["gratitude"] +
        0.15 * rub["greeting"] +
        0.20 * rub["positive_closing"] +
        0.10 * rub["imposition_control"]
    )
    return clamp01(score)


def score_persona_adherence(response: str, persona: Optional[str]) -> float:
    text = normalize_text(response).lower()
    if not persona:
        return 0.5
    if persona == "persona_1":
        formal = min(count_patterns(text, ToneLexicon.PERSONA_FORMAL), 3) / 3.0
        informal = min(count_patterns(text, ToneLexicon.PERSONA_INFORMAL), 3) / 3.0
        return clamp01(0.7 * formal + 0.3 * (1 - informal))
    if persona == "persona_2":
        formal = min(count_patterns(text, ToneLexicon.PERSONA_FORMAL), 3) / 3.0
        informal = min(count_patterns(text, ToneLexicon.PERSONA_INFORMAL), 3) / 3.0
        return clamp01(0.7 * informal + 0.3 * (1 - formal))
    return 0.5


def repetition_ratio(text: str) -> float:
    toks = tokenize_words(text)
    if not toks:
        return 0.0
    unique = len(set(toks))
    return clamp01(1.0 - safe_div(unique, len(toks)))


def boilerplate_ratio(text: str) -> float:
    patterns = (
        ToneLexicon.GRATITUDE +
        ToneLexicon.APOLOGY +
        ToneLexicon.POSITIVE_CLOSING +
        ToneLexicon.DEFERENCE
    )
    hits = count_patterns(text, patterns)
    toks = len(tokenize_words(text))
    return clamp01(safe_div(hits, max(5, toks / 3)))


def simple_fluency_proxy(text: str) -> float:
    stats = compute_text_stats(text)
    if stats["length_tokens"] == 0:
        return 0.0
    penalty = 0.0
    penalty += 0.5 if stats["length_tokens"] < 6 else 0.0
    penalty += 0.3 if repetition_ratio(text) > 0.45 else 0.0
    penalty += 0.2 if stats["avg_sentence_len_tokens"] > 45 else 0.0
    score = 1.0 - penalty
    return clamp01(score)


def jaccard_similarity(a: str, b: str) -> float:
    sa = set(tokenize_words(a))
    sb = set(tokenize_words(b))
    if not sa and not sb:
        return 1.0
    return safe_div(len(sa & sb), len(sa | sb))


POSITIVE_PROTOTYPES = [
    "Muito obrigado pelo carinho. Ficamos felizes em saber que sua experiência foi excelente.",
    "Agradecemos imensamente sua avaliação positiva. Esperamos recebê-lo novamente em breve.",
]
NEGATIVE_PROTOTYPES = [
    "Sentimos muito pela experiência. Queremos entender melhor o ocorrido e ajudar da melhor forma.",
    "Lamentamos o inconveniente e agradecemos por compartilhar seu feedback conosco.",
]
QUESTION_PROTOTYPES = [
    "Olá! Ficamos à disposição para ajudar. Pode nos chamar para mais informações.",
    "Obrigado pelo contato. Teremos prazer em esclarecer sua dúvida.",
]


def prototype_similarity(comment: str, response: str) -> Dict[str, float]:
    c = detect_comment_type(comment)
    if c["comment_is_negative"]:
        refs = NEGATIVE_PROTOTYPES
    elif c["comment_is_positive"]:
        refs = POSITIVE_PROTOTYPES
    elif c["comment_is_question"]:
        refs = QUESTION_PROTOTYPES
    else:
        refs = POSITIVE_PROTOTYPES + NEGATIVE_PROTOTYPES
    sims = [jaccard_similarity(response, ref) for ref in refs]
    return {
        "prototype_similarity_mean": float(np.mean(sims)) if sims else 0.0,
        "prototype_similarity_max": float(np.max(sims)) if sims else 0.0,
    }


def compute_lexical_contextual_features(comment: str, response: str, persona: Optional[str]) -> Dict[str, float]:
    rub = extract_politeness_rubric_features(response)
    ctype = detect_comment_type(comment)
    proto = prototype_similarity(comment, response)
    stats_resp = compute_text_stats(response)
    stats_comment = compute_text_stats(comment)

    features: Dict[str, float] = {}
    features.update(rub)
    features.update(ctype)
    features.update({
        "apology_appropriateness": score_apology_appropriateness(comment, response),
        "gratitude_appropriateness": score_gratitude_appropriateness(comment, response),
        "sentiment_alignment": score_sentiment_alignment(comment, response),
        "actionability": score_actionability(comment, response),
        "respectfulness": score_respectfulness(response),
        "persona_adherence": score_persona_adherence(response, persona),
        "comment_response_jaccard": jaccard_similarity(comment, response),
        "fluency_proxy": simple_fluency_proxy(response),
        "repetition_ratio": repetition_ratio(response),
        "boilerplate_ratio": boilerplate_ratio(response),
    })
    features.update(proto)
    for k, v in stats_resp.items():
        features[f"resp_{k}"] = v
    for k, v in stats_comment.items():
        features[f"comment_{k}"] = v
    return features


class DenseTfidfSimilarity(BaseEstimator, TransformerMixin):
    """
    Fits TF-IDF over comment+response text and emits a few dense similarity features.
    This keeps the model lightweight and avoids requiring external embedding services.
    """
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.comment_vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.response_vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DenseTfidfSimilarity":
        comments = X["comment"].fillna("").astype(str).tolist()
        responses = X["response"].fillna("").astype(str).tolist()
        self.comment_vectorizer.fit(comments)
        self.response_vectorizer.fit(responses)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        comments = X["comment"].fillna("").astype(str).tolist()
        responses = X["response"].fillna("").astype(str).tolist()
        c_mat = self.comment_vectorizer.transform(comments)
        r_mat = self.response_vectorizer.transform(responses)

        # project response onto comment space for a lexical overlap proxy
        c_vocab = self.comment_vectorizer.vocabulary_
        r_vocab = self.response_vectorizer.vocabulary_
        shared_terms = set(c_vocab).intersection(r_vocab)

        shared_overlap = []
        for c, r in zip(comments, responses):
            ct = set(tokenize_words(c))
            rt = set(tokenize_words(r))
            shared_overlap.append(jaccard_similarity(c, r))

        comment_density = np.asarray(c_mat.sum(axis=1)).ravel()
        response_density = np.asarray(r_mat.sum(axis=1)).ravel()
        token_balance = []
        for c, r in zip(comments, responses):
            token_balance.append(safe_div(len(tokenize_words(r)), max(1, len(tokenize_words(c)))))
        shared_vocab_ratio = safe_div(len(shared_terms), max(1, len(set(c_vocab).union(r_vocab))))

        out = np.column_stack([
            np.asarray(shared_overlap, dtype=float),
            comment_density.astype(float),
            response_density.astype(float),
            np.asarray(token_balance, dtype=float),
            np.full(len(comments), float(shared_vocab_ratio)),
        ])
        return out

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        return np.array([
            "tfidf_comment_response_jaccard",
            "tfidf_comment_density",
            "tfidf_response_density",
            "token_balance",
            "shared_vocab_ratio",
        ])


class NumericFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, persona_col: Optional[str] = "persona") -> None:
        self.persona_col = persona_col
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "NumericFeatureExtractor":
        sample_persona = X[self.persona_col].iloc[0] if self.persona_col and self.persona_col in X.columns and len(X) else None
        feat = compute_lexical_contextual_features(
            comment=normalize_text(X["comment"].iloc[0] if len(X) else ""),
            response=normalize_text(X["response"].iloc[0] if len(X) else ""),
            persona=sample_persona,
        )
        self.feature_names_ = list(feat.keys())
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        rows = []
        for _, row in X.iterrows():
            feat = compute_lexical_contextual_features(
                comment=row.get("comment", ""),
                response=row.get("response", ""),
                persona=row.get(self.persona_col, None) if self.persona_col else None,
            )
            rows.append([feat[k] for k in self.feature_names_])
        return np.asarray(rows, dtype=float)

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        return np.asarray(self.feature_names_, dtype=object)


@dataclass
class ToneCalPTBRConfig:
    random_state: int = 42
    n_splits: int = 5
    use_mord_if_available: bool = True
    scale_numeric: bool = True
    comment_col: str = "comment"
    response_col: str = "response"
    persona_col: str = "persona"
    target_col: str = "human_tone_likert"
    id_col: Optional[str] = None


class OrdinalWrapper:
    def __init__(self, use_mord_if_available: bool = True) -> None:
        self.use_mord_if_available = use_mord_if_available
        self.model = None
        self.classes_: np.ndarray | None = None
        self.backend_: str = ""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OrdinalWrapper":
        self.classes_ = np.sort(np.unique(y))
        if self.use_mord_if_available and HAVE_MORD:
            self.model = LogisticIT(alpha=1.0)
            self.model.fit(X, y)
            self.backend_ = "mord.LogisticIT"
        else:
            warnings.warn(
                "mord is not installed; falling back to multinomial LogisticRegression. "
                "For thesis reporting, prefer installing `mord` for a true ordinal model.",
                RuntimeWarning,
            )
            self.model = LogisticRegression(
                max_iter=2000,
                multi_class="multinomial",
                class_weight="balanced",
                random_state=42,
            )
            self.model.fit(X, y)
            self.backend_ = "sklearn.LogisticRegression(multinomial)"
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            if proba.shape[1] != len(self.classes_):
                raise RuntimeError("Unexpected probability output shape.")
            return proba
        # Defensive fallback
        preds = self.model.predict(X)
        out = np.zeros((len(preds), len(self.classes_)), dtype=float)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        for i, p in enumerate(preds):
            out[i, class_to_idx[p]] = 1.0
        return out

    def predict_expected_value(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba * self.classes_[None, :]).sum(axis=1)


@dataclass
class FoldMetrics:
    mae: float
    rmse: float
    spearman: float


@dataclass
class CVResults:
    fold_metrics: List[FoldMetrics] = field(default_factory=list)
    mae_mean: float = 0.0
    mae_std: float = 0.0
    rmse_mean: float = 0.0
    rmse_std: float = 0.0
    spearman_mean: float = 0.0
    spearman_std: float = 0.0
    backend: str = ""


class ToneCalPTBR:
    def __init__(self, config: Optional[ToneCalPTBRConfig] = None) -> None:
        self.config = config or ToneCalPTBRConfig()
        self.numeric_extractor = NumericFeatureExtractor(persona_col=self.config.persona_col)
        self.tfidf_extractor = DenseTfidfSimilarity()
        self.numeric_imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler() if self.config.scale_numeric else None
        self.model = OrdinalWrapper(use_mord_if_available=self.config.use_mord_if_available)
        self.numeric_feature_names_: List[str] = []
        self.tfidf_feature_names_: List[str] = []
        self.fitted_ = False

    def _assemble_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        required = [self.config.comment_col, self.config.response_col]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        work = df[[self.config.comment_col, self.config.response_col] + ([self.config.persona_col] if self.config.persona_col in df.columns else [])].copy()
        work = work.rename(columns={
            self.config.comment_col: "comment",
            self.config.response_col: "response",
            self.config.persona_col: "persona" if self.config.persona_col in work.columns else self.config.persona_col,
        })

        if fit:
            num = self.numeric_extractor.fit_transform(work)
            self.numeric_feature_names_ = list(self.numeric_extractor.get_feature_names_out())
            tfidf = self.tfidf_extractor.fit_transform(work)
            self.tfidf_feature_names_ = list(self.tfidf_extractor.get_feature_names_out())
            num = self.numeric_imputer.fit_transform(num)
            if self.scaler is not None:
                num = self.scaler.fit_transform(num)
        else:
            num = self.numeric_extractor.transform(work)
            tfidf = self.tfidf_extractor.transform(work)
            num = self.numeric_imputer.transform(num)
            if self.scaler is not None:
                num = self.scaler.transform(num)

        return np.hstack([num, tfidf])

    @property
    def feature_names_(self) -> List[str]:
        return self.numeric_feature_names_ + self.tfidf_feature_names_

    def fit(self, df: pd.DataFrame) -> "ToneCalPTBR":
        if self.config.target_col not in df.columns:
            raise ValueError(f"Missing target column: {self.config.target_col}")
        X = self._assemble_features(df, fit=True)
        y = df[self.config.target_col].astype(int).to_numpy()
        self.model.fit(X, y)
        self.fitted_ = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("Model is not fitted.")
        X = self._assemble_features(df, fit=False)
        probs = self.model.predict_proba(X)
        pred_likert = self.model.predict_expected_value(X)
        tone_quality_score = (pred_likert - 1.0) / 4.0

        out = df.copy()
        out["predicted_likert"] = pred_likert
        out["tone_quality_score"] = tone_quality_score
        for i, cls in enumerate(self.model.classes_):
            out[f"p_class_{int(cls)}"] = probs[:, i]
        return out

    def explain_row(self, row: pd.Series) -> Dict[str, Any]:
        persona = row.get(self.config.persona_col, None) if self.config.persona_col in row.index else None
        features = compute_lexical_contextual_features(
            comment=row.get(self.config.comment_col, ""),
            response=row.get(self.config.response_col, ""),
            persona=persona,
        )
        return {
            "lexical_contextual_features": features,
            "notes": {
                "tone_quality_score_interpretation": "0=equivalente a Likert 1, 1=equivalente a Likert 5",
                "model_backend": self.model.backend_,
            },
        }

    def cross_validate(self, df: pd.DataFrame) -> CVResults:
        if self.config.target_col not in df.columns:
            raise ValueError(f"Missing target column: {self.config.target_col}")

        y = df[self.config.target_col].astype(int).to_numpy()
        kf = KFold(n_splits=self.config.n_splits, shuffle=True, random_state=self.config.random_state)

        fold_metrics: List[FoldMetrics] = []
        backend = ""

        for train_idx, test_idx in kf.split(df):
            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)

            model = ToneCalPTBR(config=self.config)
            model.fit(train_df)
            pred_df = model.predict(test_df)

            y_true = test_df[self.config.target_col].astype(float).to_numpy()
            y_pred = pred_df["predicted_likert"].astype(float).to_numpy()

            mae = mean_absolute_error(y_true, y_pred)
            rmse = root_mean_squared_error(y_true, y_pred)
            if spearmanr is not None:
                rho = spearmanr(y_true, y_pred).statistic
                rho = 0.0 if rho is None or np.isnan(rho) else float(rho)
            else:
                rho = float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))

            fold_metrics.append(FoldMetrics(mae=float(mae), rmse=float(rmse), spearman=float(rho)))
            backend = model.model.backend_

        maes = np.array([m.mae for m in fold_metrics], dtype=float)
        rmses = np.array([m.rmse for m in fold_metrics], dtype=float)
        rhos = np.array([m.spearman for m in fold_metrics], dtype=float)

        return CVResults(
            fold_metrics=fold_metrics,
            mae_mean=float(maes.mean()),
            mae_std=float(maes.std()),
            rmse_mean=float(rmses.mean()),
            rmse_std=float(rmses.std()),
            spearman_mean=float(rhos.mean()),
            spearman_std=float(rhos.std()),
            backend=backend,
        )


def build_training_dataframe(
    df: pd.DataFrame,
    comment_col: str = "comment",
    response_col: str = "response",
    persona_col: str = "persona",
    human_score_cols: Optional[Sequence[str]] = None,
    target_col: str = "human_tone_likert",
) -> pd.DataFrame:
    out = df.copy()
    if human_score_cols:
        missing = [c for c in human_score_cols if c not in out.columns]
        if missing:
            raise ValueError(f"Missing human score columns: {missing}")
        out[target_col] = out[list(human_score_cols)].mean(axis=1).round().clip(1, 5).astype(int)

    needed = [comment_col, response_col, target_col]
    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if persona_col not in out.columns:
        out[persona_col] = None

    return out[[c for c in out.columns if c in set(out.columns)]].copy()


def run_training_pipeline(
    train_csv_path: str,
    output_predictions_csv: Optional[str] = None,
    comment_col: str = "comment",
    response_col: str = "response",
    persona_col: str = "persona",
    target_col: str = "human_tone_likert",
    human_score_cols: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    df = pd.read_csv(train_csv_path)
    train_df = build_training_dataframe(
        df=df,
        comment_col=comment_col,
        response_col=response_col,
        persona_col=persona_col,
        human_score_cols=human_score_cols,
        target_col=target_col,
    )

    config = ToneCalPTBRConfig(
        comment_col=comment_col,
        response_col=response_col,
        persona_col=persona_col,
        target_col=target_col,
    )
    model = ToneCalPTBR(config=config)
    cv = model.cross_validate(train_df)
    model.fit(train_df)
    pred_df = model.predict(train_df)

    if output_predictions_csv:
        pred_df.to_csv(output_predictions_csv, index=False)

    return {
        "cv": {
            "mae_mean": cv.mae_mean,
            "mae_std": cv.mae_std,
            "rmse_mean": cv.rmse_mean,
            "rmse_std": cv.rmse_std,
            "spearman_mean": cv.spearman_mean,
            "spearman_std": cv.spearman_std,
            "backend": cv.backend,
        },
        "feature_names": model.feature_names_,
        "predictions_preview": pred_df.head(5).to_dict(orient="records"),
    }


if __name__ == "__main__":
    # Minimal example:
    demo = pd.DataFrame({
        "comment": [
            "A comida estava deliciosa e o atendimento foi excelente!",
            "Demorou muito e o prato veio frio.",
            "Vocês aceitam reserva para sábado?",
            "Gostei do ambiente, mas o prato principal demorou bastante.",
            "Péssima experiência, atendimento ruim e cobrança errada.",
        ],
        "response": [
            "Olá! Muito obrigado pelo carinho. Ficamos felizes em saber que sua experiência foi excelente. Volte sempre!",
            "Sentimos muito pela demora e pelo prato frio. Queremos entender melhor o ocorrido e ajudar. Por favor, entre em contato conosco.",
            "Olá! Obrigado pelo contato. Teremos prazer em esclarecer sua dúvida e ajudar com a reserva.",
            "Agradecemos pelo seu feedback. Lamentamos a demora no prato principal e vamos verificar internamente. Estamos à disposição.",
            "Pedimos desculpas pela experiência. Lamentamos o ocorrido e queremos resolver isso da melhor forma. Por favor, fale conosco.",
        ],
        "persona": ["persona_2", "persona_1", "persona_1", "persona_2", "persona_1"],
        "human_tone_likert": [5, 4, 4, 4, 5],
    })

    tc = ToneCalPTBR()
    cv = tc.cross_validate(demo)
    tc.fit(demo)
    preds = tc.predict(demo)
    print("CV:", cv)
    print(preds[["predicted_likert", "tone_quality_score"]])
