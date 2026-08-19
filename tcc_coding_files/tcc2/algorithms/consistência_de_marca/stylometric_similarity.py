import re
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

# -----------------------------
# 1) Stopwords PT (lista curta mas útil; você pode ampliar depois)
# -----------------------------
PT_STOPWORDS = set("""
a ao aos à às aqui aquilo aquele aquela aqueles aquelas ante antes após até com como contra
da das de dela delas dele deles demais dentro desde do dos e ela elas ele eles em entre era eram
essa essas esse esses esta estas este estes estou está estão estavam foi foram há isso isto já
la lhe lhes mas me meu minha minhas meus na nas nem no nos nós o os ou para pela pelas pelo pelos
por porque qual quando que quem se sem seu sua suas seus sob sobre também tem têm tinha tinham
toda todas todo todos um uma umas uns você vocês eu tu
""".split())

# -----------------------------
# 2) Funções auxiliares
# -----------------------------
def _safe_div(a, b):
    return float(a) / float(b) if b else 0.0

_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)

def keep_function_words(text: str) -> str:
    """
    Retorna apenas stopwords (function words), para reduzir sinal de tópico e
    focar em assinatura estilística (voz da marca).
    """
    words = _WORD_RE.findall(str(text).lower())
    return " ".join([w for w in words if w in PT_STOPWORDS])

# -----------------------------
# 3) Extração de features estilométricas "Writeprints-like"
#    (reforçada com shares de pontuação e marcadores de atendimento)
# -----------------------------
def extract_writeprints_features(text: str) -> dict:
    if text is None:
        text = ""
    t = str(text)
    t_stripped = t.strip()
    lower = t_stripped.lower()

    chars = list(t)
    n_chars = len(chars)

    words = _WORD_RE.findall(lower)
    n_words = len(words)
    unique_words = set(words)

    # Sentenças (heurística simples)
    sents = [s for s in re.split(r"[.!?]+", t_stripped) if s.strip()]
    n_sents = len(sents)

    # Parágrafos
    paras = [p for p in re.split(r"\n\s*\n+", t_stripped) if p.strip()]
    n_paras = len(paras)

    # Contagens por classes
    n_letters = sum(ch.isalpha() for ch in chars)
    n_digits = sum(ch.isdigit() for ch in chars)
    n_spaces = sum(ch.isspace() for ch in chars)
    n_upper = sum(ch.isupper() for ch in chars)

    # Pontuação total
    punct = re.findall(r"[^\w\s]", t_stripped, flags=re.UNICODE)
    n_punct = len(punct)

    # Marcadores específicos (assinatura)
    exclam = t_stripped.count("!")
    quest = t_stripped.count("?")
    dots = t_stripped.count(".")
    commas = t_stripped.count(",")
    semicol = t_stripped.count(";")
    colon = t_stripped.count(":")
    quotes = (
        t_stripped.count('"')
        + t_stripped.count("“") + t_stripped.count("”")
        + t_stripped.count("'")
    )
    ellipsis = t_stripped.count("...") + t_stripped.count("…")

    # Emojis-like (heurístico simples)
    emojis_like = sum(
        ord(ch) > 127 and (not ch.isalnum()) and (not ch.isspace())
        for ch in chars
    )

    # Comprimento de palavras
    word_lens = [len(w) for w in words] if words else [0]
    avg_word_len = float(np.mean(word_lens))
    std_word_len = float(np.std(word_lens))

    # Comprimento de sentenças (em palavras)
    sent_word_counts = [len(_WORD_RE.findall(s)) for s in sents] if sents else [0]
    avg_sent_len = float(np.mean(sent_word_counts))
    std_sent_len = float(np.std(sent_word_counts))

    # Riqueza lexical
    ttr = _safe_div(len(unique_words), n_words)
    hapax = sum(1 for w in unique_words if words.count(w) == 1) if n_words else 0
    hapax_ratio = _safe_div(hapax, n_words)

    # Stopwords ratio
    stop_ct = sum(1 for w in words if w in PT_STOPWORDS)
    stop_ratio = _safe_div(stop_ct, n_words)

    # Alongamento (informalidade)
    elongated = len(re.findall(r"(.)\1{2,}", lower, flags=re.UNICODE))

    feats = {
        # Estruturais
        "n_chars": n_chars,
        "n_words": n_words,
        "n_sents": n_sents,
        "n_paras": n_paras,

        # Ratios / densidades (normalizam tamanho)
        "letters_ratio": _safe_div(n_letters, n_chars),
        "digits_ratio": _safe_div(n_digits, n_chars),
        "spaces_ratio": _safe_div(n_spaces, n_chars),
        "upper_ratio": _safe_div(n_upper, n_chars),
        "punct_ratio": _safe_div(n_punct, n_chars),

        # Pontuação por palavra
        "exclam_per_word": _safe_div(exclam, n_words),
        "quest_per_word": _safe_div(quest, n_words),
        "commas_per_word": _safe_div(commas, n_words),
        "colons_per_word": _safe_div(colon, n_words),
        "semicolon_per_word": _safe_div(semicol, n_words),
        "quotes_per_word": _safe_div(quotes, n_words),
        "ellipsis_per_word": _safe_div(ellipsis, n_words),

        # Informalidade / estilo
        "emojis_like_per_word": _safe_div(emojis_like, n_words),
        "elongated_per_word": _safe_div(elongated, n_words),

        # Lexicais
        "avg_word_len": avg_word_len,
        "std_word_len": std_word_len,
        "avg_sent_len": avg_sent_len,
        "std_sent_len": std_sent_len,
        "ttr": ttr,
        "hapax_ratio": hapax_ratio,
        "stop_ratio": stop_ratio,
    }

    # Shares de pontuação (assinatura mais “writeprints”)
    punct_types = {
        "comma": commas,
        "dot": dots,
        "exclam": exclam,
        "quest": quest,
        "colon": colon,
        "semicolon": semicol,
        "quotes": quotes,
        "ellipsis": ellipsis,
    }
    total_p = sum(punct_types.values()) or 1
    for k, v in punct_types.items():
        feats[f"punct_share_{k}"] = v / total_p

    # Marcadores de atendimento (brand voice)
    feats["has_greeting"] = int(bool(re.search(r"\b(ol[aá]|oi|bom dia|boa tarde|boa noite)\b", lower)))
    feats["has_thanks"] = int(bool(re.search(r"\b(obrigad|agradec)\w*\b", lower)))
    feats["has_apology"] = int(bool(re.search(r"\b(desculp|lament)\w*\b", lower)))

    return feats

class WriteprintsNumericTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        feats = extract_writeprints_features("x")
        self.feature_names_ = list(feats.keys())
        return self

    def transform(self, X):
        rows = []
        for t in X:
            feats = extract_writeprints_features(t)
            rows.append([feats[k] for k in self.feature_names_])
        return np.array(rows, dtype=float)

# -----------------------------
# 4) Perfil da marca/persona + Similaridade vs perfil
#    (char n-grams + function-word n-grams + numeric writeprints; com pesos)
# -----------------------------
def build_writeprints_profile(
    reference_texts,
    *,
    # Vetores
    char_ngram_range=(3, 5),
    fw_ngram_range=(1, 3),
    # TF-IDF params (mais “estilo”, menos ruído)
    char_min_df=1,
    char_max_df=0.95,
    sublinear_tf=True,
    # Pesos dos blocos
    w_char=1.0,
    w_fw=1.5,
    w_num=2.0,
):
    """
    Retorna uma função score(texts) que calcula:
      - cosine_similarity(text, centroido_do_estilo)
    Se quiser uma distância "Writeprints Distance" simples:
      distance = 1 - similarity
    """
    reference_texts = pd.Series(reference_texts).astype(str).fillna("").tolist()

    # A) Char n-grams (assinatura de escrita)
    char_vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=char_ngram_range,
        min_df=char_min_df,
        max_df=char_max_df,
        sublinear_tf=sublinear_tf,
    )

    # B) Function words n-grams (remove tópico e foca estilo)
    fw_vec = TfidfVectorizer(
        preprocessor=keep_function_words,
        analyzer="word",
        ngram_range=fw_ngram_range,
        min_df=1,
        sublinear_tf=sublinear_tf,
    )

    # C) Numeric writeprints
    num_tf = WriteprintsNumericTransformer()
    scaler = StandardScaler(with_mean=False)  # compatível com sparse

    # Fit SOMENTE na referência (sem leakage)
    X_char = char_vec.fit_transform(reference_texts)
    X_fw = fw_vec.fit_transform(reference_texts)
    X_num = scaler.fit_transform(num_tf.fit_transform(reference_texts))

    # Concat com pesos
    X = sparse.hstack(
        [
            X_char.multiply(w_char),
            X_fw.multiply(w_fw),
            sparse.csr_matrix(X_num) * w_num,
        ],
        format="csr",
    )
    centroid = sparse.csr_matrix(X.mean(axis=0))

    def score(texts, *, return_distance=False):
        texts = pd.Series(texts).astype(str).fillna("").tolist()
        A = char_vec.transform(texts)
        B = fw_vec.transform(texts)
        C = scaler.transform(num_tf.transform(texts))

        M = sparse.hstack(
            [
                A.multiply(w_char),
                B.multiply(w_fw),
                sparse.csr_matrix(C) * w_num,
            ],
            format="csr",
        )

        sims = cosine_similarity(M, centroid).ravel()
        if return_distance:
            # “Writeprints Distance” simples e interpretável
            dists = 1.0 - sims
            return sims, dists
        return sims

    return score

# -----------------------------
# 5) Exemplo de uso (no seu dataset)
# -----------------------------
if __name__ == "__main__":
    # Referência (corpus da marca): use a coluna `response` do arquivo before/after correspondente
    ref = pd.read_csv("after_llm_release_comments.csv")["response"]

    # Respostas geradas (um cenário específico)
    gen = pd.read_csv("generated_comments/after_llm_release_comments_generated_responses_persona_1_zero_shot.csv")

    gpt_col = "gpt5.5 Generated Response Persona sofisticada e formal"
    gem_col = "Gemini Generated Response Persona sofisticada e formal"
    lla_col = "llama4 Generated Response Persona sofisticada e formal"

    score_fn = build_writeprints_profile(
        ref,
        char_min_df=1,      # se tiver mais dados, experimente 2
        char_max_df=0.95,
        w_char=1.0,
        w_fw=1.5,
        w_num=2.0,
    )

    gpt_sims, gpt_dist = score_fn(gen[gpt_col], return_distance=True)
    gem_sims, gem_dist = score_fn(gen[gem_col], return_distance=True)
    lla_sims, lla_dist = score_fn(gen[lla_col], return_distance=True)

    print("=== Similarity (maior = mais no estilo) ===")
    print("GPT-5.5 mean:", float(np.mean(gpt_sims)))
    print("Gemini mean:", float(np.mean(gem_sims)))
    print("llama4 mean:", float(np.mean(lla_sims)))

    print("\n=== Distance = 1 - similarity (menor = mais no estilo) ===")
    print("GPT-5.5 mean dist:", float(np.mean(gpt_dist)))
    print("Gemini mean dist:", float(np.mean(gem_dist)))
    print("llama4 mean dist:", float(np.mean(lla_dist)))
