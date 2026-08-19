# Relatório de Concordância Inter-Avaliadores

> Avaliação humana com 15 avaliadores avaliando 72 itens idênticos.
> Métricas: Fleiss' Kappa, Krippendorff's Alpha (ordinal), ICC(2,1).

> **⚠️ Nota de verificação pendente (adicionada na publicação deste repositório).**
> Os valores de Fleiss' κ deste relatório, recalculados de forma independente a partir de
> `data/human_scores_consolidated.csv` e conferidos batendo exatamente com os números abaixo,
> **não coincidem com os valores reportados no artigo publicado** em duas dimensões:
> tom (artigo: κ=0,80 vs. aqui: κ=0,860) e, principalmente, fluência (artigo: κ=0,48,
> descrito como "concordância moderada" por efeito de teto vs. aqui: κ=0,794, "substancial").
> Os valores de empatia (0,81) e consistência de marca (0,76) batem com o artigo.
> Essa divergência está sob verificação pelos autores — consulte o artigo publicado como
> referência normativa até a confirmação, e não assuma que este relatório a substitui.

## Resumo

| Dimensão | Fleiss' κ | Interpretação | α (ordinal) | Interpretação | ICC(2,1) | Interpretação |
|----------|-----------|---------------|-------------|---------------|----------|---------------|
| empatia | 0.818 | Almost Perfect | 0.910 | Good | 0.928 | Excellent |
| consistencia_de_marca | 0.760 | Substantial | 0.866 | Good | 0.875 | Good |
| adequacao_de_tom | 0.860 | Almost Perfect | 0.938 | Good | 0.943 | Excellent |
| fluencia | 0.794 | Substantial | 0.815 | Good | 0.876 | Good |

## Detalhamento por Dimensão

### empatia

**Fleiss' Kappa:** 0.8178 (SE=0.0098, z=83.54)
- P̄ (concordância observada): 0.8775
- Pe (concordância esperada): 0.3276
- Distribuição por categoria: 1=0.000, 2=0.108, 3=0.156, 4=0.470, 5=0.265

**Krippendorff's Alpha (ordinal):** 0.9105

**ICC(2,1):** 0.9283
- ICC(2,k): 0.9949
- MS_rows (entre itens): 11.9520
- MS_cols (entre avaliadores): 0.0996
- MS_error (residual): 0.0607

### consistencia_de_marca

**Fleiss' Kappa:** 0.7599 (SE=0.0104, z=73.33)
- P̄ (concordância observada): 0.8431
- Pe (concordância esperada): 0.3466
- Distribuição por categoria: 1=0.000, 2=0.293, 3=0.459, 4=0.222, 5=0.026

**Krippendorff's Alpha (ordinal):** 0.8665

**ICC(2,1):** 0.8747
- ICC(2,k): 0.9905
- MS_rows (entre itens): 8.2896
- MS_cols (entre avaliadores): 0.1858
- MS_error (residual): 0.0769

### adequacao_de_tom

**Fleiss' Kappa:** 0.8595 (SE=0.0098, z=87.27)
- P̄ (concordância observada): 0.9058
- Pe (concordância esperada): 0.3296
- Distribuição por categoria: 1=0.000, 2=0.099, 3=0.165, 4=0.472, 5=0.264

**Krippendorff's Alpha (ordinal):** 0.9382

**ICC(2,1):** 0.9430
- ICC(2,k): 0.9960
- MS_rows (entre itens): 11.7314
- MS_cols (entre avaliadores): 0.0811
- MS_error (residual): 0.0466

### fluencia

**Fleiss' Kappa:** 0.7940 (SE=0.0415, z=19.14)
- P̄ (concordância observada): 0.9503
- Pe (concordância esperada): 0.7585
- Distribuição por categoria: 1=0.000, 2=0.000, 3=0.030, 4=0.106, 5=0.864

**Krippendorff's Alpha (ordinal):** 0.8152

**ICC(2,1):** 0.8757
- ICC(2,k): 0.9906
- MS_rows (entre itens): 2.6516
- MS_cols (entre avaliadores): 0.0247
- MS_error (residual): 0.0249

## Interpretação das Escalas

### Fleiss' Kappa
| Faixa | Interpretação |
|-------|---------------|
| < 0.00 | Poor |
| 0.00–0.20 | Slight |
| 0.21–0.40 | Fair |
| 0.41–0.60 | Moderate |
| 0.61–0.80 | Substantial |
| 0.81–1.00 | Almost Perfect |

### Krippendorff's Alpha
| Faixa | Interpretação |
|-------|---------------|
| < 0.667 | Tentative (conclusões tentativas) |
| 0.667–0.800 | Acceptable |
| > 0.800 | Good (conclusões confiáveis) |

### ICC
| Faixa | Interpretação |
|-------|---------------|
| < 0.50 | Poor |
| 0.50–0.75 | Moderate |
| 0.75–0.90 | Good |
| > 0.90 | Excellent |