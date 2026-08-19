# A Multidimensional LLM Evaluation Pipeline for Automatic Responses to Restaurant Reviews in Brazilian Portuguese

**Conferência:** ENIAC 2026 — Encontro Nacional de Inteligência Artificial e Computacional
**Autores:** Danilo Bruno, Carlos M. S. Figueiredo, Danielle Pontes, Tiago de Melo
**Instituição:** Universidade do Estado do Amazonas (UEA)

Este repositório contém o código e o corpus (anonimizado) usados no artigo acima, aceito no
ENIAC 2026. Reúne o pipeline completo: geração de respostas por três LLMs, cálculo de quatro
métricas automáticas e os dados brutos da avaliação humana que serviu de padrão-ouro.

## Resumo do artigo

Este trabalho avalia GPT-5.5, Gemini 3.1 e Llama 4 na geração automática de respostas a
avaliações de restaurantes em português brasileiro, tarefa que exige empatia, adequação de
tom, fluência e consistência de marca. Um experimento fatorial (3×2×2×2 = 2.400 respostas)
combina 15 avaliadores humanos (Likert 1–5; concordância robusta nas dimensões pragmáticas,
Fleiss' κ ∈ [0,76; 0,81], e moderada em fluência, κ = 0,48) com quatro métricas automáticas.
Nenhum modelo é universalmente superior: Llama 4 lidera em empatia, Gemini 3.1 em adequação
de tom e consistência de marca, e GPT-5.5 em fluência. Não há evidência de superioridade
consistente da estratégia zero-shot sobre a few-shot nas dimensões pragmáticas (p = 0,47 em
empatia). Mais relevante, há discordância de ranking entre as métricas automáticas e o
julgamento humano nas dimensões pragmáticas: em adequação de tom, o modelo mais bem avaliado
pelos humanos é o pior pelo proxy automático, o que desaconselha seu uso isolado em tarefas
com nuances pragmáticas.

**Principais contribuições:**
- Um *framework* reprodutível de avaliação multidimensional para respostas automáticas a
  avaliações gastronômicas em português brasileiro, combinando avaliação humana (quatro
  dimensões pragmáticas) e quatro métricas automáticas complementares.
- Um experimento fatorial completo (3 modelos × 2 personas × 2 estratégias de prompting ×
  2 períodos), totalizando 2.400 respostas e 24 cenários distintos, com avaliação humana por
  painel heterogêneo de 15 avaliadores.
- Evidência empírica de que métricas automáticas isoladas podem *inverter* o ranking humano
  em dimensões pragmáticas — o que desaconselha seu uso sem triangulação com avaliação humana
  em ciclos de avaliação de sistemas de resposta automática em produção.

## Sobre os dados

O corpus de origem compreende **200 comentários reais** de um único estabelecimento de
Manaus/AM, coletados do Google Maps, TripAdvisor e Instagram, divididos em dois períodos
(100 anteriores e 100 posteriores ao lançamento do ChatGPT, usado como marco temporal
aproximado). Cada comentário foi submetido a **3 modelos** (GPT-5.5, Gemini 3.1, Llama 4) ×
**2 personas** (formal/informal) × **2 estratégias de prompting** (zero-shot/few-shot),
gerando as **2.400 respostas** analisadas no artigo.

Sobre cada resposta foram calculadas quatro métricas automáticas — fluência (perplexidade
via GPT-2 PT-BR), adequação de tom (ToneCal), consistência de marca (estilometria
Writeprints) e empatia percebida (WASSA) — e uma amostra estratificada de 72 itens (3 por
cenário × 24 cenários) foi avaliada por **15 avaliadores humanos independentes**, em cego
quanto a modelo e período, segundo as quatro dimensões pragmáticas em escala Likert 1–5.

### Privacidade e anonimização

Nesta publicação:

- **O nome do estabelecimento foi generalizado para "Restaurante A"** em todos os arquivos,
  nomes de coluna e nomes de arquivo.
- **Nomes próprios de terceiros mencionados no corpo dos comentários** (tipicamente
  funcionários citados nominalmente por clientes, ex. "o atendimento do fulano foi ótimo")
  **foram substituídos por `[funcionário]`**, tanto no comentário original quanto nas
  respostas geradas pelos modelos e na resposta humana de referência.
- E-mails e links de contato do estabelecimento que apareciam em respostas humanas de
  referência (usadas como baseline) foram substituídos por endereços de exemplo.
- Os dados de identificação dos 15 avaliadores humanos (e-mail, nome) **nunca foram
  incluídos neste repositório** — apenas um `evaluator_id` numérico (1–15) é usado, com
  perfil agregado (idade e profissão) em `docs/perfis_avaliadores.md`.

## Estrutura

```
tcc_coding_files/
├── data/
│   └── human_scores_consolidated.csv   # avaliação humana: 15 avaliadores × 72 itens
├── docs/
│   ├── perfis_avaliadores.md           # perfil agregado dos avaliadores (idade/profissão)
│   └── relatorio_concordancia.md       # Fleiss' κ, Krippendorff's α, ICC por dimensão
├── prompts/
│   ├── gpt/ gemini/ llama/             # prompts de sistema por modelo × persona × estratégia
├── results/
│   ├── fluency/                        # perplexidade (GPT-2 PT-BR) por cenário + consolidado
│   ├── perceived_empathy/              # escores WASSA de empatia
│   ├── stylometric_similarity/         # similaridade estilométrica (Writeprints)
│   └── tonecal/                        # adequação de tom (ToneCal)
├── tcc2/
│   ├── algorithms/                     # implementação das 4 métricas automáticas
│   │   ├── adequacao_de_tom/
│   │   ├── consistência_de_marca/
│   │   ├── empatia/
│   │   └── fluencia/
│   └── api/                            # geração de respostas via GPT-5.5, Gemini 3.1, Llama 4
└── requirements.txt
```

## Reprodutibilidade

Instale as dependências:

```bash
pip install -r requirements.txt
```

Geração de respostas (requer chaves de API como variável de ambiente — nunca embutidas no
código):

```bash
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
python tcc2/api/gpt_api.py --persona persona_1 --shot zero_shot --comment "..."
python tcc2/api/gemini_api.py --persona persona_1 --shot zero_shot --comment "..."
python tcc2/api/llama_local.py --persona persona_1 --shot zero_shot --comment "..."  # via Ollama local
```

Cada métrica automática tem seu próprio script de execução em `tcc2/algorithms/<dimensão>/`.

## ⚠️ Nota de verificação pendente

Os coeficientes de concordância inter-avaliadores em `docs/relatorio_concordancia.md`
(recalculados diretamente de `data/human_scores_consolidated.csv`) **divergem dos valores
reportados no artigo publicado** nas dimensões de adequação de tom e, principalmente,
fluência. Essa divergência está sob verificação pelos autores — ver nota detalhada no
início do próprio `docs/relatorio_concordancia.md`. Até a confirmação, o artigo publicado
é a referência normativa.

## Licença

Código sob licença MIT (ver `LICENSE`). Os dados (corpus anonimizado e resultados) são
disponibilizados para fins de pesquisa e reprodutibilidade.

## Como citar

> BRUNO, D.; FIGUEIREDO, C. M. S.; PONTES, D.; DE MELO, T. A Multidimensional LLM
> Evaluation Pipeline for Automatic Responses to Restaurant Reviews in Brazilian
> Portuguese. In: **Encontro Nacional de Inteligência Artificial e Computacional (ENIAC)**,
> 2026, Brasil.

```bibtex
@inproceedings{bruno2026multidimensional,
  title     = {A Multidimensional LLM Evaluation Pipeline for Automatic Responses to
               Restaurant Reviews in Brazilian Portuguese},
  author    = {Bruno, Danilo and Figueiredo, Carlos M. S. and Pontes, Danielle and de Melo, Tiago},
  booktitle = {Encontro Nacional de Intelig{\^e}ncia Artificial e Computacional (ENIAC)},
  year      = {2026},
  address   = {Brasil}
}
```

## Contato

Dúvidas sobre o artigo, o código ou o corpus: entre em contato com os autores via
Universidade do Estado do Amazonas (UEA).
