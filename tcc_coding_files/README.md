# A Multidimensional LLM Evaluation Pipeline for Automatic Responses to Restaurant Reviews in Brazilian Portuguese

Código e dados de suporte do artigo aceito no **ENIAC** (Encontro Nacional de Inteligência
Artificial e Computacional), de autoria de Danilo Bruno, Carlos M. S. Figueiredo, Danielle
Pontes e Tiago de Melo (Universidade do Estado do Amazonas — UEA).

O trabalho avalia GPT-5.5, Gemini 3.1 e Llama 4 na geração automática de respostas a
avaliações de restaurantes em português brasileiro, combinando avaliação humana (15
avaliadores, escala Likert 1–5) com quatro métricas automáticas (fluência via perplexidade,
adequação de tom via ToneCal, consistência de marca via estilometria Writeprints e empatia
via WASSA).

## Nota sobre privacidade e anonimização dos dados

O corpus de origem contém 200 avaliações reais de um único estabelecimento, coletadas do
Google Maps, TripAdvisor e Instagram. Nesta publicação:

- **O nome do estabelecimento foi generalizado para "Restaurante A"** em todos os arquivos,
  nomes de coluna e nomes de arquivo.
- **Nomes próprios de terceiros mencionados no corpo dos comentários** (tipicamente
  funcionários citados nominalmente por clientes, ex. "o atendimento do fulano foi ótimo")
  **foram substituídos por `[funcionário]`**, tanto no comentário original quanto nas
  respostas geradas pelos modelos e na resposta humana de referência.
- E-mails e links de contato do estabelecimento que apareciam em respostas humanas de
  referência (usadas como baseline) foram substituídos por endereços de exemplo.

Os dados de identificação dos 15 avaliadores humanos (e-mail, nome) **nunca foram incluídos
neste repositório** — apenas um `evaluator_id` numérico (1–15) é usado, com perfil
agregado (idade e profissão) em `docs/perfis_avaliadores.md`.

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

## Citação

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
