# TicketIQ

**Agentic AI for customer support: classify a ticket, read its sentiment, pick the best answering strategy
with a contextual bandit, retrieve the relevant policy, and let a ReAct agent on a local LLM decide and reply.**

```
ticket ─► Naive Bayes category ─┐
       └► VADER aspect sentiment ┴► urgency ─► contextual bandit picks prompt style + top-K
                                                      │
          FAISS top-K policy passages ◄───────────────┘
                     │
                     ▼
          ReAct agent (Llama 3.2 / Qwen 2.5 on Ollama): think ─► call tools ─► answer or escalate
                     │
          customer feedback + latency ─► reward ─► bandit update
```

**Walkthrough:** [`demo.ipynb`](demo.ipynb) shows the classifier, the bandit learning curve and two tickets going
through the agent, in seconds on a laptop.

## Results

**Ticket classifier**: Naive Bayes written from scratch (counting, Laplace smoothing, log-probability scoring),
trained on a stratified 75/25 split of 160 synthetic tickets and scored with scikit-learn on the 40 held out.

| | accuracy | macro precision | macro recall | macro F1 |
|---|---|---|---|---|
| held-out | **0.950** | 0.958 | 0.950 | **0.949** |

| category | precision | recall | F1 |
|---|---|---|---|
| billing | 1.000 | 1.000 | 1.000 |
| account | 1.000 | 1.000 | 1.000 |
| technical | 0.833 | 1.000 | 0.909 |
| feature request | 1.000 | 0.800 | 0.889 |

The two errors are feature requests read as technical problems. The first version of the data scored 100%, which
only showed it was too easy, so the generator now mixes a sentence from a second category into **45%** of tickets.

**Contextual bandit**: epsilon-greedy (ε = 0.15) over **4 configurations** (concise or empathetic step-by-step
prompt × top-2 or top-5 retrieval) across **36 states** (category × urgency × customer tier). It is tested in a
simulated world with hidden helpfulness and latency per configuration; an oracle grades its choices.

| 20,000 simulated tickets | picked the best configuration |
|---|---|
| first 1,000 tickets | **54.4%** |
| last 4,000 tickets | **88.8%** (the ceiling with 15% random exploration is 88.75%) |

Average reward per ticket rises from 6.10 in the first tenth of the run to 6.84 in the last.

## How it works

| File | What it does | Key functions |
|---|---|---|
| `ticketiq/classifier.py` | Tokenizer, Naive Bayes from scratch, stratified split, evaluation | `NaiveBayesTextClassifier`, `TicketClassifierService` |
| `ticketiq/sentiment.py` | Sentiment for 1 to 3 product aspects per ticket with VADER; urgency score from category, sentiment and tier | `AspectSentimentAnalyzer`, `score_urgency` |
| `ticketiq/retrieval.py` | Knowledge base chunked by heading, TF-IDF vectors in a FAISS inner-product index, top-K search | `KnowledgeRetriever` |
| `ticketiq/bandit.py` | The 4 configurations, state key, reward (helpfulness minus latency), epsilon-greedy learner | `EpsilonGreedyContextualBandit`, `compute_reward` |
| `ticketiq/agent.py` | ReAct loop written from scratch, tools (account status, refund eligibility), Ollama client with an offline template fallback, a code rule that feature requests are never escalated | `TriageAgent`, `LlmClient` |
| `ticketiq/triage.py` | One ticket end to end, and feedback into the bandit | `TicketIQ.handle`, `TicketIQ.feedback` |
| `ticketiq/config.py` | Paths, seeds, model names (override with `TICKETIQ_*` environment variables) | `SETTINGS` |
| `reproduce.py` | Every number above | |
| `data/` | 160 labelled tickets, their generator, 5 policy documents | |

## Reproduce

```bash
pip install -r requirements.txt
python reproduce.py        # classifier report + 20,000-ticket bandit simulation, writes results/results.json
pytest -q                  # 9 checks, no model server needed
jupyter notebook demo.ipynb   # the walkthrough
```

Triage one ticket:

```bash
ollama pull qwen2.5:1.5b && ollama pull llama3.2:1b      # optional; without Ollama the template writer is used
python -m ticketiq --subject "Charged twice" --body "I was billed twice this month, please refund it." --tier pro
```

## Limitations

The tickets are synthetic and the test set is small (40 tickets). The bandit is evaluated in simulation, not on
real customers. Retrieval and sentiment have no labelled evaluation yet.
