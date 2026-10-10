# Honest AI — Implementation

Implementation of:

> Xinxi Chen, Li Wang, Wei Wu, Qi Tang, Yiyao Liu. **"Honest AI: Fine-Tuning
> 'Small' Language Models to Say 'I Don't Know', and Reducing Hallucination
> in RAG."** KDD Cup Workshop for RAG @ SIGKDD 2024.
> [arXiv:2410.09699](https://arxiv.org/abs/2410.09699)

This was the winning approach for the false-premise question type in Task 2
of the 2024 Meta KDD Cup / CRAG Challenge.

## What's implemented

| Paper section | Module | What it does |
|---|---|---|
| 3.4 — data relabeling rule | `honest_ai/data_prep.py` | Keeps original answers for `comparison`/`false_premise` question types and yes/no/true/false answers; relabels everything else to `"i don't know"`. Reserves a 250-example holdout untouched. |
| 3.4 — QLoRA fine-tuning | `honest_ai/finetune.py`, `honest_ai/config.py` | 4-bit QLoRA on `Llama-2-7b-chat` with the paper's exact hyperparameters: `alpha=16, r=64, dropout=0.1, batch_size=8, lr=2e-4, weight_decay=0.001, epochs=5`. |
| 3.3 — naive RAG | `honest_ai/rag.py` | Cosine-similarity document retrieval over Sentence-BERT embeddings. |
| 3.5 — Sentence-BERT pruner | `honest_ai/pruner.py` | Scores each sentence in a document against the query (Eq. 1), keeps the top-k sentences plus each one's following m sentences, gated by similarity threshold n. |
| 3.5 / Fig. 4 — hybrid approach | `honest_ai/hybrid.py` | Runs the RAG model with the domain+answer JSON prompt (Appendix A.1); if domain == movie and the answer is valid, uses it; otherwise falls back to the fine-tuned "Honest AI" model. "Invalid" = `"invalid question"` response or a JSON-parsing failure, exactly as in the paper. |
| Appendix A.1 | `honest_ai/prompts.py` | The domain-classification + answer JSON prompt template, verbatim. |
| 3.2 — scoring | `honest_ai/evaluate.py` | perfect=1 / acceptable=0.5 / missing=0 / incorrect=-1, macro-averaged across domains. Ships a deterministic rule-based judge (exact match + IDK/false-premise detection) plus a pluggable `llm_judge` hook for semantic-equivalence judging (CRAG itself uses GPT-4 as judge — you can back this with Claude, GPT-4, or a local model). |

## Install

```bash
pip install -r requirements.txt
```

The core logic (`data_prep`, `evaluate`, `utils`, and the pure functions in
`pruner`) has no heavy dependencies and can be imported/tested without
`torch`/`transformers`/GPU. `finetune.py`, `inference.py`, `rag.py`, and the
embedding path of `pruner.py` lazy-import their heavy deps so you only pay
for what you use.

## Usage

### 1. Prepare data (Section 3.4, Fig. 3)

Input: a CRAG-style JSONL with at least `query`, `answer`, `question_type`
per line (get the real dataset from the [CRAG benchmark repo](https://github.com/facebookresearch/CRAG) /
Meta KDD Cup 2024).

```bash
python scripts/prepare_data.py \
    --input path/to/crag_train.jsonl \
    --train_out data/train_relabeled.jsonl \
    --holdout_out data/holdout.jsonl \
    --sft_out data/train_sft.jsonl \
    --num_holdout 250
```

### 2. Fine-tune with QLoRA (Section 3.4)

```bash
python scripts/run_finetune.py \
    --train_sft data/train_sft.jsonl \
    --base_model meta-llama/Llama-2-7b-chat-hf \
    --output_dir ./honest_ai_qlora_out
```

Requires a GPU with enough VRAM for 4-bit QLoRA on a 7B model (paper
targeted models <10B specifically for resource efficiency), and access to
the gated `Llama-2-7b-chat-hf` weights on Hugging Face (`huggingface-cli login`).

### 3. Run the hybrid pipeline

```python
from honest_ai.hybrid import HybridAnswerer
from honest_ai.inference import HFTextGenerator, FineTunedHonestModel
from honest_ai.pruner import SentenceBertPruner

rag_generator = HFTextGenerator("meta-llama/Meta-Llama-3-8B-Instruct")
fine_tuned_model = FineTunedHonestModel(
    base_model="meta-llama/Llama-2-7b-chat-hf",
    adapter_path="./honest_ai_qlora_out",
)
pruner = SentenceBertPruner()  # defaults: k=3, m=2, n=0.75

answerer = HybridAnswerer(rag_generator, fine_tuned_model, pruner=pruner)
result = answerer.answer(
    query="what 2010 film was directed by christopher nolan?",
    retrieved_documents=[web_search_result_1, web_search_result_2],
)
print(result.final_answer, result.used_source, result.domain)
```

### 4. Score predictions (Section 3.2)

```bash
python scripts/run_eval.py --predictions preds.jsonl
```

where `preds.jsonl` rows look like:
`{"query": ..., "ground_truth": ..., "prediction": ..., "domain": ...}`

To get closer to CRAG's actual GPT-4-judge scoring instead of the built-in
rule-based fallback for "acceptable" vs "incorrect", pass an `llm_judge`
callable into `evaluate()`:

```python
from honest_ai.evaluate import evaluate

def my_llm_judge(question, ground_truth, prediction) -> str:
    # call Claude/GPT-4/etc., return "perfect" | "acceptable" | "incorrect"
    ...

report = evaluate(examples, llm_judge=my_llm_judge)
```

## Tests

The paper's mechanical rules (relabeling, JSON extraction, scoring rubric,
cosine similarity, hybrid routing) are covered by unit tests using fake
embedders/generators so they run without any GPU or model download:

```bash
pip install pytest numpy
python -m pytest tests/ -v
```

## Notes / deviations

- **Sentence splitting**: the paper doesn't specify its sentence tokenizer;
  `pruner.split_into_sentences` uses a lightweight regex splitter. Swap in
  `nltk.sent_tokenize` or spaCy's sentencizer for production-grade accuracy.
- **Domain-set movie-first routing**: the paper only fully explored the
  hybrid approach for the movie domain due to competition time constraints
  (Section 6). `HybridConfig.rag_first_domains` is a list so you can extend
  this once you've validated other domains have similarly low hallucination
  rates for your RAG model.
- **LLM-judge for "acceptable"**: CRAG's real scoring pipeline uses GPT-4 as
  an automatic judge for semantic equivalence. This repo ships a rule-based
  fallback (exact match / token overlap) and an `llm_judge` hook — plug in
  your own judge model to match the paper's actual eval more closely.
