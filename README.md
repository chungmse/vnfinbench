# VnFinBench

A benchmark for Vietnamese financial news understanding under real-time constraints.
Four tasks drawn from a corpus of 863,793 articles (2007-2025) from a leading
Vietnamese financial news portal:

| Task | File(s) | Test | Train/Val | Metric |
|---|---|---|---|---|
| T1 Relevance filtering | `data/t1_filtering_*` | 1,184 | 5,518/1,182 | Acc / macro-F1 |
| T2 Event categorization (10 classes) | `data/t2_categorization_*` | 1,250 | 20,000/1,250 | Acc / macro-F1 |
| T3 Ticker linking | `data/t3_ticker_linking_test.jsonl` | 461 | - | per-article P/R/F1 |
| T4 Summarization | `data/t4_summarization_test.jsonl` | 1,000 | - | ROUGE-1/2/L |

## Data format and copyright

Each row carries `id`, `url`, and the task label only. **Article text is not
redistributed**; fetch it from the source URL for research use, respecting the
publisher's terms. For T3, `label` is the gold ticker set and `aliases` is the
retrieved candidate list used by the RAG-alias prompting mode. For T4, `label`
is the editor-written lead (sapo), which serves as the reference summary.

T2 labels are the portal editors' section assignments. T1 labels were produced by
an LLM-assisted pipeline following `ANNOTATION.md` and reviewed in full by a human
annotator.

## Contents

- `prompts/` - the exact Vietnamese prompts for all tasks and modes
- `dict/vn_tickers.json` - 1,401 listed 3-letter tickers with company names
- `dict/vn_aliases.json` - company-name aliases per ticker
- `dict/ticker_ambiguity.json` - per-ticker financial vs non-financial occurrence
  counts over the full corpus (the noise-ratio analysis in the paper)
- `eval/` - dataset construction, inference runners (vLLM OpenAI-compatible API,
  PhoBERT), evaluation, and throughput measurement scripts

## Reproducing

1. Serve an LLM: `vllm serve <model> --max-model-len 8192 --served-model-name llm`
2. Run a task: `python eval/run_llm.py --task filtering --model llm --family qwen --input data/t1_filtering_test.jsonl --max-tokens 10`
   (article text must be joined onto the rows first; see `eval/build_testsets.py`)
3. Score: `python eval/eval.py --task filtering results/filtering_llm_zeroshot.jsonl`

PhoBERT: `python eval/train_phobert.py --task categorization --train ... --val ...`
then `python eval/infer_phobert.py --task categorization --input ...`.
