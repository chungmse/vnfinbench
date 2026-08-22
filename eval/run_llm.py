#!/usr/bin/env python3
"""Chạy 1 task qua vLLM (OpenAI-compatible API), ghi predictions + latency.

Dùng: python run_llm.py --task filtering --model qwen3-14b-awq \
        --input data/filtering_test.jsonl --base-url http://localhost:8000/v1
Input JSONL: mỗi dòng {"id", "title", "body", "label"(optional), "aliases"(EL-RAG)}
Output: results/{task}_{model}_{mode}.jsonl  mỗi dòng {"id","pred","latency_ms","label"}
"""
import argparse, json, time, pathlib, sys
from openai import OpenAI

ap = argparse.ArgumentParser()
ap.add_argument('--task', required=True, choices=['filtering','categorization','entity_linking','summarization'])
ap.add_argument('--mode', default='zeroshot', choices=['zeroshot','rag'])
ap.add_argument('--model', required=True)          # tên model đã serve trên vLLM
ap.add_argument('--input', required=True)
ap.add_argument('--base-url', default='http://localhost:8000/v1')
ap.add_argument('--labels', default='')            # categorization: nhãn cách nhau dấu phẩy
ap.add_argument('--max-body', type=int, default=3000)  # cắt body theo ký tự
ap.add_argument('--max-tokens', type=int, default=256)
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--family', default='qwen', choices=['qwen','gptoss'],
                help='qwen: tat thinking; gptoss: reasoning effort low')
a = ap.parse_args()

extra = {}
if a.family == 'qwen':
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
elif a.family == 'gptoss':
    extra = {"reasoning_effort": "low"}

tpl_name = f"{a.task}_{a.mode}" if a.task == 'entity_linking' else a.task
tpl = open(f'prompts/{tpl_name}.txt').read()
client = OpenAI(base_url=a.base_url, api_key='none')
out_path = pathlib.Path(f'results/{a.task}_{a.model.replace("/","_")}_{a.mode}.jsonl')
out_path.parent.mkdir(exist_ok=True)
done = set()
if out_path.exists():   # resume
    done = {json.loads(l)['id'] for l in open(out_path)}
out = open(out_path, 'a')

rows = [json.loads(l) for l in open(a.input)]
if a.limit: rows = rows[:a.limit]
for i, r in enumerate(rows):
    if r['id'] in done: continue
    prompt = tpl.format(title=r.get('title',''), body=r.get('body','')[:a.max_body],
                        labels=a.labels, aliases=r.get('aliases',''))
    t0 = time.perf_counter()
    resp = client.chat.completions.create(model=a.model, max_tokens=a.max_tokens,
        temperature=0, messages=[{"role":"user","content":prompt}], extra_body=extra)
    ms = (time.perf_counter()-t0)*1000
    import re as _re
    pred = _re.sub(r'<think>.*?</think>', '', resp.choices[0].message.content or '', flags=_re.S).strip()
    out.write(json.dumps({"id": r['id'], "pred": pred,
                          "latency_ms": round(ms,1), "label": r.get('label')}, ensure_ascii=False)+'\n')
    out.flush()
    if i % 50 == 0: print(f"{i}/{len(rows)} avg_ms_last={ms:.0f}", file=sys.stderr)
print("done:", out_path)
