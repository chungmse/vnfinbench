#!/usr/bin/env python3
"""Inference PhoBERT tren test set + do latency tung bai (batch=1 de cong bang voi LLM API)."""
import argparse, json, time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ap=argparse.ArgumentParser()
ap.add_argument('--task', required=True)
ap.add_argument('--input', required=True)
ap.add_argument('--maxlen', type=int, default=256)
a=ap.parse_args()

labels=json.load(open(f'ckpt-{a.task}-labels.json'))
tok=AutoTokenizer.from_pretrained(f'ckpt/{a.task}/best')
model=AutoModelForSequenceClassification.from_pretrained(f'ckpt/{a.task}/best').cuda().eval()

out=open(f'results/{a.task}_phobert_ft.jsonl','w')
rows=[json.loads(l) for l in open(a.input)]
# warmup
for r in rows[:3]:
    x=tok(r['title']+' . '+r['body'][:2000], truncation=True, max_length=a.maxlen, return_tensors='pt').to('cuda')
    with torch.no_grad(): model(**x)
for r in rows:
    t0=time.perf_counter()
    x=tok(r['title']+' . '+r['body'][:2000], truncation=True, max_length=a.maxlen, return_tensors='pt').to('cuda')
    with torch.no_grad(): logits=model(**x).logits
    pred=labels[int(logits.argmax())]
    ms=(time.perf_counter()-t0)*1000
    out.write(json.dumps({'id':r['id'],'pred':pred,'latency_ms':round(ms,1),'label':r.get('label')},ensure_ascii=False)+'\n')
print('done', a.task, len(rows))
