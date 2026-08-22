#!/usr/bin/env python3
import json, time, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tok=AutoTokenizer.from_pretrained('ckpt/summ_bartpho/best')
model=AutoModelForSeq2SeqLM.from_pretrained('ckpt/summ_bartpho/best').cuda().eval().half()
rows=[json.loads(l) for l in open('data/summ_test.jsonl')]
out=open('results/summarization_bartpho_ft.jsonl','w')
for i,r in enumerate(rows):
    t0=time.perf_counter()
    x=tok(r['title']+'. '+r['body'][:3000], truncation=True, max_length=512, return_tensors='pt').to('cuda')
    with torch.no_grad():
        y=model.generate(**x, max_new_tokens=96, num_beams=4, no_repeat_ngram_size=3)
    pred=tok.decode(y[0], skip_special_tokens=True)
    ms=(time.perf_counter()-t0)*1000
    out.write(json.dumps({'id':r['id'],'pred':pred,'latency_ms':round(ms,1),'label':r['label']},ensure_ascii=False)+'\n')
    out.flush()
    if i%100==0: print(i, flush=True)
print('done')
