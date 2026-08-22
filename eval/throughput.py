#!/usr/bin/env python3
"""Do throughput vLLM: gui N bai song song (concurrency C), tinh bai/phut.
Dung: python throughput.py --input data/filt_test.jsonl --n 200 --conc 16 --task filtering --max-tokens 10 --family qwen
"""
import argparse, json, time, asyncio
from openai import AsyncOpenAI

ap=argparse.ArgumentParser()
ap.add_argument('--input', required=True); ap.add_argument('--n', type=int, default=200)
ap.add_argument('--conc', type=int, default=16); ap.add_argument('--task', default='filtering')
ap.add_argument('--max-tokens', type=int, default=10)
ap.add_argument('--family', default='qwen')
ap.add_argument('--base-url', default='http://localhost:8000/v1')
a=ap.parse_args()
tpl=open(f'prompts/{a.task}.txt' if a.task!='entity_linking' else 'prompts/entity_linking_zeroshot.txt').read()
extra={"chat_template_kwargs":{"enable_thinking":False}} if a.family=='qwen' else {"reasoning_effort":"low"}
rows=[json.loads(l) for l in open(a.input)][:a.n]
client=AsyncOpenAI(base_url=a.base_url, api_key='none')
sem=asyncio.Semaphore(a.conc)
async def one(r):
    async with sem:
        await client.chat.completions.create(model='llm', max_tokens=a.max_tokens, temperature=0,
            messages=[{"role":"user","content":tpl.format(title=r.get('title',''),body=r.get('body','')[:3000],labels='',aliases='')}],
            extra_body=extra)
async def main():
    t0=time.perf_counter()
    await asyncio.gather(*[one(r) for r in rows])
    dt=time.perf_counter()-t0
    print(json.dumps({'task':a.task,'n':len(rows),'conc':a.conc,'seconds':round(dt,1),
                      'per_min':round(len(rows)/dt*60,1)}))
asyncio.run(main())
