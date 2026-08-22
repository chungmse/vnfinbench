#!/usr/bin/env python3
"""Chấm điểm predictions JSONL cho cả 4 task + thống kê latency.
Dùng: python eval.py --task filtering results/filtering_qwen3_zeroshot.jsonl
"""
import argparse, json, re, statistics as st

ap = argparse.ArgumentParser()
ap.add_argument('--task', required=True, choices=['filtering','categorization','entity_linking','summarization'])
ap.add_argument('files', nargs='+')
a = ap.parse_args()

def norm(s): return re.sub(r'\s+',' ', (s or '').strip().lower())

for f in a.files:
    rows = [json.loads(l) for l in open(f)]
    lat = [r['latency_ms'] for r in rows if r.get('latency_ms')]
    res = {"file": f, "n": len(rows)}
    if a.task in ('filtering','categorization'):
        pairs = [(norm(r['label']), norm(r['pred'])) for r in rows if r.get('label') is not None]
        # pred có thể dài -> lấy nhãn xuất hiện đầu tiên trong pred
        labels = sorted({g for g,_ in pairs})
        by_len = sorted(labels, key=len, reverse=True)   # nhan dai truoc: "thi truong chung khoan" > "thi truong"
        tp = {l:0 for l in labels}; fp = dict(tp); fn = dict(tp); correct=0
        for g,p in pairs:
            hit = next((l for l in by_len if l in p), None)
            if hit == g: correct+=1; tp[g]+=1
            else:
                fn[g]+=1
                if hit: fp[hit]+=1
        f1s=[]
        for l in labels:
            pr = tp[l]/(tp[l]+fp[l]) if tp[l]+fp[l] else 0
            rc = tp[l]/(tp[l]+fn[l]) if tp[l]+fn[l] else 0
            f1s.append(2*pr*rc/(pr+rc) if pr+rc else 0)
        res.update(acc=round(correct/len(pairs),4), macro_f1=round(sum(f1s)/len(f1s),4))
    elif a.task == 'entity_linking':
        P=R=0; n=0
        for r in rows:
            gold = set(json.loads(r['label'])) if isinstance(r['label'],str) else set(r['label'] or [])
            m = re.search(r'\{.*\}', r['pred'] or '', re.S)
            try: pred = set(json.loads(m.group(0)).get('tickers',[])) if m else set()
            except Exception: pred = set()
            pred = {t.upper() for t in pred}
            if not gold and not pred: continue
            n+=1
            P += len(gold&pred)/len(pred) if pred else 0
            R += len(gold&pred)/len(gold) if gold else 0
        p,rc = P/n, R/n
        res.update(precision=round(p,4), recall=round(rc,4), f1=round(2*p*rc/(p+rc),4) if p+rc else 0)
    elif a.task == 'summarization':
        from rouge_score import rouge_scorer
        sc = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=False)
        agg = {k:[] for k in ('rouge1','rouge2','rougeL')}
        for r in rows:
            s = sc.score(r['label'], r['pred'] or '')
            for k in agg: agg[k].append(s[k].fmeasure)
        res.update({k: round(sum(v)/len(v),4) for k,v in agg.items()})
    if lat:
        res.update(lat_avg_ms=round(st.mean(lat),1), lat_p95_ms=round(sorted(lat)[int(len(lat)*0.95)-1],1))
    print(json.dumps(res, ensure_ascii=False))
