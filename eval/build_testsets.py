#!/usr/bin/env python3
"""Dung test/train set 4 task tu du lieu local CafeF. Chay tren laptop.
Output: data_out/*.jsonl (nho, de day len server)."""
import json, random, glob, os, re, unicodedata

SRC='/Users/chungnv/Downloads/paper2/data'
OUT='data_out'; os.makedirs(OUT, exist_ok=True)
random.seed(42)

FIN_CATS={'Thị trường chứng khoán','Tài chính - ngân hàng','Tài chính quốc tế',
          'Kinh tế vĩ mô - Đầu tư','Doanh nghiệp','Bất động sản','Thị trường'}

def norm_words(s): return len((s or '').split())

# ---- quet toan bo merged, thu thap ung vien theo task ----
cat_pool={}; summ_pool=[]; el_pool=[]
n=0
for f in sorted(glob.glob(f'{SRC}/cafef/merged/part-*.jsonl')):
    for l in open(f):
        d=json.loads(l); n+=1
        cat=(d.get('category') or '').strip()
        title=(d.get('title') or '').strip(); body=(d.get('content') or '').strip()
        sapo=(d.get('sapo') or '').strip()
        if not title or norm_words(body)<80: continue
        rec={'id':d['id'],'title':title,'body':body[:6000]}
        # categorization: chi lay cat pho bien (chot nhan sau)
        if cat:
            cat_pool.setdefault(cat,[])
            if len(cat_pool[cat])<4000: cat_pool[cat].append({**rec,'label':cat})
        # summarization: sapo du dai, khong trung title
        if 20<=norm_words(sapo)<=80 and sapo[:40]!=title[:40] and len(summ_pool)<200000:
            summ_pool.append({**rec,'label':sapo})
        # EL: bai chung khoan/doanh nghiep + tag co dang ma CK
        if cat in ('Thị trường chứng khoán','Doanh nghiệp'):
            tick_tags=[t.strip() for t in (d.get('tags') or []) if re.fullmatch(r'[A-Z]{3}', (t or '').strip())]
            if tick_tags and len(el_pool)<100000:
                el_pool.append({**rec,'tag_tickers':tick_tags})
print('scanned',n)
print('cats:', {k:len(v) for k,v in sorted(cat_pool.items(), key=lambda x:-len(x[1]))})
print('summ_pool',len(summ_pool),'el_pool',len(el_pool))

# ---- categorization: 8 nhan tai chinh-kinh te ro nhat, du mau ----
CAT_KEEP=[c for c in cat_pool if c in FIN_CATS or c in ('Kinh tế số','Xã hội','Sống')]
CAT_KEEP=[c for c in CAT_KEEP if len(cat_pool[c])>=1500]
print('cat_keep:',CAT_KEEP)
tr,va,te=[],[],[]
for c in CAT_KEEP:
    rows=cat_pool[c][:]; random.shuffle(rows)
    te+=rows[:125]; va+=rows[125:250]; tr+=rows[250:2250]
for name,rows in [('cat_train',tr),('cat_val',va),('cat_test',te)]:
    random.shuffle(rows)
    with open(f'{OUT}/{name}.jsonl','w') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
print('categorization:',len(tr),len(va),len(te))

# ---- summarization: 1000 test ----
random.shuffle(summ_pool)
with open(f'{OUT}/summ_test.jsonl','w') as f:
    for r in summ_pool[:1000]: f.write(json.dumps(r,ensure_ascii=False)+'\n')

# ---- EL: 600 ung vien cho gan nhan (auto tag_tickers -> can soat tay) ----
random.shuffle(el_pool)
with open(f'{OUT}/el_candidates.jsonl','w') as f:
    for r in el_pool[:600]: f.write(json.dumps(r,ensure_ascii=False)+'\n')

# ---- filtering: copy splits co san ----
import shutil
for s in ('train','val','test'):
    rows=[json.loads(l) for l in open(f'{SRC}/interesting/datasets/{s}.jsonl')]
    with open(f'{OUT}/filt_{s}.jsonl','w') as f:
        for d in rows:
            f.write(json.dumps({'id':d['id'],'title':d.get('title',''),
              'body':((d.get('sapo') or '')+'\n'+(d.get('content') or ''))[:6000],
              'label':'co' if d['label']==1 else 'khong'},ensure_ascii=False)+'\n')
print('filtering copied')
