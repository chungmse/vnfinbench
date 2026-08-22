#!/usr/bin/env python3
"""Fine-tune PhoBERT cho filtering (2 lop) / categorization (N lop) tren .125.
Dung: python train_phobert.py --task categorization --train data/cat_train.jsonl --val data/cat_val.jsonl
Input JSONL: {"id","title","body","label"}. Checkpoint: ckpt/<task>/ (tu resume).
"""
import argparse, json, glob
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--task', required=True)
ap.add_argument('--train', required=True); ap.add_argument('--val', required=True)
ap.add_argument('--model', default='vinai/phobert-base-v2')
ap.add_argument('--epochs', type=int, default=5)
ap.add_argument('--bs', type=int, default=32)
ap.add_argument('--maxlen', type=int, default=256)
a = ap.parse_args()

def load(p):
    rows=[json.loads(l) for l in open(p)]
    return [{'text': (r.get('title','')+' . '+r.get('body',''))[:2000], 'label_str': str(r['label'])} for r in rows]
tr, va = load(a.train), load(a.val)
labels = sorted({r['label_str'] for r in tr})
l2i = {l:i for i,l in enumerate(labels)}
json.dump(labels, open(f'ckpt-{a.task}-labels.json','w'), ensure_ascii=False)

tok = AutoTokenizer.from_pretrained(a.model)
def enc(rows):
    d = Dataset.from_list([{'text':r['text'],'label':l2i[r['label_str']]} for r in rows])
    return d.map(lambda b: tok(b['text'], truncation=True, max_length=a.maxlen), batched=True)
dtr, dva = enc(tr), enc(va)

model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=len(labels))
def metrics(p):
    preds = np.argmax(p.predictions, -1)
    from sklearn.metrics import accuracy_score, f1_score
    return {'acc': accuracy_score(p.label_ids, preds), 'macro_f1': f1_score(p.label_ids, preds, average='macro')}
args = TrainingArguments(output_dir=f'ckpt/{a.task}', num_train_epochs=a.epochs,
    per_device_train_batch_size=a.bs, learning_rate=2e-5,
    eval_strategy='epoch', save_strategy='epoch', save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model='macro_f1', fp16=True, report_to=[])
tr_ = Trainer(model=model, args=args, train_dataset=dtr, eval_dataset=dva,
              compute_metrics=metrics, processing_class=tok)
ck = sorted(glob.glob(f'ckpt/{a.task}/checkpoint-*'))
tr_.train(resume_from_checkpoint=ck[-1] if ck else None)
tr_.save_model(f'ckpt/{a.task}/best')
print('saved', f'ckpt/{a.task}/best')
