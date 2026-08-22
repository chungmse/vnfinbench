#!/usr/bin/env python3
"""Fine-tune BARTpho-syllable cho tom tat (article -> sapo) tren .125. Resume tu checkpoint."""
import json, glob
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq)

MODEL='vinai/bartpho-syllable'
tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForSeq2SeqLM.from_pretrained(MODEL)

def load(p):
    rows=[json.loads(l) for l in open(p)]
    return Dataset.from_list([{'src': r['title']+'. '+r['body'][:3000], 'tgt': r['label']} for r in rows])
def enc(b):
    x=tok(b['src'], truncation=True, max_length=512)
    y=tok(text_target=b['tgt'], truncation=True, max_length=96)
    x['labels']=y['input_ids']
    return x
dtr=load('data/summ_train_50k.jsonl').map(enc,batched=True,remove_columns=['src','tgt'])
dva=load('data/summ_val.jsonl').map(enc,batched=True,remove_columns=['src','tgt'])

args=Seq2SeqTrainingArguments(output_dir='ckpt/summ_bartpho', num_train_epochs=2,
    per_device_train_batch_size=16, per_device_eval_batch_size=16,
    learning_rate=3e-5, fp16=True, eval_strategy='steps', eval_steps=1500,
    save_strategy='steps', save_steps=1500, save_total_limit=2,
    logging_steps=200, report_to=[], predict_with_generate=False)
tr=Seq2SeqTrainer(model=model,args=args,train_dataset=dtr,eval_dataset=dva,
    data_collator=DataCollatorForSeq2Seq(tok,model=model),processing_class=tok)
ck=sorted(glob.glob('ckpt/summ_bartpho/checkpoint-*'))
tr.train(resume_from_checkpoint=ck[-1] if ck else None)
tr.save_model('ckpt/summ_bartpho/best')
print('saved')
