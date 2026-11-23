"""Provided evaluation/generation harness. References never enter model input."""
import argparse,json,re,time,hashlib
from pathlib import Path
from rouge_score.rouge_scorer import RougeScorer

from text_units import sentences
def lead(text,k=2):return ' '.join(sentences(text)[:k])
def score(predictions,references):
    scorer=RougeScorer(['rouge1','rouge2','rougeL'],use_stemmer=True)
    if len(predictions)!=len(references) or not predictions:raise ValueError('Need equal nonempty paired results.')
    rows=[{k:v.fmeasure for k,v in scorer.score(r,p).items()} for p,r in zip(predictions,references)]
    return {k:sum(r[k] for r in rows)/len(rows) for k in rows[0]}
def generate(texts,model_path,max_new_tokens=48,batch_size=4):
    import os
    os.environ.setdefault('HF_HOME',str(Path(model_path).parent/'hf-cache'))
    import torch
    from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
    torch.set_num_threads(1)
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    model=AutoModelForSeq2SeqLM.from_pretrained(model_path,local_files_only=True,use_safetensors=True).eval()
    results=[];truncated=[];start=time.perf_counter()
    for i in range(0,len(texts),batch_size):
        part=texts[i:i+batch_size];full=tokenizer(['summarize: '+t for t in part],truncation=False)['input_ids']
        truncated.extend(len(ids)>512 for ids in full)
        inputs=tokenizer(['summarize: '+t for t in part],return_tensors='pt',padding=True,truncation=True,max_length=512)
        with torch.no_grad():out=model.generate(**inputs,max_new_tokens=max_new_tokens,num_beams=2,do_sample=False)
        results.extend(tokenizer.batch_decode(out,skip_special_tokens=True))
    return results,dict(seconds=time.perf_counter()-start,truncated=sum(truncated),batch_size=batch_size,max_input_tokens=512,max_new_tokens=max_new_tokens,num_beams=2,do_sample=False)
def load(path):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for r in rows:assert all(k in r for k in ['id','source','reference','split'])
    assert len({r['id'] for r in rows})==len(rows)
    assert len({r['source'].lower().strip() for r in rows})==len(rows)
    groups={}
    for r in rows:
        group=r.get('group_id',r.get('url',r['id']));assert group not in groups or groups[group]==r['split'];groups[group]=r['split']
    return rows
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--model',required=True);p.add_argument('--encoder',required=True);p.add_argument('--split',choices=['validation','test'],default='validation');p.add_argument('--limit',type=int,default=8);p.add_argument('--max-new-tokens',type=int,default=48);p.add_argument('--out',type=Path,default=Path('results.json'));p.add_argument('--reference-method',action='store_true',help='Instructor only: requires separate reference module, absent from student ZIP.');a=p.parse_args()
    allrows=load(a.data);rows=[r for r in allrows if r['split']==a.split][:a.limit]
    if a.split=='test':assert {r['id'] for r in rows}=={'OZ-09','OZ-10','OZ-11','OZ-12'},'Final Oz comparison requires all4 fixed test excerpts.'
    if a.reference_method:
        from reference_embedding import extractive
    else:from student_methods import extractive
    from embedding_tools import SentenceEncoder
    encoder=SentenceEncoder(a.encoder)
    outputs={};runtime={}
    for name,fn in [('lead2',lead),('embedding',lambda text:extractive(text,encoder.encode))]:
        start=time.perf_counter();outputs[name]=[fn(r['source']) for r in rows];runtime[name]={'seconds':time.perf_counter()-start}
    outputs['generative'],runtime['generative']=generate([r['source'] for r in rows],a.model,a.max_new_tokens)
    results={name:dict(metrics=score(pred,[r['reference'] for r in rows]),runtime=runtime[name],mean_words=sum(len(s.split()) for s in pred)/len(pred)) for name,pred in outputs.items()}
    a.out.write_text(json.dumps(dict(data_sha256=hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),encoder_metadata=json.loads((Path(a.encoder)/'TNKA_MODEL.json').read_text()),encoder_sha256=hashlib.sha256((Path(a.encoder)/'model.safetensors').read_bytes()).hexdigest(),encoder_truncated_sentences=encoder.truncated,model_metadata=json.loads((Path(a.model)/'TNKA_MODEL.json').read_text()),model_sha256=hashlib.sha256((Path(a.model)/'model.safetensors').read_bytes()).hexdigest(),split=a.split,ids=[r['id'] for r in rows],results=results,outputs=outputs),ensure_ascii=False,indent=2));print(json.dumps(results,indent=2))
