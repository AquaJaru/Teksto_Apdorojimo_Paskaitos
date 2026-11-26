import argparse,json,re,time
from collections import Counter,defaultdict
from pathlib import Path
from core import restore,answer,tokens,load_rows
from retrieval import Retriever

def is_correct(gen,gold,eos):return bool(eos) and tokens(gen)==tokens(gold)

def score(cases,with_retrieval=False):
    a=[c for c in cases if c['answerable']];u=[c for c in cases if not c['answerable']]
    out=dict(n_questions=len(cases),n_answerable=len(a),n_unanswerable=len(u),
        em_correct=sum(is_correct(c['generated'],c['answer'],c['eos']) for c in a),em_total=len(a),
        abstention_correct=sum(c['eos'] and c['generated']=='unknown' for c in u),abstention_total=len(u),
        unsupported_count=sum(c['generated'] not in ('unknown','') for c in u),unsupported_total=len(u),
        incomplete_count=sum(not c['eos'] for c in cases),incomplete_total=len(cases),
        wrong_answer_on_answerable=sum(not is_correct(c['generated'],c['answer'],c['eos']) and c['generated'] not in ('unknown','') for c in a),
        false_unknown_on_answerable=sum(c['generated']=='unknown' for c in a))
    if with_retrieval:
        out.update(retrieval_correct=sum(c['retrieved_id']==c['document_id'] for c in cases),retrieval_total=len(cases))
        bad=[c for c in a if not is_correct(c['generated'],c['answer'],c['eos'])]
        out['answerable_errors_wrong_card']=sum(c['retrieved_id']!=c['document_id'] for c in bad)
        out['answerable_errors_right_card']=sum(c['retrieved_id']==c['document_id'] for c in bad)
    return out

def per_field(cases):
    g=defaultdict(list)
    for c in cases:g[c['field']].append(c)
    return {f:dict(answerable=f"{sum(is_correct(c['generated'],c['answer'],c['eos']) for c in v if c['answerable'])}/{sum(c['answerable'] for c in v)}",
                   unanswerable=f"{sum(c['generated']=='unknown' for c in v if not c['answerable'])}/{sum(not c['answerable'] for c in v)}") for f,v in sorted(g.items())}

FIELD_RE=lambda f:re.compile(r'(?:^|;\s*)'+re.escape(f)+r'\s*:\s*([^;]+?)\s*;')
def read_field(context,field):
    m=FIELD_RE(field).search(context);return m.group(1) if m else 'unknown'

def run(checkpoint,data,split,lang,query_mode='full'):
    model,vocab,_=restore(checkpoint);data=Path(data)
    rows=load_rows(data/f'{split}.jsonl');train=load_rows(data/'train.jsonl')
    docs={d['id']:d for d in (json.loads(x) for x in (data/'documents.jsonl').read_text(encoding='utf-8').splitlines())}
    retriever=Retriever(data)
    majority={f:Counter(r['answer'] for r in train if r['field']==f).most_common(1)[0][0] for f in {r['field'] for r in train}}
    results={};all_cases={}
    def gpt_cases(ctx_fn,retr=False):
        cs=[]
        for r in rows:
            extra={}
            if retr:
                q=r['query'][lang]
                ranked=retriever.search(q,k=3,title_only=(query_mode=='title_only'))
                top=ranked[0]['document'];ctx=top['context']
                extra=dict(retrieved_id=top['id'],retrieved_score=ranked[0]['score'],top3=[(x['document']['id'],round(x['score'],3)) for x in ranked])
            else:ctx=ctx_fn(r)
            gen,eos=answer(model,vocab,ctx,r['question'][lang])
            cs.append(dict(id=r['id'],document_id=r['document_id'],field=r['field'],answerable=r['answerable'],answer=r['answer'],context=ctx,generated=gen,eos=eos,**extra))
        return cs
    t=time.perf_counter()
    all_cases['no_source']=gpt_cases(lambda r:'')
    all_cases['correct_card']=gpt_cases(lambda r:docs[r['document_id']]['context'])
    all_cases['retrieved_card']=gpt_cases(None,retr=True)
    all_cases['training_majority']=[dict(id=r['id'],document_id=r['document_id'],field=r['field'],answerable=r['answerable'],answer=r['answer'],context='',generated=majority[r['field']],eos=True) for r in rows]
    all_cases['field_reading_rule']=[dict(id=r['id'],document_id=r['document_id'],field=r['field'],answerable=r['answerable'],answer=r['answer'],context=docs[r['document_id']]['context'],generated=read_field(docs[r['document_id']]['context'],r['field']),eos=True) for r in rows]
    all_cases['field_reading_rule_on_retrieved']=[dict(c,generated=read_field(c['context'],c['field']),eos=True) for c in all_cases['retrieved_card']]
    for k,cs in all_cases.items():
        results[k]=score(cs,with_retrieval=k in('retrieved_card','field_reading_rule_on_retrieved'));results[k]['per_field']=per_field(cs)
    return dict(split=split,checkpoint=str(checkpoint),query_mode=query_mode,seconds=round(time.perf_counter()-t,1),majority=majority,results=results),all_cases

def pct(x,n):return 'netaikoma' if not n else f'{x}/{n} ({100*x/n:.1f}%)'
def print_table(rep):
    print(f"{'condition':32} {'retrieval_top1':18} {'answerable_EM':18} {'abstention':16} {'unsupported':16} {'incomplete':16}")
    for k,r in rep['results'].items():
        print(f"{k:32} {pct(r.get('retrieval_correct',0),r.get('retrieval_total',0)):18} {pct(r['em_correct'],r['em_total']):18} {pct(r['abstention_correct'],r['abstention_total']):16} {pct(r['unsupported_count'],r['unsupported_total']):16} {pct(r['incomplete_count'],r['incomplete_total']):16}")

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--data',default='data/research');p.add_argument('--split',default='validation',choices=['validation','test']);p.add_argument('--lang',default='LT');p.add_argument('--query-mode',default='full',choices=['full','title_only']);p.add_argument('--out',required=True)
    a=p.parse_args();rep,cases=run(a.checkpoint,a.data,a.split,a.lang,a.query_mode)
    Path(a.out).write_text(json.dumps(rep,indent=2,ensure_ascii=False),encoding='utf-8');Path(a.out).with_suffix('.cases.json').write_text(json.dumps(cases,indent=1,ensure_ascii=False),encoding='utf-8')
    print_table(rep)
