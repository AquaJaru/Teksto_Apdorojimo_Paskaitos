import json
from pathlib import Path

def _load(p):
    p=Path(p);return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None

def _strip(res):
    return {k:{a:b for a,b in v.items() if a!='per_field'}|{'per_field':v.get('per_field',{})} for k,v in res.items()}

def build(root,checkpoint):
    root=Path(root);out=Path(checkpoint).parent
    rep=dict(available=[])
    tr=_load(out/'training.json')
    if tr:
        rep['model']={k:tr.get(k) for k in ['config','parameters','vocab','train_rows','validation_rows','steps','wall_seconds','best_validation_em','python','torch','training_device','threads']}
        rep['available'].append('model')
    h=_load(out/'history.json')
    if h:rep['history']=[dict(step=r['step'],train=r['train_loss'],val=r['validation_loss'],em=r['validation']['answerable_em']) for r in h];rep['available'].append('history')
    for split in ['validation','test']:
        e=_load(out/f'exp_{split}.json')
        if e:rep[split]=dict(results=_strip(e['results']),majority=e.get('majority'));rep['available'].append(split)
    base=_load(out/'exp_validation.json');aug=_load(root/'outputs/repair_aug_s42/exp_validation.json');s7=_load(root/'outputs/baseline_s7/exp_validation.json')
    if base and aug:
        rep['repair']=dict(prediction=(out/'PREDICTION.txt').read_text(encoding='utf-8') if (out/'PREDICTION.txt').exists() else '',
            base=base['results']['correct_card'],after=aug['results']['correct_card'],
            seed7=s7['results']['correct_card'] if s7 else None,
            history_after=[dict(step=r['step'],val=r['validation_loss'],em=r['validation']['answerable_em']) for r in (_load(root/'outputs/repair_aug_s42/history.json') or [])])
        rep['available'].append('repair')
    cases=_load(out/'exp_validation.cases.json')
    if cases:
        tok=lambda s:s.replace(' ','')
        gpt=[c for c in cases['correct_card'] if c['answerable'] and tok(c['generated'])!=tok(c['answer'])]
        ret=[c for c in cases['retrieved_card'] if c['retrieved_id']!=c['document_id'] and c['generated'] not in ('unknown',) and not c['answerable']]
        ret=ret or [c for c in cases['retrieved_card'] if c['retrieved_id']!=c['document_id']]
        gpt.sort(key=lambda c:(c['field']!='references',-len(c['answer'])))
        rep['errors']=dict(gpt=gpt[:3],retrieval=ret[:2],gpt_total=len(gpt),retrieval_total=sum(c['retrieved_id']!=c['document_id'] for c in cases['retrieved_card']))
        rep['available'].append('errors')
    pq=_load(out/'personal_questions.json')
    if pq:rep['personal']=pq;rep['available'].append('personal')
    return rep
