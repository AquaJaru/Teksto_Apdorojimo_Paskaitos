import argparse,json,random,shutil
from pathlib import Path
from core import train,Config,load_rows

PREFIX='Reference metadata. Title : '
def shuffle_fields(source,rng):
    assert source.startswith(PREFIX) and source.endswith(' ;')
    parts=source[len(PREFIX):-2].split(' ; ')
    title,fields=parts[0],parts[1:]
    rng.shuffle(fields)
    return PREFIX+' ; '.join([title]+fields)+' ;'

def make_data(src,dst,k,seed):
    src,dst=Path(src),Path(dst);dst.mkdir(parents=True,exist_ok=True);rng=random.Random(seed)
    for name in ['validation.jsonl','test.jsonl','documents.jsonl','questions.json']:shutil.copy(src/name,dst/name)
    rows=load_rows(src/'train.jsonl');out=list(rows)
    for r in rows:
        for _ in range(k):out.append(dict(r,source=shuffle_fields(r['source'],rng)))
    (dst/'train.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in out)+'\n',encoding='utf-8')
    return len(rows),len(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/research');p.add_argument('--out',required=True);p.add_argument('--k',type=int,default=2);p.add_argument('--seed',type=int,default=42);p.add_argument('--minutes',type=float,default=30)
    a=p.parse_args();aug=Path(a.out)/'aug_data';print('rows',make_data(a.data,aug,a.k,a.seed))
    print(train(aug,a.out,'LT',a.minutes,10000,cfg=Config(dim=96,heads=4,seed=a.seed)))
