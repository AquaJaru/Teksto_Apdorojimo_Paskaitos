from pathlib import Path
from dataclasses import dataclass,asdict
from collections import Counter
import torch,math,re,json,time,random,hashlib,os
from torch import nn
import torch.nn.functional as F

def tokens(s):
    pieces=re.findall(r'[\w.]+|[^\w\s]',s.lower(),re.UNICODE)
    return [part for word in pieces for part in (list(word) if word.isdigit() else [word])]
SPECIAL=['<pad>','<bos>','<eos>','<unk>','<context>','<question>','<answer>']
@dataclass
class Config:
    dim:int=64
    heads:int=2
    blocks:int=2
    context:int=128
    batch:int=32
    lr:float=.002
    seed:int=42

from student_model import CausalAttention as ReferenceAttention
class Block(nn.Module):
    def __init__(self,cfg,attention):
        super().__init__();self.a=attention(cfg.dim,cfg.heads);self.n1=nn.LayerNorm(cfg.dim);self.n2=nn.LayerNorm(cfg.dim);self.ff=nn.Sequential(nn.Linear(cfg.dim,2*cfg.dim),nn.GELU(),nn.Linear(2*cfg.dim,cfg.dim))
    def forward(self,x):x=x+self.a(self.n1(x));return x+self.ff(self.n2(x))
class GPT(nn.Module):
    def __init__(self,n,cfg,attention=ReferenceAttention):
        super().__init__();self.cfg=cfg;self.emb=nn.Embedding(n,cfg.dim);self.pos=nn.Embedding(cfg.context,cfg.dim);self.blocks=nn.Sequential(*[Block(cfg,attention) for _ in range(cfg.blocks)]);self.norm=nn.LayerNorm(cfg.dim);self.head=nn.Linear(cfg.dim,n)
    def hidden(self,x):return self.norm(self.blocks(self.emb(x)+self.pos(torch.arange(x.shape[1],device=x.device))))
    def forward(self,x):return self.head(self.hidden(x))

def load_rows(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
def vocab_fit(train,lang):return SPECIAL+sorted(set('0123456789')|{t for r in train for t in tokens(r['source']+' '+r['question'][lang]+' '+r['answer'])})
def prefix(vocab,source,question):
    table={w:i for i,w in enumerate(vocab)}
    return [1,4]+[table.get(t,3) for t in tokens(source)]+[5]+[table.get(t,3) for t in tokens(question)]+[6]
def pack(rows,vocab,lang,cfg):
    table={t:i for i,t in enumerate(vocab)};out=[]
    for r in rows:
        pre=prefix(vocab,r['source'],r['question'][lang]);answer=[table.get(t,3) for t in tokens(r['answer'])]+[2]
        if 3 in answer:raise ValueError('Unseen target vocabulary; fix split/data before training')
        if len(pre)+len(answer)>cfg.context:raise ValueError(f'Context overflow: {len(pre)+len(answer)}')
        seq=pre+answer;inp=seq[:-1];y=[-100]*(len(pre)-1)+answer
        out.append((inp,y))
    length=max(len(a) for a,b in out);x=torch.zeros(len(out),length,dtype=torch.long);y=torch.full_like(x,-100)
    for i,(a,b) in enumerate(out):x[i,:len(a)]=torch.tensor(a);y[i,:len(b)]=torch.tensor(b)
    return x,y

def answer(model,vocab,source,question,limit=8):
    ids=prefix(vocab,source,question);chosen=[];stop=False
    if len(ids)+limit>model.cfg.context:raise ValueError('Input too long; shorten source explicitly')
    model.eval()
    with torch.no_grad():
        for _ in range(limit):
            i=int(model(torch.tensor([ids]))[0,-1].argmax());ids.append(i)
            if i==2:stop=True;break
            chosen.append(vocab[i])
    return (''.join(chosen) if chosen and all(c.isdigit() for c in chosen) else ' '.join(chosen)),stop

def metrics(cases):
    correct=lambda c:c['eos'] and tokens(c['generated'])==tokens(c['answer'])
    a=[r for r in cases if r['answerable']];u=[r for r in cases if not r['answerable']]
    div=lambda x,n:x/n if n else None
    return dict(n=len(cases),answerable=len(a),unanswerable=len(u),em=sum(map(correct,cases))/len(cases),answerable_em=div(sum(map(correct,a)),len(a)),unanswerable_em=div(sum(map(correct,u)),len(u)),false_abstention=div(sum(r['generated']=='unknown' for r in a),len(a)),unsupported=div(sum(r['generated']!='unknown' for r in u),len(u)))
def evaluate(model,vocab,rows,lang,context_override=None):
    cases=[]
    for r in rows:
        text=r['source'] if context_override is None else context_override(r)
        a,eos=answer(model,vocab,text,r['question'][lang]);cases.append(dict(r,generated=a,eos=eos))
    return metrics(cases),cases

def train(data,out,lang='EN',minutes=30,max_steps=10000,cfg=None,attention=ReferenceAttention):
    cfg=cfg or Config();random.seed(cfg.seed);torch.manual_seed(cfg.seed);torch.set_num_threads(1)
    out=Path(out);out.mkdir(parents=True,exist_ok=True);data=Path(data)
    rows=load_rows(data/'train.jsonl');valid=load_rows(data/'validation.jsonl');vocab=vocab_fit(rows,lang)
    x,y=pack(rows,vocab,lang,cfg);vx,vy=pack(valid,vocab,lang,cfg)
    model=GPT(len(vocab),cfg,attention);optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=.01)
    start=time.perf_counter();history=[];best=(-1,float('inf'));patience=0;training_start=start;warm_time=None
    for step in range(1,max_steps+1):
        model.train();ix=torch.randint(len(rows),(cfg.batch,));batch_x=x[ix];batch_y=y[ix]
        hidden=model.hidden(batch_x);mask=batch_y!=-100
        loss=F.cross_entropy(model.head(hidden[mask]),batch_y[mask]);optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step==50:warm_time=(time.perf_counter()-start)/50;print('50-step calibration:',round(warm_time,4),'s/step;',round(warm_time*max_steps/60,1),'min for step cap',flush=True)
        if step%500==0 or step==max_steps or time.perf_counter()-start>minutes*60-30:
            model.eval()
            with torch.no_grad():
                h=model.hidden(vx);m=vy!=-100;vl=float(F.cross_entropy(model.head(h[m]),vy[m]))
            met,cases=evaluate(model,vocab,valid,lang)
            elapsed=time.perf_counter()-start;history.append(dict(step=step,seconds=elapsed,train_loss=float(loss.detach()),validation_loss=vl,validation=met))
            print('Step',step,'elapsed',round(elapsed,1),'val',met,flush=True)
            score=(met['em'],-vl)
            if score>(best[0],-best[1]):
                best=(met['em'],vl);patience=0
                torch.save(dict(state=model.state_dict(),vocab=vocab,config=asdict(cfg),lang=lang,step=step),out/'best.pt')
            else:patience+=1
            (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
            if met['em']>=.98 or patience>=4 or elapsed>minutes*60-30:break
    duration=time.perf_counter()-start
    report=dict(config=asdict(cfg),parameters=sum(p.numel() for p in model.parameters()),vocab=len(vocab),train_rows=len(rows),validation_rows=len(valid),steps=step,wall_seconds=duration,warm_s_per_step=warm_time,training_device='cpu',threads=1,best_validation_em=best[0],data_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in data.glob('*.jsonl')},python=__import__('sys').version,torch=torch.__version__)
    (out/'training.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return report

def restore(path,attention=ReferenceAttention):
    torch.set_num_threads(1);s=torch.load(path,map_location='cpu',weights_only=True);m=GPT(len(s['vocab']),Config(**s['config']),attention);m.load_state_dict(s['state']);m.eval();return m,s['vocab'],s['lang']
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--lang',default='EN');p.add_argument('--minutes',type=float,default=30);p.add_argument('--max-steps',type=int,default=10000);p.add_argument('--dim',type=int,default=64);p.add_argument('--heads',type=int,default=2);p.add_argument('--seed',type=int,default=42);a=p.parse_args();print(train(a.data,a.out,a.lang,a.minutes,a.max_steps,cfg=Config(dim=a.dim,heads=a.heads,seed=a.seed)))
