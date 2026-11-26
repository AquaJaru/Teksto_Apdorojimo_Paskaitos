import json,sys
from pathlib import Path
from core import restore,answer,tokens
from retrieval import Retriever

Q=[
 ('Improving Transportation Planning Using Machine Learning','Kiek literatūros nuorodų nurodyta?','template','10.4018/978-1-7998-9220-5.ch184','40'),
 ('Michel Electrons Liquid Argon TPCs','Koks publikacijos tipas nurodytas?','template','10.2172/1996940','proceedings-article'),
 ('Trustworthy Machine Learning with Noisy Labels','Kuris publikavimo mėnuo nurodytas?','template','10.1007/978-981-96-9396-2_2','8'),
 ('Quantum Machine Learning and Deep Learning Fundamentals','Kiek literatūros nuorodų nurodyta?','template','10.3390/make7030075','104'),
 ('generative art machine learning deep learning','Kurie publikacijos metai nurodyti?','template','10.70593/978-81-981271-4-3_5','unknown'),
 ('Reducing MEG interference','Kiek literatūros nuorodų nurodyta?','template','10.1016/j.mlwa.2023.100462','unknown'),
 ('Mining commit messages software refactorings','Kokiais metais išleista ši publikacija?','free','10.1016/j.mlwa.2022.100316','unknown'),
 ('RDR and Machine Learning','Kiek autorių parašė šį darbą?','free','10.1201/9781003126157-9','2'),
 ('Diabetes Prediction Based on Machine Learning','Kas yra pirmasis autorius?','free','10.1145/3635638.3635643','unknown'),
 ('Reducing MEG interference','Ar straipsnis turi licenciją?','free','10.1016/j.mlwa.2023.100462','yes'),
]
EDITS=[
 ('10.1201/9781003126157-9','year','2021','2019','Kurie publikacijos metai nurodyti?'),
 ('10.1201/9781003126157-9','authors','2','7','Kiek autorių nurodyta?'),
 ('10.2172/1996940','type','proceedings-article','journal-article','Koks publikacijos tipas nurodytas?'),
]
def main(ckpt,out):
    model,vocab,_=restore(ckpt);r=Retriever('data/research');docs={d['id']:d for d in r.docs};rows=[]
    for kw,q,kind,doc_id,exp in Q:
        ranked=r.search(kw+' — '+q);top=ranked[0]['document'];gen,eos=answer(model,vocab,top['context'],q)
        rows.append(dict(kind=kind,keywords=kw,question=q,expected_doc=doc_id,retrieved_doc=top['id'],retrieval_ok=top['id']==doc_id,score=round(ranked[0]['score'],3),source_url=top['source_url'],context=top['context'],expected=exp,generated=gen,eos=eos,correct=eos and tokens(gen)==tokens(exp)))
    edits=[]
    for doc_id,field,old,new,q in EDITS:
        ctx=docs[doc_id]['context'];assert f'{field} : {old} ;' in ctx;edited=ctx.replace(f'{field} : {old} ;',f'{field} : {new} ;')
        g0,_=answer(model,vocab,ctx,q);g1,e1=answer(model,vocab,edited,q)
        edits.append(dict(doc=doc_id,field=field,original=old,edited=new,answer_original=g0,answer_edited=g1,eos=e1,follows_edit=tokens(g1)==tokens(new)))
    Path(out).write_text(json.dumps(dict(questions=rows,edited_facts=edits),indent=1,ensure_ascii=False),encoding='utf-8')
    for x in rows:print(f"{x['kind']:8} ret={'OK ' if x['retrieval_ok'] else 'BAD'} exp={x['expected']:>20} gen={x['generated']:>22} {'✓' if x['correct'] else '✗'}  {x['question']}")
    for e in edits:print('EDIT',e)
if __name__=='__main__':main(sys.argv[1],sys.argv[2])
