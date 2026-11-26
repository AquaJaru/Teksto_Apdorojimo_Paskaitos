from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
import json,time,argparse,threading,html
from core import restore,answer,tokens
from retrieval import Retriever
import report_data

class Pipeline:
    def __init__(self,root,topic,checkpoint,encoder,lang=None):
        self.root=Path(root);self.topic=topic;self.model,self.vocab,self.lang=restore(checkpoint);self.lang=lang or self.lang
        self.retriever=Retriever(self.root/'data'/topic,encoder);self.questions=json.loads((self.root/'data'/topic/'questions.json').read_text(encoding='utf-8'));self.lock=threading.Lock()
    def ask(self,keywords,field,mode='retrieval',selected_id=None):
        if field not in self.questions or not 1<=len(keywords)<=400:raise ValueError('Choose a supported question and enter a short title or keywords.')
        question=self.questions[field][0 if self.lang=='EN' else 1];query=keywords+' — '+question;t=time.perf_counter()
        with self.lock:
            ranked=self.retriever.search(query);search_ms=(time.perf_counter()-t)*1000
            if mode=='retrieval':doc=ranked[0]['document'];context=doc['context']
            elif mode=='manual':doc=next((d for d in self.retriever.docs if d['id']==selected_id),None);context=doc['context'] if doc else ''
            elif mode=='no_source':doc=None;context=''
            else:raise ValueError('Unknown mode')
            before=time.perf_counter();generated,eos=answer(self.model,self.vocab,context,question);generation_ms=(time.perf_counter()-before)*1000
        known=set(self.vocab);prompt=['<bos>','<context>']+[t if t in known else '<unk>' for t in tokens(context)]+['<question>']+[t if t in known else '<unk>' for t in tokens(question)]+['<answer>']
        return dict(prompt_tokens=prompt,question_text=question,keywords=keywords,question=query,answer=generated or '[empty]',eos=eos,source=doc,context=context,ranked=[dict(id=r['document']['id'],title=r['document']['title'],url=r['document']['source_url'],score=r['score']) for r in ranked],search_ms=search_ms,generation_ms=generation_ms,mode=mode,parameters=sum(p.numel() for p in self.model.parameters()),vector_size=int(self.retriever.vectors.shape[1]),steps=['Question and supported field','Corpus-trained LSA retrieval','Top-1 metadata context' if context else 'No context','Tiny GPT generates answer tokens'])

def serve(pipeline,port=8904,checkpoint='outputs/research_LT/best.pt'):
    base=f'http://127.0.0.1:{port}'
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*a):pass
        def send(self,obj,status=200,kind='application/json; charset=utf-8'):
            data=obj.encode() if isinstance(obj,str) else json.dumps(obj,ensure_ascii=False).encode();self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
        def do_GET(self):
            if self.path=='/':return self.send((Path(__file__).parent/'ui.html').read_text(encoding='utf-8'),kind='text/html; charset=utf-8')
            if self.path=='/config':return self.send(dict(topic=pipeline.topic,lang=pipeline.lang,questions=pipeline.questions,examples=[dict(id=d['id'],title=d['title']) for d in pipeline.retriever.docs if d['split']=='validation'][:12]))
            if self.path=='/report':return self.send(report_data.build(Path(__file__).parent,checkpoint))
            if self.path=='/health':return self.send(dict(app='tnka-l4-v2',topic=pipeline.topic))
            return self.send({'error':'Not found'},404)
        def do_POST(self):
            if self.path!='/ask':return self.send({'error':'Not found'},404)
            if self.headers.get('Origin') not in (None,base):return self.send({'error':'Local UI only'},403)
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<5000:raise ValueError('Invalid input length')
                data=json.loads(self.rfile.read(size));result=pipeline.ask(data['keywords'],data['field'],data.get('mode','retrieval'),data.get('selected_id'));self.send(result)
            except (ValueError,KeyError,StopIteration) as e:self.send({'error':str(e)},400)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler);print(base,flush=True);return server
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).parent);p.add_argument('--topic',choices=['software','university','research'],default='software');p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--encoder',type=Path);p.add_argument('--port',type=int,default=8904);a=p.parse_args()
    serve(Pipeline(a.root,a.topic,a.checkpoint,a.encoder),a.port,a.checkpoint).serve_forever()
