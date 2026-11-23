"""Provided frozen sentence-encoder infrastructure; no ranking solution here."""
from pathlib import Path
import os
class SentenceEncoder:
    def __init__(self,path):
        os.environ.setdefault('HF_HOME',str(Path(path).parent/'hf-cache'))
        import torch
        from transformers import AutoModel,AutoTokenizer
        torch.set_num_threads(1)
        self.torch=torch
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,clean_up_tokenization_spaces=False)
        self.model=AutoModel.from_pretrained(path,local_files_only=True,use_safetensors=True).eval()
        self.truncated=0
    def encode(self,texts):
        torch=self.torch;results=[]
        for start in range(0,len(texts),32):
            part=texts[start:start+32]
            lengths=self.tokenizer(part,truncation=False,verbose=False)['input_ids']
            self.truncated+=sum(len(x)>256 for x in lengths)
            x=self.tokenizer(part,padding=True,truncation=True,max_length=256,return_tensors='pt')
            with torch.no_grad():hidden=self.model(**x).last_hidden_state
            mask=x['attention_mask'].unsqueeze(-1).float()
            pooled=(hidden*mask).sum(1)/mask.sum(1).clamp(min=1e-9)
            results.append(torch.nn.functional.normalize(pooled,p=2,dim=1).cpu().numpy())
        import numpy as np
        return np.concatenate(results) if results else np.empty((0,384))
