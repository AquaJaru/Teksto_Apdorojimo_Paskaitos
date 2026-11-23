"""Explicit setup only; no extra Python packages required beyond Transformers."""
from pathlib import Path
import argparse,json
from huggingface_hub import snapshot_download
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
snapshot_download('sentence-transformers/all-MiniLM-L6-v2',revision=revision,local_dir=a.out,allow_patterns=['model.safetensors','config.json','tokenizer.json','tokenizer_config.json','vocab.txt','special_tokens_map.json','sentence_bert_config.json'])
(a.out/'TNKA_MODEL.json').write_text(json.dumps(dict(model='sentence-transformers/all-MiniLM-L6-v2',revision=revision,license='Apache-2.0',dimensions=384,max_tokens=256,pooling='attention-mask mean then L2 normalize'),indent=2))
