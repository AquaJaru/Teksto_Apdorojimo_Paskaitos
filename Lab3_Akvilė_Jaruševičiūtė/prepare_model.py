import argparse,json
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--revision',required=True);a=p.parse_args()
assert len(a.revision)==40,'Use the reviewed exact commit SHA.'
path=snapshot_download('google-t5/t5-small',revision=a.revision,local_dir=a.out,allow_patterns=['config.json','generation_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','spiece.model','model.safetensors'])
(a.out/'TNKA_MODEL.json').write_text(json.dumps(dict(model='google-t5/t5-small',revision=a.revision,license='Apache-2.0'),indent=2));print(path)
