import argparse, json, re, time
from pathlib import Path
from rouge_score.rouge_scorer import RougeScorer
DATA = 'DATA/oz/oz12.jsonl'
MODEL = 'DATA/t5-small'
ENC = 'DATA/all-MiniLM-L6-v2'
NEUTRAL_INTRO = 'It was a quiet day, and the sky above the road was pale and clear.'
scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def rows(split=None):
    r = [json.loads(l) for l in Path(DATA).read_text().splitlines() if l.strip()]
    return [x for x in r if split is None or x['split'] == split]

def rouge(pred, ref):
    return {k: round(v.fmeasure, 4) for k, v in scorer.score(ref, pred).items()}

def macro(preds, refs):
    per = [rouge(p, r) for p, r in zip(preds, refs)]
    return {k: round(sum((d[k] for d in per)) / len(per), 4) for k in per[0]}

def cut(text):
    return not re.search('[.!?][”"\\\']?\\s*$', text.strip())

def cmd_scores(a):
    from embedding_tools import SentenceEncoder
    from student_methods import score_table, extractive
    enc = SentenceEncoder(ENC)
    r = next((x for x in rows() if x['id'] == a.id))
    table = score_table(r['source'], enc.encode)
    top = sorted(table, key=lambda t: (-t[1], t[0]))[:2]
    print(f'| # | balas | atrinktas | sakinys |\n|---|---|---|---|')
    for i, s, p in table:
        print(f"| {i} | {s:.4f} | {('TAIP' if (i, s, p) in top else '')} | {p} |")
    print('\nB išvestis:', extractive(r['source'], enc.encode))
    print('Encoderio nukirpti sakiniai (>256 tok.):', enc.truncated)

def cmd_length(a):
    refs = {x['id']: x['reference'] for x in rows()}
    for f in a.files:
        d = json.loads(Path(f).read_text())
        print(f"\n## {f}  (T5 laikas: {d['results']['generative']['runtime']['seconds']:.2f} s)")
        print('| id | žodžiai | nutrūko | R1 | R2 | RL | T5 išvestis |\n|---|---|---|---|---|---|---|')
        outs = d['outputs']['generative']
        for i, o in zip(d['ids'], outs):
            m = rouge(o, refs[i])
            print(f"| {i} | {len(o.split())} | {('taip' if cut(o) else 'ne')} | {m['rouge1']} | {m['rouge2']} | {m['rougeL']} | {o} |")
        print(f'Nutrūko: {sum(map(cut, outs))}/{len(outs)}; vid. žodžių: {sum((len(o.split()) for o in outs)) / len(outs):.1f}')
        print('Macro:', {k: v['metrics'] for k, v in d['results'].items()})

def cmd_fix(a):
    from embedding_tools import SentenceEncoder
    from student_methods import extractive, extractive_mmr
    enc = SentenceEncoder(ENC)
    val = rows('validation')
    assert len(val) == 8
    res = {}
    for name, fn in [('baseline', extractive), ('mmr', extractive_mmr)]:
        t = time.perf_counter()
        preds = [fn(r['source'], enc.encode) for r in val]
        res[name] = dict(metrics=macro(preds, [r['reference'] for r in val]), mean_words=sum((len(p.split()) for p in preds)) / len(preds), seconds=time.perf_counter() - t, outputs=dict(zip([r['id'] for r in val], preds)))
    print('| metodas | R1 | R2 | RL | vid. žodžių |\n|---|---|---|---|---|')
    for n, v in res.items():
        m = v['metrics']
        print(f"| {n} | {m['rouge1']} | {m['rouge2']} | {m['rougeL']} | {v['mean_words']:.1f} |")
    for r in val:
        b, m = (res['baseline']['outputs'][r['id']], res['mmr']['outputs'][r['id']])
        print(f"\n### {r['id']} {('(NEPASIKEITĖ)' if b == m else '')}\nBAZINIS: {b}\nMMR:     {m}")
    Path('fix_validation.json').write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print('\nIšsaugota: fix_validation.json')

def cmd_intro(a):
    from embedding_tools import SentenceEncoder
    from student_methods import extractive
    from evaluate import lead, generate
    enc = SentenceEncoder(ENC)
    r = next((x for x in rows('validation') if x['id'] == a.id))
    variants = {'original': r['source'], 'with_intro': NEUTRAL_INTRO + ' ' + r['source']}
    out = {}
    gen, info = generate(list(variants.values()), MODEL, a.max_new_tokens)
    for (k, src), g in zip(variants.items(), gen):
        out[k] = {'lead2': lead(src), 'embedding': extractive(src, enc.encode), 'generative': g}
    for k in out:
        print(f'\n## {k}')
        for m, t in out[k].items():
            print(f"- {m}: {t}\n  ROUGE: {rouge(t, r['reference'])}")
    Path(f'intro_{a.id}.json').write_text(json.dumps(dict(id=a.id, intro=NEUTRAL_INTRO, gen=info, outputs=out), ensure_ascii=False, indent=2))
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest='cmd', required=True)
    x = s.add_parser('scores')
    x.add_argument('--id', required=True)
    x = s.add_parser('length')
    x.add_argument('files', nargs='+')
    s.add_parser('fix')
    x = s.add_parser('intro')
    x.add_argument('--id', default='OZ-02')
    x.add_argument('--max-new-tokens', type=int, default=64)
    a = p.parse_args()
    globals()['cmd_' + a.cmd](a)
