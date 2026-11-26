import json,csv
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
O='outputs/research_LT/'
NOTE={'no_source':'GPT without any card: guesses train priors (month 6) or unknown',
'correct_card':'oracle card by document_id; errors are pure GPT errors (multi-digit references, month)',
'retrieved_card':'LSA top-1 card; all retrieval misses are exact duplicate titles (score 1.0 ties)',
'training_majority':'most frequent TRAIN answer per field (year 2023, references 0, month unknown ...)',
'field_reading_rule':'regex field : value ; on the correct card, else unknown; no learning'}
cols=['split','condition','n_questions','n_answerable','n_unanswerable','retrieval_correct','retrieval_total','em_correct','em_total','abstention_correct','abstention_total','unsupported_count','unsupported_total','incomplete_count','incomplete_total','notes']
out=[]
for split in ['validation','test']:
    r=json.load(open(O+f'exp_{split}.json',encoding='utf-8'))['results']
    for c in NOTE:
        row={k:r[c].get(k,'') for k in cols[2:-1]};row.update(split=split,condition=c,notes=NOTE[c])
        if c!='retrieved_card':row['retrieval_correct']=row['retrieval_total']='netaikoma'
        out.append(row)
with open('EXPERIMENT_RESULTS.csv','w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,cols);w.writeheader();w.writerows(out)
rep=[];runs={'baseline_seed42':'outputs/research_LT/exp_validation.json','baseline_seed7 (noise check)':'outputs/baseline_s7/exp_validation.json','repair_field_shuffle_seed42':'outputs/repair_aug_s42/exp_validation.json'}
for name,p in runs.items():
    r=json.load(open(p,encoding='utf-8'))['results']
    for c in ['correct_card','retrieved_card','no_source']:
        x=r[c];rep.append(dict(run=name,condition=c,em=f"{x['em_correct']}/{x['em_total']}",abstention=f"{x['abstention_correct']}/{x['abstention_total']}",unsupported=f"{x['unsupported_count']}/{x['unsupported_total']}",incomplete=f"{x['incomplete_count']}/{x['incomplete_total']}",**{f'em_{k}':v['answerable'] for k,v in x['per_field'].items()}))
with open(O+'repair_before_after.csv','w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,list(rep[0]));w.writeheader();w.writerows(rep)
fig,ax=plt.subplots(1,2,figsize=(11,4))
for name,p,col in [('bazinis (seed 42)','outputs/research_LT/history.json','#2a5bd7'),('pataisymas: laukų maišymas','outputs/repair_aug_s42/history.json','#c2410c')]:
    h=json.load(open(p,encoding='utf-8'));s=[x['step'] for x in h]
    ax[0].plot(s,[x['train_loss'] for x in h],'--',color=col,lw=1,label=name+' · mokymo (paskutinis batch)');ax[0].plot(s,[x['validation_loss'] for x in h],'-',color=col,marker='o',ms=3,label=name+' · validavimo')
    ax[1].plot(s,[x['validation']['answerable_em'] for x in h],'-',color=col,marker='o',ms=3,label=name)
ax[0].set_xlabel('Mokymo žingsnis');ax[0].set_ylabel('Kryžminė entropija (atsakymo + EOS tokenai)');ax[0].legend(fontsize=8);ax[0].set_title('Nuostolis')
ax[1].set_xlabel('Mokymo žingsnis');ax[1].set_ylabel('Validavimo answerable EM');ax[1].legend(fontsize=8);ax[1].set_title('Tikslus sutapimas (validavimas)')
plt.tight_layout();plt.savefig(O+'loss_curve.png',dpi=130);print('ok')
