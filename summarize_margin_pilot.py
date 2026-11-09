"""Audit the fixed pilot endpoint; plot without importing the training runtime."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys


ROOT=Path(__file__).resolve().parent
NAME='margin_pilot_20261109'


def rows(path):
    with path.open(encoding='utf-8',newline='') as handle:return list(csv.DictReader(handle))


def sha(path):
    result=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):result.update(block)
    return result.hexdigest()


def analyze(args):
    import torch
    torch.set_num_threads(2)
    root=ROOT/'results'/NAME
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    run=json.loads((root/'run_summary.json').read_text(encoding='utf-8'))
    assert run['status']=='finished' and len(run['arms'])==4
    assert all(a['exit_code']==0 and not a['telemetry_errors'] for a in run['arms'])
    valid=rows(root/'validation.csv');ids=[r['video_id'] for r in valid];idset=set(ids)
    train_ids=set((root/'train_video_ids.txt').read_text().splitlines())
    assert len(train_ids)==8000 and len(idset)==1000 and not(train_ids&idset)
    assert sha(root/'train_video_ids.txt')==protocol['split']['train_video_ids_sha256']
    assert sha(root/'validation.csv')==protocol['split']['validation_csv_sha256']
    reports={};curve=[];checkpoint_checks=[]
    for setting in protocol['fixed_arm_order']:
        name=setting['name'];folder=root/'arms'/name
        report=json.loads((folder/'arm_report.json').read_text(encoding='utf-8'))
        assert report['status']=='finished' and report['steps']==3000 and report['all_model_tensors_finite']
        assert report['seed']==setting['seed'] and report['margin']==setting['margin']
        assert report['protocol_sha256']==run['protocol_sha256']==sha(root/'protocol.json')
        assert [v['step'] for v in report['validations']]==[0,1000,2000,3000]
        assert len(report['training_chunks'])==3 and all(c['steps']==1000 for c in report['training_chunks'])
        batches=rows(folder/'batch_order.csv');assert len(batches)==3000
        batch_hash=hashlib.sha256();seen=set();input_hashes=[]
        for step,batch in enumerate(batches,1):
            assert int(batch['step'])==step
            pairs=json.loads(batch['pair_ids']);videos=json.loads(batch['video_ids'])
            assert len(pairs)==len(videos)==4 and all(0<=p<160000 for p in pairs)
            assert all(v in train_ids and v not in idset for v in videos)
            batch_hash.update(json.dumps([step,pairs,videos],separators=(',',':')).encode());seen.update(videos)
            if step<=3:input_hashes.append(batch['first_three_input_sha256'])
        assert batch_hash.hexdigest()==report['batch_order_sha256']
        assert input_hashes==report['first_three_input_sha256']
        training=[]
        for path in folder.rglob('training_steps.csv'):training.extend(rows(path))
        training.sort(key=lambda r:int(r['step']))
        assert [int(r['step']) for r in training]==list(range(1,3001))
        assert all(math.isfinite(float(r['loss'])) and int(r['batch_size'])==4 for r in training)
        for validation in report['validations']:
            data=rows(folder/f'validation_queries_step{validation["step"]:04d}.csv')
            assert [r['video_id'] for r in data]==ids
            ranks=[int(r['rank']) for r in data];metrics=validation['metrics']
            for key,k in [('R1',1),('R5',5),('R10',10),('R50',50),('R100',100)]:assert abs(sum(x<=k for x in ranks)/10-metrics[key])<1e-9
            assert abs(statistics.mean(ranks)-metrics['MeanR'])<1e-9
            assert statistics.median(ranks)==metrics['MedR']
            assert all((int(r['rank'])==1)==(r['video_id']==r['top1_video_id']) for r in data)
            curve.append({'arm':name,'seed':report['seed'],'margin':report['margin'],'step':validation['step'],**metrics})
        checkpoint=ROOT/'outputs'/NAME/name/'checkpoint-last.pth'
        assert sha(checkpoint)==report['final_checkpoint_sha256']
        state=torch.load(str(checkpoint),map_location='cpu')
        assert state['global_step']==3000 and state['data_state']['next_batch']==3000
        assert state['data_state']['dataset_size']==160000 and state['partial_epoch'] is True
        assert state['lr_scheduler']['last_epoch']==3000
        assert {int(v['step']) for v in state['optimizer']['state'].values()}=={3000}
        assert all(p in state['optimizer']['state'] for p in state['optimizer']['param_groups'][1]['params'])
        assert all(g['lr']==0 for g in state['optimizer']['param_groups'])
        assert all(bool(torch.isfinite(v).all()) for v in state['state_dict'].values())
        pool_hash=hashlib.sha256()
        for key,value in state['state_dict'].items():
            if key.startswith('pool_frames.'):
                pool_hash.update(key[len('pool_frames.'):].encode());pool_hash.update(value.numpy().tobytes())
        assert pool_hash.hexdigest()!=report['initial_pooling_sha256'],'Pooling weights must actually train after validation'
        checkpoint_checks.append({'arm':name,'global_step':3000,'pooling_optimizer_parameters_updated':len(state['optimizer']['param_groups'][1]['params']),
                                  'all_model_tensors_finite':True,'seen_training_videos':len(seen),'checkpoint_sha256':report['final_checkpoint_sha256']})
        del state
        reports[name]=report
    comparisons=[]
    for seed in [24,25]:
        base=reports[f'seed{seed}_baseline'];candidate=reports[f'seed{seed}_margin002']
        for key in ['initial_pooling_sha256','initial_logit_scale','base_clip_sha256','batch_order_sha256','first_three_input_sha256']:
            assert base[key]==candidate[key],(seed,key)
        assert base['validations'][0]['metrics']==candidate['validations'][0]['metrics']
        bm=base['validations'][-1]['metrics'];cm=candidate['validations'][-1]['metrics']
        a=rows(root/'arms'/f'seed{seed}_baseline'/'validation_queries_step3000.csv')
        b=rows(root/'arms'/f'seed{seed}_margin002'/'validation_queries_step3000.csv')
        groups={'both_hit':0,'baseline_only_hit':0,'margin_only_hit':0,'both_miss':0};paired=[]
        for aa,bb in zip(a,b):
            ar=int(aa['rank']);br=int(bb['rank'])
            name='both_hit' if ar==br==1 else 'baseline_only_hit' if ar==1 else 'margin_only_hit' if br==1 else 'both_miss'
            groups[name]+=1
            paired.append({'video_id':aa['video_id'],'caption':aa['caption'],'baseline_rank':ar,'margin_rank':br,
                           'baseline_top1':aa['top1_video_id'],'margin_top1':bb['top1_video_id']})
        delta=round(cm['R1']-bm['R1'],6)
        assert abs(delta-(groups['margin_only_hit']-groups['baseline_only_hit'])/10)<1e-9
        comparisons.append({'seed':seed,'baseline_final':bm,'margin_final':cm,'R1_difference_pp':delta,'paired_R1_groups':groups,
                            'matched_initialization_data_and_first_three_inputs':True})
        with (root/f'paired_validation_seed{seed}.csv').open('w',encoding='utf-8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    baseline_values=[x['baseline_final']['R1'] for x in comparisons];margin_values=[x['margin_final']['R1'] for x in comparisons]
    differences=[x['R1_difference_pp'] for x in comparisons]
    mean_difference=round(statistics.mean(differences),6)
    passed=all(d>0 for d in differences) and mean_difference>=1.0
    result={'completed_at_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(root/'protocol.json'),
            'validation_protocol':'Internal holdout from the original training 9K: train 8000 videos, validate 1000, one query per video. Not official 1K-A.',
            'primary_endpoint':'R@1 at fixed final 3000 updates; two paired seeds; 12000 pair exposures per arm.',
            'paired_results':comparisons,'baseline_R1_mean':round(statistics.mean(baseline_values),6),'margin_R1_mean':round(statistics.mean(margin_values),6),
            'baseline_R1_sample_sd':statistics.stdev(baseline_values),'margin_R1_sample_sd':statistics.stdev(margin_values),
            'mean_paired_R1_difference_pp':mean_difference,'fixed_screen_passed':passed,
            'screening_rule':protocol['screening_rule'],'outcome':'passes_fixed_pilot_screen' if passed else 'does_not_pass_fixed_pilot_screen',
            'total_process_seconds':sum(a['process_seconds'] for a in run['arms']),
            'gpu_peak_used_mib':max(a['gpu_peak_used_mib'] for a in run['arms']),
            'max_temperature_c':max(a['max_temperature_c'] for a in run['arms']),
            'memory_errors_observed':0,'checkpoint_checks':checkpoint_checks,
            'limitations':protocol['limitations']+['The screening rule is not a statistical significance test; two seeds cannot establish a reliable generalization gain.']}
    (root/'analysis_summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    with (root/'validation_curve.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(curve[0]));writer.writeheader();writer.writerows(curve)
    print(json.dumps(result,indent=2))


def plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    root=ROOT/'results'/NAME;result=json.loads((root/'analysis_summary.json').read_text())
    curve=rows(root/'validation_curve.csv');directory=root/'figures';directory.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for label,margin,color in [('Original loss',0.,'#1f77b4'),('Margin 0.02',.02,'#e07a21')]:
        means=[];deviations=[]
        for step in [0,1000,2000,3000]:
            values=[float(r['R1']) for r in curve if float(r['margin'])==margin and int(r['step'])==step]
            means.append(statistics.mean(values));deviations.append(statistics.stdev(values))
        ax.errorbar([0,1000,2000,3000],means,yerr=deviations,marker='o',capsize=4,label=label,color=color)
    ax.set(xlabel='Optimizer updates (batch 4, full-model training)',ylabel='Internal validation Recall@1 (%)',
           title='8K train / 1K internal holdout: mean and sample SD over two seeds')
    ax.set_xticks([0,1000,2000,3000]);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(directory/'validation_r1_curve.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    x=np.arange(2);width=.32
    for offset,key,label,color in [(-width/2,'baseline_final','Original loss','#1f77b4'),(width/2,'margin_final','Margin 0.02','#e07a21')]:
        values=[r[key]['R1'] for r in result['paired_results']]
        bars=axes[0].bar(x+offset,values,width=width,label=label,color=color)
        axes[0].bar_label(bars,fmt='%.1f',padding=3)
    axes[0].set(xticks=x,xticklabels=['Seed 24','Seed 25'],ylabel='Internal validation R@1 (%)',title='Fixed final endpoint: 3000 updates')
    axes[0].set_ylim(0,max(r[k]['R1'] for r in result['paired_results'] for k in ['baseline_final','margin_final'])+6)
    axes[0].legend(fontsize=9,loc='lower left')
    values=[r['R1_difference_pp'] for r in result['paired_results']]+[result['mean_paired_R1_difference_pp']]
    axes[1].bar(['Seed 24','Seed 25','Mean'],values,color=['#2f8c61' if v>0 else '#bb5544' for v in values])
    axes[1].axhline(0,color='black',linewidth=.8);axes[1].axhline(1,color='#666666',linestyle='--',label='Pre-fixed mean screen: +1.0 pp')
    axes[1].set(ylabel='Margin minus baseline (percentage points)',title='Paired change at equal training budget');axes[1].legend(fontsize=8)
    axes[1].set_ylim(min(min(values),0)-.25,max(1.35,max(values)+.25))
    fig.tight_layout();fig.savefig(directory/'paired_final_r1.png',dpi=180);plt.close(fig)
    (root/'plotting_environment.json').write_text(json.dumps({'python':sys.version,'matplotlib':matplotlib.__version__,'numpy':np.__version__},indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['analyze','plot'])
    parser.add_argument('--experiment-name',default=NAME);args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.experiment_name):parser.error('Invalid experiment directory name')
    NAME=args.experiment_name
    (analyze if args.mode=='analyze' else plot)(args)
