"""A bounded, paired additive-margin pilot on an internal MSR-VTT split.

Existing full-9K checkpoints are not loaded: they have already seen the proposed
holdout videos. Both conditions start from the same base CLIP + identity pooling.
"""

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.data_factory import DataFactory
from datasets.model_transforms import init_transform_dict
from datasets.video_capture import VideoCapture
from model.clip_transformer import CLIPTransformer
from modules.loss import CLIPLoss, MarginCLIPLoss
from modules.metrics import t2v_metrics
from modules.optimization import AdamW, get_cosine_schedule_with_warmup
from modules.training_state import capture_random_state, restore_random_state, training_directory_lock
from trainer.trainer import Trainer


ROOT = Path(__file__).resolve().parent
NAME = 'margin_pilot_20261109'
ARMS = [('seed24_baseline', 24, 0.0), ('seed24_margin002', 24, 0.02),
        ('seed25_margin002', 25, 0.02), ('seed25_baseline', 25, 0.0)]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def make_records(video_ids, annotations, validation_size=1000, split_seed=24):
    assert len(video_ids) == len(set(video_ids)) and 0 < validation_size < len(video_ids)
    captions = {video: [] for video in video_ids}
    for row in annotations:
        if row['video_id'] in captions:
            captions[row['video_id']].append(row['caption'])
    assert all(captions.values())
    key = lambda video: hashlib.sha256(f'split|{split_seed}|{video}'.encode()).hexdigest()
    holdout = set(sorted(video_ids, key=key)[:validation_size])
    train = [];valid = []
    for video in video_ids:
        if video in holdout:
            choice = int(hashlib.sha256(f'caption|{split_seed}|{video}'.encode()).hexdigest(), 16) % len(captions[video])
            valid.append({'video_id': video, 'sentence': captions[video][choice], 'caption_index': choice})
        else:
            for index, caption in enumerate(captions[video]):
                train.append({'video_id': video, 'sentence': caption, 'caption_index': index, 'pair_id': len(train)})
    assert not ({r['video_id'] for r in train} & {r['video_id'] for r in valid})
    return train, valid


def read_source(source):
    with (source/'data/MSRVTT/MSRVTT_train.9k.csv').open(encoding='utf-8', newline='') as handle:
        video_ids = [r['video_id'] for r in csv.DictReader(handle)]
    annotations = json.loads((source/'data/MSRVTT/MSRVTT_data.json').read_text(encoding='utf-8'))['sentences']
    return make_records(video_ids, annotations)


def prepare(args):
    source = args.source_root.resolve();output = ROOT/'results'/NAME
    if (output/'protocol.json').exists():
        raise FileExistsError('The fixed pilot protocol already exists')
    output.mkdir(parents=True, exist_ok=True)
    train, valid = read_source(source)
    train_ids = list(dict.fromkeys(r['video_id'] for r in train))
    valid_ids = {r['video_id'] for r in valid}
    with (source/'data/MSRVTT/MSRVTT_JSFUSION_test.csv').open(encoding='utf-8', newline='') as handle:
        official_ids = {r['video_id'] for r in csv.DictReader(handle)}
    assert len(train_ids)==8000 and len(valid_ids)==1000 and len(train)==160000
    assert not ((set(train_ids)|valid_ids) & official_ids)
    (output/'train_video_ids.txt').write_text('\n'.join(train_ids)+'\n', encoding='utf-8')
    with (output/'validation.csv').open('w', encoding='utf-8', newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['video_id','sentence','caption_index']);writer.writeheader();writer.writerows(valid)
    weights=source/'openai/clip-vit-base-patch32/pytorch_model.bin'
    protocol = {
        'created_before_training_at_utc': datetime.now(timezone.utc).isoformat(),
        'base_repository_commit': '144fb325c0c93182c5e74450ba18c94e048ca29b',
        'hypothesis': 'A fixed additive positive cosine margin may help separate confusing retrieval candidates. This is a pilot, not an expected improvement guarantee.',
        'margin_formula': 'L = original symmetric CLIPLoss(S - m*I, logit_scale), m=0 baseline or m=0.02 candidate; inference uses unchanged cosine scores.',
        'initialization': 'Base openai/clip-vit-base-patch32 + original fresh X-Pool pooling; no prior MSR-VTT full-9K checkpoint is loaded.',
        'base_clip_sha256': digest(weights),
        'split': {'parent_videos':9000,'training_videos':8000,'training_caption_pairs':len(train),'validation_videos_and_queries':1000,
                  'seed':24,'video_selection':'Lowest SHA256(split|24|video_id) values, 1000 videos',
                  'validation_caption_selection':'SHA256(caption|24|video_id) modulo that video caption count; one caption per video',
                  'train_video_ids_sha256':digest(output/'train_video_ids.txt'),'validation_csv_sha256':digest(output/'validation.csv'),
                  'annotation_sha256':digest(source/'data/MSRVTT/MSRVTT_data.json'),'official_1ka_excluded':True},
        'training': {'seeds':[24,25],'steps_per_arm':3000,'batch_size':4,'precision':'float32','num_frames':12,'input_resolution':224,
                     'clip_lr':1e-6,'pooling_lr':3e-5,'weight_decay':0.2,'warmup_steps':300,'scheduler':'cosine to zero at 3000 updates',
                     'transformer_dropout':0.3,'workers':2,'all_clip_and_pooling_parameters_trainable':True,'amp':False,'gradient_accumulation':False},
        'evaluation': {'steps':[0,1000,2000,3000],'protocol':'Internal holdout 1000 videos / one query each / text-to-video; not official 1K-A',
                       'ranking':'Original t2v_metrics, unmodified cosine scores','validation_loss':'Original margin-free CLIPLoss with evaluation batch size 4'},
        'primary_endpoint':'Paired R@1 difference at the fixed final 3000 updates for each seed. Best-validation checkpoint is secondary only.',
        'screening_rule':'Consider further study only if both seed differences are positive and their mean is at least +1.0 percentage point, with no memory failures. This is a practical screening threshold, not statistical significance.',
        'fixed_arm_order':[{'name':name,'seed':seed,'margin':margin} for name,seed,margin in ARMS],
        'matching':'Same initialization, pair order, augmentation seeds, optimizer and update/evaluation budgets within each seed. Record pair-order hashes and first three input hashes.',
        'limitations':['Short full-model pilot: 12000 pair exposures per arm, 7.5% of one 160000-pair training epoch.',
                       'Two seeds and one internal split do not establish generalization or publication-level significance.',
                       'The hypothesis follows earlier official-test error inspection; this internal validation is exploratory. Official 1K-A is not evaluated or used for choosing settings in this pilot.'],
    }
    write_json(output/'protocol.json',protocol)
    print(json.dumps({'protocol':str(output/'protocol.json'),'sha256':digest(output/'protocol.json'),'train_pairs':len(train),'validation_queries':len(valid)},ensure_ascii=True),flush=True)


class PilotDataset(Dataset):
    def __init__(self, records, source, transform):
        self.records=records;self.source=source;self.transform=transform
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        row=self.records[index]
        frames,_=VideoCapture.load_frames_from_video(str(self.source/'data/MSRVTT/vids'/(row['video_id']+'.mp4')),12,'uniform')
        return {'video_id':row['video_id'],'text':row['sentence'],'video':self.transform(frames),'pair_id':row.get('pair_id',index)}


class BatchAudit:
    def __init__(self,path):
        self.handle=path.open('x',encoding='utf-8',newline='');self.writer=csv.writer(self.handle)
        self.writer.writerow(['step','pair_ids','video_ids','first_three_input_sha256'])
        self.hash=hashlib.sha256();self.steps=0;self.first_inputs=[];self.duplicate_video_batches=0
    def hook(self,model,inputs):
        if not model.training:return
        data=inputs[0];self.steps+=1
        pairs=data['pair_id'].tolist();videos=list(data['video_id'])
        text=json.dumps([self.steps,pairs,videos],separators=(',',':'))
        self.hash.update(text.encode());pixel=''
        if self.steps<=3:
            pixel=hashlib.sha256(data['video'].detach().cpu().numpy().tobytes()).hexdigest();self.first_inputs.append(pixel)
        self.duplicate_video_batches+=len(set(videos))<len(videos)
        self.writer.writerow([self.steps,json.dumps(pairs),json.dumps(videos),pixel]);self.handle.flush()
    def close(self):self.handle.close()


def arm(args):
    source=args.source_root.resolve();public=ROOT/'results'/NAME;protocol=json.loads((public/'protocol.json').read_text(encoding='utf-8'))
    settings=next(a for a in protocol['fixed_arm_order'] if a['name']==args.arm)
    output=ROOT/'outputs'/NAME/args.arm
    output.mkdir(parents=True,exist_ok=True)
    if (output/'arm_report.json').exists() or (output/'checkpoint-last.pth').exists():raise FileExistsError(output)
    seed=settings['seed'];margin=settings['margin'];os.chdir(source)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available(),'This pilot is intended for the measured CUDA environment'
    train,valid=read_source(source)
    with (public/'validation.csv').open(encoding='utf-8',newline='') as handle:saved_valid=list(csv.DictReader(handle))
    assert [(r['video_id'],r['sentence']) for r in valid]==[(r['video_id'],r['sentence']) for r in saved_valid]
    assert digest(source/'openai/clip-vit-base-patch32/pytorch_model.bin')==protocol['base_clip_sha256']
    config=SimpleNamespace(dataset_name='MSRVTT_INTERNAL_PILOT',msrvtt_train_file='internal8k',videos_dir='data/MSRVTT/vids',num_frames=12,
        video_sample_type='uniform',input_res=224,exp_name=args.arm,model_path=str(output),output_dir=str(output.parent),
        save_every=1,log_step=100,evals_per_epoch=0,load_epoch=None,eval_window_size=5,metric='t2v',huggingface=True,
        arch='clip_transformer',clip_arch='ViT-B/32',embed_dim=512,loss=f'clip_margin_{margin}',clip_lr=1e-6,noclip_lr=3e-5,
        batch_size=4,num_epochs=1,weight_decay=0.2,warmup_proportion=0.1,max_train_seconds=0,max_train_steps=1000,profile_training=True,
        resume=None,pooling_type='transformer',k=-1,attention_temperature=0.01,num_mha_heads=1,transformer_dropout=0.3,num_workers=2,
        seed=seed,no_tensorboard=True,tb_log_dir=str(output/'unused_tensorboard'),protocol_sha256=digest(public/'protocol.json'))
    transforms=init_transform_dict(224)
    training_loader=DataFactory.training_loader(PilotDataset(train,source,transforms['clip_train']),config)
    validation_loader=DataLoader(PilotDataset(valid,source,transforms['clip_test']),batch_size=4,shuffle=False,num_workers=2,
                                 generator=torch.Generator().manual_seed(424242))
    from transformers import CLIPTokenizer
    tokenizer=CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32',TOKENIZERS_PARALLELISM=False)
    model=CLIPTransformer(config)
    pooling_hash=hashlib.sha256()
    for name,tensor in model.pool_frames.state_dict().items():pooling_hash.update(name.encode());pooling_hash.update(tensor.numpy().tobytes())
    initial_logit_scale=float(model.clip.logit_scale)
    model=model.cuda()
    grouped=[{'params':[p for n,p in model.named_parameters() if n.startswith('clip.')],'lr':1e-6},
             {'params':[p for n,p in model.named_parameters() if not n.startswith('clip.')],'lr':3e-5}]
    optimizer=AdamW(grouped,weight_decay=0.2)
    scheduler=get_cosine_schedule_with_warmup(optimizer,300,3000)
    loss=CLIPLoss() if margin==0 else MarginCLIPLoss(margin)
    current_step=[0]
    def metrics(sims):
        result=t2v_metrics(sims)
        scores=sims[:,0,:];order=torch.argsort(scores,dim=1,descending=True);ranks=torch.diag(torch.argsort(order,dim=1))+1
        with (output/f'validation_queries_step{current_step[0]:04d}.csv').open('w',encoding='utf-8',newline='') as handle:
            writer=csv.writer(handle);writer.writerow(['video_id','caption','rank','top1_video_id'])
            for i,row in enumerate(valid):writer.writerow([row['video_id'],row['sentence'],int(ranks[i]),valid[int(order[i,0])]['video_id']])
        assert abs(result['R1']-float((ranks==1).sum())/10)<1e-9
        return result
    trainer=Trainer(model,loss,metrics,optimizer,config,training_loader,validation_loader,tokenizer,scheduler,None)
    audit=BatchAudit(output/'batch_order.csv');hook=model.register_forward_pre_hook(audit.hook)
    report={'status':'running','arm':args.arm,'seed':seed,'margin':margin,'protocol_sha256':config.protocol_sha256,
            'started_at_utc':datetime.now(timezone.utc).isoformat(),'initial_pooling_sha256':pooling_hash.hexdigest(),
            'initial_logit_scale':initial_logit_scale,'base_clip_sha256':protocol['base_clip_sha256'],
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'validations':[],'training_chunks':[]}
    write_json(output/'arm_report.json',report)
    def validate():
        state=capture_random_state();original_loss=trainer.loss;trainer.loss=CLIPLoss().cuda();current_step[0]=trainer.global_step
        start=time.perf_counter()
        try:result=trainer._valid_epoch_step(0,trainer.global_step,3000)
        finally:trainer.loss=original_loss;restore_random_state(state);model.train()
        item={'step':trainer.global_step,'metrics':{k:float(result[k]) for k in ['R1','R5','R10','R50','R100','MedR','MeanR']},
              'canonical_batch4_clip_loss':float(result['loss_val']),'seconds':round(time.perf_counter()-start,3)}
        report['validations'].append(item);write_json(output/'arm_report.json',report)
        print('VALIDATION',json.dumps(item),flush=True)
    try:
        with training_directory_lock(str(output)):
            validate()
            for target in [1000,2000,3000]:
                trainer.train();assert trainer.global_step==target
                profile=json.loads((output/'training_profile.json').read_text(encoding='utf-8'))
                report['training_chunks'].append({'target_step':target,'elapsed_seconds':profile['elapsed_seconds'],'steps':profile['completed_steps']})
                validate()
                best=max(report['validations'],key=lambda x:(x['metrics']['R1'],-x['step']))
                if best['step']==target:
                    shutil.copyfile(output/'checkpoint-last.pth',output/'best_validation.pth')
            assert audit.steps==3000 and training_loader.batch_sampler.next_batch==3000
            assert {int(s['step']) for s in optimizer.state.values()}=={3000}
            assert scheduler.last_epoch==3000 and all(g['lr']==0 for g in optimizer.param_groups)
            assert all(bool(torch.isfinite(v).all()) for v in model.state_dict().values())
        report.update(status='finished',completed_at_utc=datetime.now(timezone.utc).isoformat(),steps=3000,
            batch_order_sha256=audit.hash.hexdigest(),first_three_input_sha256=audit.first_inputs,
            duplicate_video_batches=audit.duplicate_video_batches,scheduler_last_epoch=3000,
            final_checkpoint_sha256=digest(output/'checkpoint-last.pth'),all_model_tensors_finite=True)
        report['best_validation_step']=max(report['validations'],key=lambda x:(x['metrics']['R1'],-x['step']))['step']
        write_json(output/'arm_report.json',report)
    except BaseException as error:
        report.update(status='failed',error=type(error).__name__,completed_steps=trainer.global_step)
        write_json(output/'arm_report.json',report);raise
    finally:hook.remove();audit.close()


def redact(text,source):
    for prefix in [str(source),str(ROOT),str(Path.home())]:
        for variant in [prefix,json.dumps(prefix,ensure_ascii=True)[1:-1],json.dumps(prefix,ensure_ascii=False)[1:-1]]:
            text=text.replace(variant,'<local-root>')
    return text


def export_arm(source,name):
    private=ROOT/'outputs'/NAME/name;public=ROOT/'results'/NAME/'arms'/name
    for path in private.rglob('*'):
        if path.is_file() and path.suffix in {'.json','.csv','.log'}:
            destination=public/path.relative_to(private);destination.parent.mkdir(parents=True,exist_ok=True)
            if path.suffix=='.csv':shutil.copyfile(path,destination)
            elif path.suffix=='.json':
                value=json.loads(path.read_text(encoding='utf-8'))
                # Redact raw JSON text so escaped Windows paths are also covered.
                destination.write_text(redact(json.dumps(value,ensure_ascii=False,indent=2),source)+'\n',encoding='utf-8')
            else:destination.write_text(redact(path.read_text(encoding='utf-8',errors='replace'),source),encoding='utf-8')


def run(args):
    source=args.source_root.resolve();public=ROOT/'results'/NAME
    protocol=json.loads((public/'protocol.json').read_text(encoding='utf-8'))
    if (public/'run_summary.json').exists():raise FileExistsError('Choose a new experiment before rerunning')
    assert all(not (ROOT/'outputs'/NAME/name).exists() for name,_,_ in ARMS),'Existing pilot output directories are preserved'
    report={'status':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':digest(public/'protocol.json'),'arms':[]}
    write_json(public/'run_summary.json',report)
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        for name,_,_ in ARMS:
            directory=ROOT/'outputs'/NAME/name;directory.mkdir(parents=True)
            samples=[];errors=[];stop=threading.Event()
            def monitor():
                while not stop.is_set():
                    try:
                        result=subprocess.run(['nvidia-smi','--id=0','--query-gpu=memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
                        samples.append([datetime.now(timezone.utc).isoformat()]+[float(x.strip()) for x in result.stdout.strip().split(',')])
                    except Exception as error:errors.append(type(error).__name__)
                    stop.wait(5)
            thread=threading.Thread(target=monitor,daemon=True);thread.start()
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'arm','--source-root',str(source),'--arm',name,'--experiment-name',NAME]
            print('STARTING',name,flush=True);start=time.perf_counter()
            try:
                with (directory/'run.log').open('w',encoding='utf-8') as log:
                    process=subprocess.Popen(command,cwd=source,env=dict(os.environ,PYTHONUTF8='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),stdout=log,stderr=subprocess.STDOUT)
                    while process.poll() is None:
                        if time.perf_counter()-start>5400:
                            process.terminate();process.wait(timeout=30);raise TimeoutError('Pilot arm exceeded its 90-minute process budget')
                        time.sleep(5)
                    code=process.returncode
            finally:stop.set();thread.join(timeout=12)
            with (directory/'gpu_samples.csv').open('w',newline='',encoding='utf-8') as handle:
                writer=csv.writer(handle);writer.writerow(['utc_time','device_used_mib','gpu_utilization_percent','temperature_c']);writer.writerows(samples)
            item={'arm':name,'exit_code':code,'process_seconds':round(time.perf_counter()-start,3),
                  'gpu_peak_used_mib':max((r[1] for r in samples),default=None),'max_temperature_c':max((r[3] for r in samples),default=None),'telemetry_errors':errors}
            report['arms'].append(item);export_arm(source,name);write_json(public/'run_summary.json',report)
            print('FINISHED',json.dumps(item),flush=True)
            if code:raise RuntimeError(name+' failed; inspect the preserved log')
        report.update(status='finished',completed_at_utc=datetime.now(timezone.utc).isoformat());write_json(public/'run_summary.json',report)
    except BaseException as error:
        report.update(status='failed',error=type(error).__name__);write_json(public/'run_summary.json',report);raise
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def main():
    global NAME
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','run','arm'])
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--arm',choices=[a[0] for a in ARMS])
    parser.add_argument('--experiment-name',default=NAME,help='New result/output directory name for an independent rerun')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.experiment_name):
        parser.error('--experiment-name must be a single safe directory name')
    NAME=args.experiment_name
    if args.mode=='arm' and not args.arm:parser.error('--arm is required')
    {'prepare':prepare,'run':run,'arm':arm}[args.mode](args)


if __name__=='__main__':main()
