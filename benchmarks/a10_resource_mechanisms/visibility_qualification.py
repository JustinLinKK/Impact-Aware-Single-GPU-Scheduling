"""Bounded empirical NVML visibility qualification, not kernel PID mapping.

Controller imports only stdlib. CUDA is imported by the source-pinned worker
only after a private empty MPS pipe directory is installed in its environment.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET

SCHEMA='a10_nvml_visibility_qualification_v1'
PHASES=('idle_initial','one_a','two_a_b','one_a_after_b','idle_final')
SOURCE=Path(__file__).resolve()
GPU_LOCK='/tmp/localml-a10-feedback-pilot.lock'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def birth(pid):
    fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
    return {'pid':pid,'ppid':int(fields[1]),'start_ticks':int(fields[19])}
def atomic(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    if path.exists():raise FileExistsError('qualification never replaces evidence')
    with temp.open('x') as stream:json.dump(value,stream,sort_keys=True,allow_nan=False);stream.write('\n')
    temp.rename(path)
def mps_identity(path,*,empty=True):
    path=Path(path);info=path.lstat()
    if (not path.is_absolute() or path.is_symlink() or not stat.S_ISDIR(info.st_mode)
            or info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o700
            or (empty and any(path.iterdir()))):raise ValueError('private empty nonsymlink MPS directory required')
    if any(parent.is_symlink() for parent in path.parents):raise ValueError('MPS directory parent symlink')
    return {'path':str(path),'device':info.st_dev,'inode':info.st_ino,'uid':info.st_uid,'mode':stat.S_IMODE(info.st_mode),'empty':True}
def parse_query(raw,gpu_uuid):
    if len(raw)>1024*1024:raise ValueError('oversized NVML XML evidence')
    root=ET.fromstring(raw);gpus=root.findall('gpu')
    if len(gpus)!=1 or gpus[0].findtext('uuid')!=gpu_uuid:raise ValueError('wrong qualification GPU UUID')
    compute=[];graphics=[]
    for row in gpus[0].findall('./processes/process_info'):
        pid=int(row.findtext('pid','0'));kind=row.findtext('type','')
        if pid<=0:raise ValueError('invalid GPU process evidence')
        (compute if kind=='C' else graphics).append(pid)
    if graphics:raise ValueError('graphics or ambiguous GPU client present')
    if len(set(compute))!=len(compute):raise ValueError('duplicate NVML PID evidence')
    return sorted(compute)
def query(gpu_uuid,deadline):
    started=time.monotonic();remaining=deadline-started
    if remaining<=0:raise TimeoutError('qualification query deadline')
    command=['nvidia-smi','-q','-x','-i',gpu_uuid]
    result=subprocess.run(command,check=True,text=True,capture_output=True,timeout=min(5,remaining))
    return {'command':command,'started_monotonic':started,'finished_monotonic':time.monotonic(),
            'raw_xml':result.stdout,'compute_pids':parse_query(result.stdout,gpu_uuid)}
def challenge_value(nonce):return int(hashlib.sha256(nonce.encode()).hexdigest()[:4],16)
def challenge_result(nonce):return sum(i ^ challenge_value(nonce) for i in range(4096))
def check_worker(proof,*,source_sha,gpu_uuid,mps,parent,owned=None):
    if (proof.get('source_sha256')!=source_sha or proof.get('gpu_uuid')!=gpu_uuid
            or proof.get('parent')!=parent or proof.get('mps_identity')!=mps
            or proof.get('checksum')!=challenge_result(proof.get('nonce',''))
            or proof.get('allocation_bytes')!=8*1024*1024 or not proof.get('synchronized')
            or proof.get('pid')==parent['pid'] or proof.get('ppid')!=parent['pid']
            or type(proof.get('start_ticks'))is not int or proof['start_ticks']<=0
            or not re.fullmatch('[0-9a-f]{32}',str(proof.get('nonce','')))):
        raise ValueError('owned CUDA nonce/source/UUID/ancestry/birth proof mismatch')
    if owned is not None and any(proof[k]!=owned[k] for k in ('pid','ppid','start_ticks','nonce')):
        raise ValueError('changed owned probe identity/nonce')
    return proof


def audit_qualification(proof,*,expected_source_sha256,gpu_uuid):
    if (proof.get('schema')!=SCHEMA or proof.get('complete')is not True
            or proof.get('source_sha256')!=expected_source_sha256 or proof.get('gpu_uuid')!=gpu_uuid
            or proof.get('association')!='empirical_sampled_NVML_visibility_under_completeness'
            or not re.fullmatch('[0-9a-f-]{36}',str(proof.get('boot_id','')))):
        raise ValueError('complete source/GPU-bound visibility qualification required')
    start,end=proof['started_monotonic'],proof['finished_monotonic']
    if not 0<end-start<=120:raise ValueError('qualification absolute120s bound')
    workers=proof['workers'];parent=proof['parent'];mps=proof['mps_identity']
    if (len(workers)!=2 or workers[0]['pid']==workers[1]['pid'] or workers[0]['nonce']==workers[1]['nonce']
            or mps.get('empty')is not True or mps.get('mode')!=0o700):raise ValueError('two distinct owned probes/private MPS proof required')
    for worker in workers:
        if not start<=worker.get('ready_monotonic',-1)<=end:raise ValueError('stale worker CUDA proof')
        check_worker(worker,source_sha=expected_source_sha256,gpu_uuid=gpu_uuid,mps=mps,parent=parent)
        if worker.get('ack')!={'nonce':worker['nonce'],'pid':worker['pid'],'start_ticks':worker['start_ticks']}:
            raise ValueError('missing source-pinned live allocation acknowledgment')
        if worker.get('exit_code')!=0:raise ValueError('qualification probe exit failed')
    if workers[0]['runtime_identity']!=workers[1]['runtime_identity']:raise ValueError('probe runtime changed')
    if not workers[0].get('runtime_identity',{}).get('torch_version'):raise ValueError('actual worker runtime identity required')
    phases=proof['phases']
    if [p['name'] for p in phases]!=list(PHASES):raise ValueError('exact0→1→2→1→0 visibility phases required')
    # Acknowledged live contexts bracket every accepted visibility observation.
    for worker,prior_index,active_index,release_index,next_index in ((workers[0],0,1,3,4),(workers[1],1,2,2,3)):
        ready,ack,release,exited=(worker.get(k,-1) for k in ('ready_monotonic','acknowledged_monotonic','released_monotonic','exit_observed_monotonic'))
        if not (phases[prior_index]['observations'][-1]['finished_monotonic']<=ready<=ack
                <=phases[active_index]['observations'][0]['started_monotonic']
                <=phases[release_index]['observations'][-1]['finished_monotonic']<=release<=exited
                <=phases[next_index]['observations'][0]['started_monotonic']):
            raise ValueError('CUDA live/ack/release visibility brackets invalid')
    expected_counts=(0,1,2,1,0);sets=[];last=start
    stable_sets=[set(p['observations'][0]['compute_pids']) for p in phases]
    allowed=[{frozenset()}, {frozenset(),frozenset(stable_sets[1])},
             {frozenset(stable_sets[1]),frozenset(stable_sets[2])},
             {frozenset(stable_sets[2]),frozenset(stable_sets[1])},
             {frozenset(stable_sets[1]),frozenset()}]
    for phase_index,(phase,count) in enumerate(zip(phases,expected_counts)):
        observations=phase['observations']
        if len(observations)<3:raise ValueError('repeated visibility observations required')
        phase_sets=[]
        transitions=phase.get('transition_observations',[])
        for observation in transitions+observations:
            a,b=observation['started_monotonic'],observation['finished_monotonic']
            if not last<=a<=b<=end:raise ValueError('stale/out-of-order qualification observation')
            last=b
            if observation['command']!=['nvidia-smi','-q','-x','-i',gpu_uuid]:raise ValueError('wrong qualification query identity')
            pids=parse_query(observation['raw_xml'],gpu_uuid)
            if pids!=observation['compute_pids'] or frozenset(pids) not in allowed[phase_index]:raise ValueError('NVML visibility phase/transition mismatch')
            if observation in observations:
                if len(pids)!=count:raise ValueError('NVML visibility phase count mismatch')
                phase_sets.append(pids)
            if observation.get('mps_identity')!=mps:raise ValueError('MPS directory replacement/nonempty proof')
            live=observation.get('owned_live',[])
            expected_live=[] if count==0 else [workers[0]] if count==1 else workers
            expected_identities=[{k:w[k] for k in ('pid','ppid','start_ticks','nonce')} for w in expected_live]
            if live!=expected_identities or observation.get('owned_live_after')!=expected_identities:raise ValueError('owned live probe identity changed across query')
        if any(pids!=phase_sets[0] for pids in phase_sets):raise ValueError('unstable NVML PID association')
        sets.append(set(phase_sets[0]))
    if not sets[1]<sets[2] or sets[3]!=sets[1]:raise ValueError('distinct two-client visibility association not proven')
    return {'valid':True,'association':proof['association'],'source_sha256':expected_source_sha256,
            'gpu_uuid':gpu_uuid,'boot_id':proof['boot_id'],'compute_pid_a':next(iter(sets[1])),
            'compute_pid_b':next(iter(sets[2]-sets[1])),'runtime_identity':workers[0]['runtime_identity'],
            'sampling_limit':'Repeated observations do not exclude transient unsampled clients; not kernel PID mapping.'}


def worker(args):
    mps=mps_identity(args.mps_dir)
    if os.environ.get('CUDA_MPS_PIPE_DIRECTORY')!=args.mps_dir or os.environ.get('CUDA_VISIBLE_DEVICES')!=args.gpu_uuid:
        raise ValueError('MPS/UUID environment must be installed before CUDA import')
    parent=load(Path(args.directory)/'parent.json')
    if birth(os.getppid())!=parent or sha(SOURCE)!=args.source_sha:raise ValueError('source/parent birth mismatch')
    # The context and allocation stay live until the controller releases them.
    import torch
    props=torch.cuda.get_device_properties(0)
    actual=str(props.uuid)
    if actual.removeprefix('GPU-')!=args.gpu_uuid.removeprefix('GPU-'):raise ValueError('CUDA runtime UUID mismatch')
    allocation=torch.empty(8*1024*1024,dtype=torch.uint8,device='cuda:0')
    vector=torch.arange(4096,dtype=torch.int64,device='cuda:0')
    checksum=int(torch.bitwise_xor(vector,challenge_value(args.nonce)).sum().item());torch.cuda.synchronize()
    proof={**birth(os.getpid()),'nonce':args.nonce,'source_sha256':args.source_sha,'parent':parent,
        'gpu_uuid':args.gpu_uuid,'mps_identity':mps,'checksum':checksum,'allocation_bytes':allocation.numel(),
        'synchronized':True,'runtime_identity':{'torch_version':torch.__version__,'cuda_version':torch.version.cuda},
        'ready_monotonic':time.monotonic()}
    directory=Path(args.directory);atomic(directory/'ready.json',proof)
    acknowledged=False
    while time.monotonic()<args.deadline:
        if birth(os.getppid())!=parent or mps_identity(args.mps_dir)!=mps:raise ValueError('parent/MPS identity changed')
        if not acknowledged and (directory/'ack.json').exists():
            ack=load(directory/'ack.json')
            if ack!={'nonce':args.nonce,'pid':proof['pid'],'start_ticks':proof['start_ticks']}:raise ValueError('wrong CUDA proof acknowledgment')
            atomic(directory/'acknowledged.json',ack);acknowledged=True
        if acknowledged and (directory/'release.json').exists():
            if load(directory/'release.json')!={'nonce':args.nonce}:raise ValueError('wrong probe release nonce')
            torch.cuda.synchronize();del allocation,vector
            return
        time.sleep(.05)
    raise TimeoutError('owned CUDA probe absolute deadline')


def qualify(gpu_uuid,output,*,max_seconds=120):
    import fcntl
    if not re.fullmatch('GPU-[0-9a-fA-F-]+',gpu_uuid) or type(max_seconds)is not int or not 15<=max_seconds<=120:
        raise ValueError('exact GPU UUID and finite<=120s bound required')
    output=Path(output).absolute();output.mkdir(parents=False,exist_ok=False)
    started=time.monotonic();deadline=started+max_seconds;work_deadline=deadline-8
    source_sha=sha(SOURCE);parent=birth(os.getpid());children=[];proof={'schema':SCHEMA,'complete':False,
        'source_sha256':source_sha,'gpu_uuid':gpu_uuid,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'qualification_id':str(uuid.uuid4()),'parent':parent,'started_monotonic':started,
        'association':'empirical_sampled_NVML_visibility_under_completeness','phases':[],'workers':[]}
    pipe=output/'mps-pipe';pipe.mkdir(mode=0o700);mps=mps_identity(pipe);proof['mps_identity']=mps
    old_handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP)}
    def interrupted(signum,frame):raise InterruptedError('qualification interrupted; owned probes only cleanup')
    def wait(predicate):
        while time.monotonic()<work_deadline:
            if predicate():return
            time.sleep(.05)
        raise TimeoutError('bounded qualification wait')
    def live(entries):
        values=[]
        for entry in entries:
            if entry['process'].poll()is not None or birth(entry['pid'])!={k:entry[k] for k in ('pid','ppid','start_ticks')}:raise ValueError('owned probe PID/birth changed or died')
            values.append({k:entry[k] for k in ('pid','ppid','start_ticks','nonce')})
        return values
    associations={}
    def phase(name,count,entries):
        observations=[];record={'name':name,'observations':observations};proof['phases'].append(record)
        while time.monotonic()<work_deadline:
            owned=live(entries);sample=query(gpu_uuid,work_deadline)
            sample.update(owned_live=owned,owned_live_after=live(entries),mps_identity=mps_identity(pipe));observations.append(sample)
            if sample['mps_identity']!=mps:raise ValueError('MPS directory changed')
            seen=set(sample['compute_pids'])
            prior=associations.get('a',set())
            if name==PHASES[0] and seen:raise ValueError('initial GPU not idle')
            if name==PHASES[1]:
                if len(seen)>1 or (prior and seen and seen!=prior):raise ValueError('unexpected extra/changed compute client')
                if seen:associations['a']=seen
            if name==PHASES[2]:
                if not prior<=seen or len(seen)>2:raise ValueError('existing compute PID changed or extra client')
                if len(seen)==2:
                    if 'both' in associations and seen!=associations['both']:raise ValueError('second compute PID changed')
                    associations['both']=seen
            if name==PHASES[3] and seen not in (prior,associations['both']):raise ValueError('wrong client after B release')
            if name==PHASES[4] and seen not in (set(),prior):raise ValueError('unknown final residual client')
            if len(observations)>128:raise TimeoutError('bounded transition observation count')
            if len(sample['compute_pids'])==count:
                if len(observations)>=3 and all(o['compute_pids']==sample['compute_pids'] for o in observations[-3:]):
                    record['transition_observations']=observations[:-3];record['observations']=observations[-3:];return
            time.sleep(.15)
        raise TimeoutError('visibility phase did not stabilize')
    def launch(index):
        directory=output/('probe-'+str(index));directory.mkdir();atomic(directory/'parent.json',parent)
        nonce=uuid.uuid4().hex;env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu_uuid,CUDA_MPS_PIPE_DIRECTORY=str(pipe),CUDA_MPS_LOG_DIRECTORY=str(output/'mps-log'))
        log=(directory/'worker.log').open('x')
        process=subprocess.Popen([sys.executable,str(SOURCE),'--worker','--gpu-uuid',gpu_uuid,'--directory',str(directory),
            '--mps-dir',str(pipe),'--nonce',nonce,'--source-sha',source_sha,'--deadline',str(work_deadline)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        # Cleanup scope owns the child immediately after Popen, including failed receipt writes.
        entry={'process':process,'pid':process.pid,'nonce':nonce,'directory':directory,'log':log};children.append(entry)
        entry.update(birth(process.pid));atomic(directory/'owned-launch.json',{k:entry[k] for k in ('pid','ppid','start_ticks','nonce')})
        wait(lambda:(directory/'ready.json').is_file() or process.poll()is not None)
        if process.poll()is not None:raise ValueError('owned CUDA probe failed before ready; retained worker.log')
        value=check_worker(load(directory/'ready.json'),source_sha=source_sha,gpu_uuid=gpu_uuid,mps=mps,parent=parent,owned=entry)
        ack={k:value[k] for k in ('nonce','pid','start_ticks')};atomic(directory/'ack.json',ack)
        wait(lambda:(directory/'acknowledged.json').is_file() or process.poll()is not None)
        if process.poll()is not None or load(directory/'acknowledged.json')!=ack:raise ValueError('probe did not acknowledge retained live context')
        value['ack']=ack;value['acknowledged_monotonic']=time.monotonic();proof['workers'].append(value);return entry
    def release(entry):
        value=next(w for w in proof['workers'] if w['pid']==entry['pid']);value['released_monotonic']=time.monotonic()
        atomic(entry['directory']/'release.json',{'nonce':entry['nonce']})
        wait(lambda:entry['process'].poll()is not None)
        code=entry['process'].returncode
        value.update(exit_code=code,exit_observed_monotonic=time.monotonic())
        if code!=0:raise ValueError('owned probe release failed')
    lock=open(GPU_LOCK,'a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for s in old_handlers:signal.signal(s,interrupted)
        phase(PHASES[0],0,[]);a=launch(0);phase(PHASES[1],1,[a]);b=launch(1);phase(PHASES[2],2,[a,b])
        release(b);phase(PHASES[3],1,[a]);release(a);phase(PHASES[4],0,[])
        if sha(SOURCE)!=source_sha:raise ValueError('qualification source changed')
        proof.update(complete=True,finished_monotonic=time.monotonic());proof['audit']=audit_qualification(proof,expected_source_sha256=source_sha,gpu_uuid=gpu_uuid)
    except BaseException as exc:
        proof.update(complete=False,error=str(exc),finished_monotonic=time.monotonic());raise
    finally:
        # A repeated shutdown signal cannot strand a probe between TERM and reap.
        # Record it while the two bounded owned-child cleanup waits complete.
        proof['cleanup_signals']=[]
        def cleanup_signal(signum,frame):proof['cleanup_signals'].append(signum)
        for s in old_handlers:signal.signal(s,cleanup_signal)
        cleanup_errors=[]
        for entry in children:
            process=entry['process']
            try:
                if process.poll()is None:
                    if birth(process.pid)!={k:entry[k] for k in ('pid','ppid','start_ticks')}:raise ValueError('refuse cleanup of changed PID identity')
                    process.terminate()
                    try:process.wait(timeout=min(2,max(.05,deadline-time.monotonic())))
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=2)
            except BaseException as exc:
                cleanup_errors.append(str(exc))
            finally:entry['log'].close()
        if cleanup_errors:proof.update(complete=False,cleanup_errors=cleanup_errors)
        proof['finished_monotonic']=time.monotonic()
        if proof['finished_monotonic']-started>max_seconds:proof.update(complete=False,error='qualification total bound exceeded')
        try:atomic(output/'qualification-proof.json',proof)
        finally:
            lock.close()
            for s,h in old_handlers.items():signal.signal(s,h)
    return proof


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--gpu-uuid',required=True);parser.add_argument('--output',type=Path)
    parser.add_argument('--max-seconds',type=int,default=120);parser.add_argument('--worker',action='store_true')
    parser.add_argument('--directory');parser.add_argument('--mps-dir');parser.add_argument('--nonce');parser.add_argument('--source-sha');parser.add_argument('--deadline',type=float)
    args=parser.parse_args()
    if args.worker:worker(args)
    else:
        value=qualify(args.gpu_uuid,args.output,max_seconds=args.max_seconds)
        print(json.dumps({'complete':value['complete'],'qualification_id':value['qualification_id']}))

if __name__=='__main__':main()
