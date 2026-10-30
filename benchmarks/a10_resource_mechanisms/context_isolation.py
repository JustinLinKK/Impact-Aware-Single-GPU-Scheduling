"""Empirical sampled CUDA-context attribution, conditional on NVML completeness.

No host PID mapping is claimed. Qualification + fresh owned CUDA proof + idle /
singleton brackets establish an association only for this launch's lifetime.
"""
from __future__ import annotations
import hashlib,json,os,re,signal,time,uuid
from pathlib import Path
from . import visibility_qualification as visibility

SOURCE=Path(__file__).resolve()
SCHEMA='a10_empirical_context_association_v1'
_SENTINEL=None

def checksum(nonce):return (visibility.challenge_value(nonce)^0x5A5A)*3+7

def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def owned(records,root):
    by={r['pid']:r for r in records}
    if len(by)!=len(records) or root['pid'] not in by or by[root['pid']]['start_ticks']!=root['start_ticks']:
        raise ValueError('missing/reused context launch root')
    result={root['pid']}
    while True:
        extra={pid for pid,row in by.items() if row['ppid'] in result}-result
        if not extra:return by,result
        result|=extra

def check_live(records,root,worker):
    by,pids=owned(records,root)
    if worker['pid'] not in pids or by[worker['pid']].get('state')=='Z' or any(by[worker['pid']][k]!=worker[k] for k in ('pid','ppid','start_ticks')):
        raise ValueError('CUDA proof is not a live owned descendant with matching birth')

def qualification_gate(proof,gpu_uuid,*,runtime=None,live=False):
    result=visibility.audit_qualification(proof,expected_source_sha256=visibility.sha(visibility.SOURCE),gpu_uuid=gpu_uuid)
    if runtime is not None and result['runtime_identity']['torch_version']!=runtime['torch_version']:
        raise ValueError('qualification/calibration actual runtime changed')
    if live and (proof['boot_id']!=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                 or not 0<=time.monotonic()-proof['finished_monotonic']<=600):
        raise ValueError('stale qualification or different host boot')
    return result

def audit_association(value,qualification,*,spec,gpu_uuid,require_finished=True):
    qualification_gate(qualification,gpu_uuid)
    request=value['request'];ready=value['ready'];root=value['root'];pipe=value['mps_identity']
    if (value.get('schema')!=SCHEMA or value.get('qualification_sha256')!=canonical(qualification)
            or request.get('source_sha256')!=visibility.sha(SOURCE) or request.get('gpu_uuid')!=gpu_uuid
            or request.get('spec_sha256')!=spec['spec_sha256'] or request.get('signature')!=spec['expected_runtime_signature']
            or ready.get('request')!=request or ready.get('source_sha256')!=visibility.sha(SOURCE)
            or ready.get('checksum')!=checksum(request['nonce'])
            or not re.fullmatch('[0-9a-f]{32}',str(request.get('nonce','')))
            or ready.get('sentinel_bytes')!=8
            or ready.get('mps_identity')!=pipe or pipe.get('empty')is not True or pipe.get('mode')!=0o700
            or ready.get('gpu_uuid')!=gpu_uuid or ready.get('synchronized')is not True
            or ready.get('runtime_identity')!=qualification['workers'][0]['runtime_identity']
            or ready.get('runtime_identity',{}).get('torch_version')!=spec['torch_version']
            or value.get('ack')!={'nonce':request['nonce'],'pid':ready['pid'],'start_ticks':ready['start_ticks']}
            or len(value.get('idle_queries',[]))<3 or len(value.get('singleton_queries',[]))<3):
        raise ValueError('fresh source/spec/runtime/nonce/context association proof mismatch')
    if not (request['issued_monotonic']<=ready['ready_monotonic']<=value['acknowledged_monotonic']<=value['released_monotonic']):
        raise ValueError('context challenge acknowledgment/release chronology invalid')
    last=request['issued_monotonic'];inferred=None
    for phase in ('idle_queries','singleton_queries'):
        for observation in value[phase]:
            pids=visibility.parse_query(observation['raw_xml'],gpu_uuid)
            if pids!=observation['compute_pids'] or observation['command']!=['nvidia-smi','-q','-x','-i',gpu_uuid]:raise ValueError('raw context NVML query mismatch')
            if observation['mps_identity']!=pipe:raise ValueError('MPS pipe replaced/nonempty')
            a,b=observation['started_monotonic'],observation['finished_monotonic']
            if not last<=a<=b<=value['released_monotonic']:raise ValueError('stale context query')
            last=b
            if phase=='idle_queries':
                if pids:raise ValueError('context launch not initially idle')
                if b>ready['ready_monotonic']:raise ValueError('idle evidence after CUDA proof')
            else:
                if a<ready['ready_monotonic'] or len(pids)!=1:raise ValueError('context singleton unavailable')
                if inferred is not None and inferred!=pids[0]:raise ValueError('context NVML PID changed')
                inferred=pids[0]
                for which in ('process_snapshot_before','process_snapshot_after'):check_live(observation[which],root,ready)
    if require_finished:
        finished=value.get('finished',{})
        if (any(finished.get(k)!=ready[k] for k in ('pid','start_ticks','source_sha256'))
                or finished.get('nonce')!=request['nonce']
                or not ready['ready_monotonic']<=value['released_monotonic']<=finished.get('released_monotonic',-1)
                or value.get('eligible_after_monotonic')!=finished['released_monotonic']):
            raise ValueError('missing/mismatched worker-observed context release interval')
    if inferred!=value['compute_pid'] or value['released_monotonic']<last:raise ValueError('wrong inferred NVML context PID')
    return {'compute_pid':inferred,'visible_pid':ready['pid'],'start_ticks':ready['start_ticks'],
            'method':'empirical sampled association under NVML completeness; not kernel PID mapping'}

def sample_lifecycle(sample,association):
    """Classify only the already-associated singleton; teardown never measures."""
    pids=[app['pid'] for app in sample['compute_apps']]
    if not pids:return 'terminal_context_absent_or_zombie'
    if pids!=[association['compute_pid']]:raise ValueError('unknown/extra GPU context during eligible calibration')
    root,worker=association['root'],association['ready']
    states=[]
    for key in ('process_snapshot_before','process_snapshot'):
        by,_=owned(sample[key],root)
        row=by.get(worker['pid'])
        if row is not None and any(row[k]!=worker[k] for k in ('pid','ppid','start_ticks')):
            raise ValueError('context worker reused or ancestry changed during query')
        terminal=row is None or row.get('state')=='Z'
        if not terminal:check_live(sample[key],root,worker)
        states.append(terminal)
    if states==[True,False]:raise ValueError('terminal context resurrected during query')
    return ('known_context_terminal_transition' if states==[False,True] else
            'known_context_terminal_drain' if states==[True,True] else 'live')

def audit_sample(sample,association):
    pids=[app['pid'] for app in sample['compute_apps']]
    query=sample['nvml_process_query']
    if (visibility.parse_query(query['raw_xml'],sample['gpu']['gpu_uuid'])!=pids
            or query['compute_pids']!=pids or query['command']!=['nvidia-smi','-q','-x','-i',sample['gpu']['gpu_uuid']]
            or not sample['sample_started_monotonic']<=query['started_monotonic']<=query['finished_monotonic']<=sample['sample_finished_monotonic']
            or sample.get('mps_identity')!=association['mps_identity']):raise ValueError('raw eligible context query/MPS/clock mismatch')
    if any(app['gpu_uuid']!=sample['gpu']['gpu_uuid'] for app in sample['compute_apps']):raise ValueError('sample process GPU UUID mismatch')
    if not pids:
        if sample.get('window_eligible')is not False or sample.get('drain_classification')!='terminal_context_absent_or_zombie':
            raise ValueError('empty visibility cannot be a live eligible sample')
        for key in ('process_snapshot_before','process_snapshot'):
            matches=[r for r in sample[key] if r['pid']==association['ready']['pid']]
            if len(matches)>1 or any(r['start_ticks']!=association['ready']['start_ticks'] or r.get('state')!='Z' for r in matches):
                raise ValueError('retained-sentinel worker still live or reused while NVML is empty')
        if sample.get('context_association_sha256')!=canonical(association):raise ValueError('unbound terminal drain')
        return {'observed_compute_pids':[],'window_eligible':False,'classification':'terminal_context_absent_or_zombie'}
    classification=sample_lifecycle(sample,association)
    if classification!='live':
        if (sample.get('window_eligible')is not False or sample.get('drain_classification')!=classification
                or sample.get('context_association_sha256')!=canonical(association)):
            raise ValueError('unbound or eligible terminal context transition')
        return {'observed_compute_pids':pids,'window_eligible':False,'classification':classification}
    if sample.get('sample_started_monotonic',-1)<association['eligible_after_monotonic']:raise ValueError('startup sample is not eligible')
    if sample.get('context_association_sha256')!=canonical(association) or sample.get('window_eligible')is not True:
        raise ValueError('startup or unbound context sample is ineligible')
    return {'observed_compute_pids':pids,'visible_owned_context_pid':association['ready']['pid'],
            'method':'empirical_sampled_NVML_association_under_completeness'}


def worker_challenge(spec):
    """Called before model construction/windows; environment was set before import."""
    directory_raw=os.environ.get('A10_CONTEXT_PROOF_DIRECTORY')
    if directory_raw is None:return  # Evaluation uses identical workload without this calibration-only challenge.
    directory=Path(directory_raw);deadline=float(os.environ['A10_CONTEXT_PROOF_DEADLINE'])
    while not (directory/'request.json').is_file():
        if time.monotonic()>=deadline:raise TimeoutError('context request deadline')
        time.sleep(.02)
    parent_identity=visibility.birth(os.getppid())
    request=visibility.load(directory/'request.json');pipe=request['mps_identity']
    if (visibility.mps_identity(pipe['path'])!=pipe or os.environ.get('CUDA_MPS_PIPE_DIRECTORY')!=pipe['path']
            or os.environ.get('CUDA_VISIBLE_DEVICES')!=request['gpu_uuid'] or request['source_sha256']!=visibility.sha(SOURCE)
            or request['spec_sha256']!=spec['spec_sha256'] or request['signature']!=spec['expected_runtime_signature']):
        raise ValueError('context challenge source/spec/MPS/UUID environment mismatch')
    global _SENTINEL
    import torch
    if str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')!=request['gpu_uuid'].removeprefix('GPU-'):raise ValueError('actual CUDA context UUID mismatch')
    _SENTINEL=torch.tensor(visibility.challenge_value(request['nonce']),dtype=torch.int64,device='cuda:0')
    _SENTINEL.bitwise_xor_(0x5A5A).mul_(3).add_(7)
    observed=int(_SENTINEL.item());torch.cuda.synchronize()
    ready={**visibility.birth(os.getpid()),'source_sha256':visibility.sha(SOURCE),'request':request,
           'gpu_uuid':request['gpu_uuid'],'mps_identity':pipe,'checksum':observed,'sentinel_bytes':8,'synchronized':True,
           'runtime_identity':{'torch_version':torch.__version__,'cuda_version':torch.version.cuda},'ready_monotonic':time.monotonic()}
    visibility.atomic(directory/'ready.json',ready)
    while time.monotonic()<deadline:
        if visibility.birth(os.getppid())!=parent_identity:raise ValueError('context parent exited/reused')
        if visibility.mps_identity(pipe['path'])!=pipe:raise ValueError('context MPS path replaced')
        if (directory/'ack.json').is_file():
            ack=visibility.load(directory/'ack.json')
            if ack!={'nonce':request['nonce'],'pid':ready['pid'],'start_ticks':ready['start_ticks']}:raise ValueError('context nonce acknowledgment mismatch')
            visibility.atomic(directory/'acknowledged.json',ack);break
        time.sleep(.02)
    else:raise TimeoutError('context proof acknowledgment deadline')
    while not (directory/'release.json').is_file():
        if visibility.birth(os.getppid())!=parent_identity:raise ValueError('context parent exited/reused')
        if time.monotonic()>=deadline:raise TimeoutError('context proof release deadline')
        time.sleep(.02)
    if visibility.load(directory/'release.json')!={'nonce':request['nonce']}:raise ValueError('wrong context release nonce')
    torch.cuda.synchronize()  # Retain exactly one int64 sentinel until this worker exits.
    visibility.atomic(directory/'finished.json',{'nonce':request['nonce'],'pid':ready['pid'],
        'start_ticks':ready['start_ticks'],'source_sha256':ready['source_sha256'],'released_monotonic':time.monotonic()})


def prepare(directory,*,gpu_uuid,spec,qualification,deadline):
    directory=Path(directory);directory.mkdir(exist_ok=False);pipe=directory/'mps-pipe';pipe.mkdir(mode=0o700)
    mps=visibility.mps_identity(pipe);request={'nonce':uuid.uuid4().hex,'gpu_uuid':gpu_uuid,
        'spec_sha256':spec['spec_sha256'],'signature':spec['expected_runtime_signature'],
        'source_sha256':visibility.sha(SOURCE),'mps_identity':mps,'issued_monotonic':time.monotonic(),'issued_unix':time.time()}
    value={'schema':SCHEMA,'request':request,'mps_identity':mps,'qualification_sha256':canonical(qualification),'idle_queries':[],'singleton_queries':[]}
    for _ in range(3):
        sample=visibility.query(gpu_uuid,deadline);sample['mps_identity']=visibility.mps_identity(pipe)
        if sample['compute_pids']:raise ValueError('GPU not idle before fresh context launch')
        value['idle_queries'].append(sample)
    visibility.atomic(directory/'request.json',request)
    return value,{'A10_CONTEXT_PROOF_DIRECTORY':str(directory),'A10_CONTEXT_PROOF_DEADLINE':str(deadline),
        'CUDA_VISIBLE_DEVICES':gpu_uuid,'CUDA_MPS_PIPE_DIRECTORY':str(pipe),'CUDA_MPS_LOG_DIRECTORY':str(directory/'mps-log')}

def associate(value,directory,*,handle,root,snapshot,qualification,spec,gpu_uuid,deadline):
    directory=Path(directory);value['root']=root
    def wait(path):
        while time.monotonic()<deadline:
            if handle.process.poll()is not None:raise ValueError('context worker exited before association')
            if path.is_file():return visibility.load(path)
            time.sleep(.02)
        raise TimeoutError('bounded fresh context proof wait')
    ready=wait(directory/'ready.json');value['ready']=ready
    for _ in range(3):
        before=snapshot();sample=visibility.query(gpu_uuid,deadline);after=snapshot()
        check_live(before,root,ready);check_live(after,root,ready);value['ownership_verified']=True
        sample.update(process_snapshot_before=before,process_snapshot_after=after,mps_identity=visibility.mps_identity(value['mps_identity']['path']))
        if len(sample['compute_pids'])!=1:raise ValueError('fresh owned CUDA context did not produce one visible client')
        value['singleton_queries'].append(sample)
    value['compute_pid']=value['singleton_queries'][0]['compute_pids'][0]
    ack={k:ready[k] for k in ('pid','start_ticks')};ack['nonce']=value['request']['nonce'];value['ack']=ack
    visibility.atomic(directory/'ack.json',ack)
    if wait(directory/'acknowledged.json')!=ack:raise ValueError('context acknowledgment changed')
    value['acknowledged_monotonic']=time.monotonic();value['released_monotonic']=time.monotonic()
    audit_association(value,qualification,spec=spec,gpu_uuid=gpu_uuid,require_finished=False)
    visibility.atomic(directory/'release.json',{'nonce':value['request']['nonce']})
    value['finished']=wait(directory/'finished.json');value['eligible_after_monotonic']=value['finished']['released_monotonic']
    audit_association(value,qualification,spec=spec,gpu_uuid=gpu_uuid)
    visibility.atomic(directory/'association.json',value)
    return value


def cleanup_owned(handle, *, snapshot, root=None, association=None):
    """Kill only observed descendants with matching birth, including other sessions."""
    records=snapshot();by={r['pid']:r for r in records};identities={}
    root=root or (association or {}).get('root')
    if root is not None:
        try:
            by,pids=owned(records,root)
            identities.update({pid:by[pid]['start_ticks'] for pid in pids})
        except ValueError:pass
    if association is not None and association.get('ownership_verified')is True:
        for key in ('root','ready'):
            row=association.get(key,{})
            if row.get('pid') and row.get('start_ticks'):identities[row['pid']]=row['start_ticks']
    if handle.process.poll()is None and handle.process.pid in by:identities.setdefault(handle.process.pid,by[handle.process.pid]['start_ticks'])
    old={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP)}
    def alive(pid,ticks):
        try:return visibility.birth(pid)['start_ticks']==ticks
        except (OSError,ValueError,IndexError):return False
    try:
        for s in old:signal.signal(s,lambda *_:None)
        for sig in (signal.SIGTERM,signal.SIGKILL):
            for pid,ticks in identities.items():
                if alive(pid,ticks):
                    try:os.kill(pid,sig)
                    except ProcessLookupError:pass
            until=time.monotonic()+.75
            while time.monotonic()<until and any(alive(pid,ticks) for pid,ticks in identities.items()):time.sleep(.02)
        try:handle.process.wait(timeout=.25)
        except Exception:pass
    finally:
        for s,h in old.items():signal.signal(s,h)


def audit_raw_evidence(spec,evidence,*,gpu_uuid):
    """Pure offline entry point for local exporters; no Torch/runtime recomputation."""
    qualification=evidence['visibility_qualification'];qualification_gate(qualification,gpu_uuid)
    launches=evidence['launches']
    if len(launches)!=2 or len({launch['context_association']['request']['nonce'] for launch in launches})!=2:
        raise ValueError('two fresh per-launch/resume context associations required')
    if evidence['resources'].get('calibration_context_sentinel_bytes')!=8:raise ValueError('explicit calibration-only sentinel accounting required')
    for launch in launches:
        association=launch['context_association']
        if association['root']!={'pid':launch['pid'],'start_ticks':launch['pid_start_ticks']}:raise ValueError('context root differs from launch birth')
        audit_association(association,qualification,spec=spec,gpu_uuid=gpu_uuid)
    for sample in evidence['samples']:
        index=sample['launch']
        if type(index)is not int or index not in (0,1) or sample['gpu']['gpu_uuid']!=gpu_uuid:
            raise ValueError('eligible sample launch/GPU identity mismatch')
        audit_sample(sample,launches[index]['context_association'])
    for window in evidence['windows']:
        index=window['launch']
        if type(index)is not int or index not in (0,1):raise ValueError('window launch identity mismatch')
        association=launches[index]['context_association'];request=association['request']
        release=association['eligible_after_monotonic']+request['issued_unix']-request['issued_monotonic']
        if window['started_at']<release:raise ValueError('window overlaps unacknowledged context challenge')
    return {'valid':True,'method':'empirical sampled NVML isolation conditional on completeness; not kernel PID mapping',
            'calibration_only_sentinel_bytes':8,'memory_observations_corrected':False}


def bounded_context_deadline(*,started,job_started,max_seconds,job_timeout,absolute_unix,now_monotonic=None,now_unix=None):
    now_monotonic=time.monotonic() if now_monotonic is None else now_monotonic
    now_unix=time.time() if now_unix is None else now_unix
    deadline=min(started+max_seconds,job_started+job_timeout,now_monotonic+45,
                 now_monotonic+(absolute_unix-now_unix))
    if deadline<=now_monotonic:raise TimeoutError('absolute/job/campaign context deadline before preparation/dispatch')
    return deadline
