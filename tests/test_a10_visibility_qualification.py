"""CPU-only saved qualification evidence and cleanup fixtures; never imports CUDA."""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
from localml_scheduler.benchmarks.a10_resource_mechanisms import visibility_qualification as q

GPU='GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee';SHA='a'*64

def xml(pids,*,gpu=GPU,kind='C'):
    return '<nvidia_smi_log><gpu><uuid>'+gpu+'</uuid><processes>'+''.join('<process_info><pid>'+str(p)+'</pid><type>'+kind+'</type></process_info>' for p in pids)+'</processes></gpu></nvidia_smi_log>'

def fixture():
    parent={'pid':10,'ppid':1,'start_ticks':50};mps={'path':'/private/empty','device':1,'inode':2,'uid':1000,'mode':0o700,'empty':True}
    workers=[]
    for i in range(2):
        nonce=str(i+1)*32
        w={'pid':20+i,'ppid':10,'start_ticks':60+i,'nonce':nonce,'source_sha256':SHA,'gpu_uuid':GPU,
            'parent':parent,'mps_identity':mps,'checksum':q.challenge_result(nonce),'allocation_bytes':8*1024*1024,
            'synchronized':True,'runtime_identity':{'torch_version':'fixture'},'ready_monotonic':104.5+3*i,'acknowledged_monotonic':104.6+3*i,
            'released_monotonic':113.5 if i==0 else 110.5,'exit_observed_monotonic':113.6 if i==0 else 110.6,'exit_code':0}
        w['ack']={k:w[k] for k in ('nonce','pid','start_ticks')};workers.append(w)
    proof={'schema':q.SCHEMA,'complete':True,'source_sha256':SHA,'gpu_uuid':GPU,'boot_id':'a'*8+'-'+ 'b'*4+'-'+ 'c'*4+'-'+ 'd'*4+'-'+ 'e'*12,
        'parent':parent,'mps_identity':mps,'workers':workers,'started_monotonic':100.,'finished_monotonic':120.,
        'association':'empirical_sampled_NVML_visibility_under_completeness','phases':[]}
    now=102.
    for name,pids,live in zip(q.PHASES,[[],[1000],[1000,2000],[1000],[]],[[],workers[:1],workers,workers[:1],[]]):
        samples=[]
        for _ in range(3):
            samples.append({'command':['nvidia-smi','-q','-x','-i',GPU],'raw_xml':xml(pids),'compute_pids':pids,
                'started_monotonic':now,'finished_monotonic':now+.1,'mps_identity':mps,
                'owned_live':[{k:w[k] for k in ('pid','ppid','start_ticks','nonce')} for w in live]});now+=1
        for sample in samples:sample['owned_live_after']=deepcopy(sample['owned_live'])
        proof['phases'].append({'name':name,'observations':samples,'transition_observations':[]})
    return proof

def audit(p):return q.audit_qualification(p,expected_source_sha256=SHA,gpu_uuid=GPU)

def test_two_owned_context_visibility_replays_and_never_claims_kernel_mapping():
    result=audit(fixture());assert result['valid'] and result['compute_pid_a']==1000 and result['compute_pid_b']==2000
    assert 'not kernel PID mapping' in result['sampling_limit']

@pytest.mark.parametrize('kind',['extra','changed_pid','one_instead_two','graphics','wrong_uuid','raw_tamper','wrong_command',
    'nonce','checksum','ancestry','birth','stale','source','ack','mps','runtime','incomplete','deadline','order','observations','ack_timing','release_timing','birth_after'])
def test_fail_closed_evidence_rejections(kind):
    p=deepcopy(fixture());sample=p['phases'][2]['observations'][0]
    if kind in ('extra','changed_pid','one_instead_two'):
        pids={'extra':[1000,2000,3000],'changed_pid':[1000,3000],'one_instead_two':[1000]}[kind]
        sample.update(compute_pids=pids,raw_xml=xml(pids))
    elif kind=='graphics':sample['raw_xml']=xml([1000],kind='G')
    elif kind=='wrong_uuid':sample['raw_xml']=xml([1000,2000],gpu='GPU-wrong')
    elif kind=='raw_tamper':sample['raw_xml']=xml([1000])
    elif kind=='wrong_command':sample['command'][-1]='GPU-wrong'
    elif kind=='nonce':p['workers'][0]['nonce']='9'*32
    elif kind=='checksum':p['workers'][0]['checksum']+=1
    elif kind=='ancestry':p['workers'][0]['ppid']=99
    elif kind=='birth':sample['owned_live'][0]['start_ticks']+=1
    elif kind=='birth_after':sample['owned_live_after'][0]['start_ticks']+=1
    elif kind=='stale':p['workers'][0]['ready_monotonic']=1.
    elif kind=='source':p['source_sha256']='b'*64
    elif kind=='ack':p['workers'][0]['ack']['nonce']='8'*32
    elif kind=='mps':sample['mps_identity']={**p['mps_identity'],'inode':999}
    elif kind=='runtime':p['workers'][1]['runtime_identity']={'torch_version':'changed'}
    elif kind=='incomplete':p['complete']=False
    elif kind=='deadline':p['finished_monotonic']=222.
    elif kind=='order':sample['started_monotonic']=50.
    elif kind=='ack_timing':p['workers'][0]['acknowledged_monotonic']=110.
    elif kind=='release_timing':p['workers'][1]['released_monotonic']=109.
    else:p['phases'][0]['observations'].pop()
    with pytest.raises((ValueError,KeyError)):audit(p)


def test_retained_startup_transition_replays_but_unknown_transient_fails():
    p=deepcopy(fixture());phase=p['phases'][1];transition=deepcopy(phase['observations'][0])
    transition.update(started_monotonic=104.2,finished_monotonic=104.3,compute_pids=[],raw_xml=xml([]))
    phase['transition_observations']=[transition]
    assert audit(p)['valid']
    transition.update(compute_pids=[9000],raw_xml=xml([9000]))
    with pytest.raises(ValueError,match='transition'):audit(p)


def test_private_mps_path_rejects_nonempty_symlink_replacement(tmp_path):
    pipe=tmp_path/'pipe';pipe.mkdir(mode=0o700);identity=q.mps_identity(pipe)
    (pipe/'server').write_text('not empty')
    with pytest.raises(ValueError,match='MPS'):q.mps_identity(pipe)
    (pipe/'server').unlink();pipe.rmdir();pipe.mkdir(mode=0o700)
    # A reused inode is possible; replacement identity is compared as a complete record by audit.
    changed={**identity,'inode':identity['inode']+1};assert changed!=identity
    pipe.rmdir();pipe.symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError,match='MPS'):q.mps_identity(pipe)


def test_wrong_owned_launch_birth_and_nonce_fail():
    p=fixture();w=p['workers'][0];owned={k:w[k] for k in ('pid','ppid','start_ticks','nonce')};owned['start_ticks']+=1
    with pytest.raises(ValueError,match='identity'):q.check_worker(w,source_sha=SHA,gpu_uuid=GPU,mps=p['mps_identity'],parent=p['parent'],owned=owned)


def test_failed_launch_receipt_write_still_cleans_only_owned_child(tmp_path,monkeypatch):
    class Child:
        pid=43210;code=None;terminated=False
        def poll(self):return self.code
        def terminate(self):
            import fcntl
            with open(q.GPU_LOCK,'a') as competing:
                with pytest.raises(BlockingIOError):fcntl.flock(competing,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.terminated=True;self.code=-15
        def wait(self,timeout):
            import signal
            signal.raise_signal(signal.SIGTERM)
            return self.code
    child=Child();parent={'pid':os.getpid(),'ppid':os.getppid(),'start_ticks':100}
    monkeypatch.setattr(q,'birth',lambda pid:parent if pid==os.getpid() else {'pid':pid,'ppid':os.getpid(),'start_ticks':101})
    monkeypatch.setattr(q.subprocess,'Popen',lambda *a,**kw:child)
    monkeypatch.setattr(q,'GPU_LOCK',str(tmp_path/'gpu.lock'))
    def query(gpu_uuid,deadline):
        import time
        now=time.monotonic();return {'command':['nvidia-smi','-q','-x','-i',GPU],'raw_xml':xml([]),'compute_pids':[], 'started_monotonic':now,'finished_monotonic':now}
    monkeypatch.setattr(q,'query',query);monkeypatch.setattr(q.time,'sleep',lambda seconds:None)
    original=q.atomic
    def fail(path,value):
        if Path(path).name=='owned-launch.json':raise OSError('injected launch receipt failure')
        return original(path,value)
    monkeypatch.setattr(q,'atomic',fail)
    with pytest.raises(OSError,match='receipt'):q.qualify(GPU,tmp_path/'output',max_seconds=15)
    proof=json.loads((tmp_path/'output/qualification-proof.json').read_text())
    assert child.terminated and not proof['complete'] and 'receipt failure' in proof['error']
    import signal
    assert proof['cleanup_signals']==[signal.SIGTERM]


def test_no_cuda_import_on_controller_or_pure_replay():
    # Module imports only stdlib; Torch is confined to worker after MPS checks.
    import ast
    tree=ast.parse(Path(q.__file__).read_text())
    imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert all('torch' not in ast.unparse(n) for n in imports)
    assert audit(fixture())['valid']
