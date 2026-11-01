"""Opt-in harness routing; the original baseline implementations are unchanged.

The V5.1 lifecycle adapter is supplied explicitly after the policy API is frozen.
No candidate adapter or injected CPU fixture constitutes empirical evidence.
"""
import argparse,json,signal,traceback
from pathlib import Path
from ..a10_v5_ablation import episode as baseline
from ..a10_v5_ablation.episode import atomic_json,check_deadline,with_hard_deadline
ARMS=('fixed_three','combined','v5_1')
SOURCE_COMMIT_NOTE=baseline.SOURCE_COMMIT_NOTE
source_snapshot=baseline.source_snapshot

def run_episode(manifest,manifest_path,raw_settings,output,calibration,*,arm,mode,
                max_seconds,source_commit,service_factory=None,gpu_reader=None,
                fifo_episode=None,controller_factory=None):
    if arm not in ARMS:raise ValueError('exact three-arm comparison required')
    if arm in ('fixed_three','combined'):
        if fifo_episode is not None:raise ValueError('FIFO adapter cannot change a baseline arm')
        return baseline.run_episode(manifest,manifest_path,raw_settings,output,calibration,
            arm=arm,mode=mode,max_seconds=max_seconds,source_commit=source_commit,
            service_factory=service_factory,gpu_reader=gpu_reader)
    if fifo_episode is None:
        from .fifo_episode import run_episode as fifo_episode
    return fifo_episode(manifest,manifest_path,raw_settings,output,calibration,
        arm=arm,mode=mode,max_seconds=max_seconds,source_commit=source_commit,
        service_factory=service_factory,gpu_reader=gpu_reader,controller_factory=controller_factory)


def validate_manifest(manifest,*,arm,mode):
    if arm not in ARMS:raise ValueError('exact three-arm comparison required')
    if arm=='v5_1' and any(str(k).startswith('development') and k not in {'development','development_seed_offset','development_scope'} for k in manifest):
        raise ValueError('no inherited development controls in V5.1 primary comparison')
    return baseline.validate_manifest(manifest,arm='fixed_three' if arm=='v5_1' else arm,mode=mode)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',nargs='?',choices=('run','check'),default='run')
    for name in ('manifest','settings','output','source-commit'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--calibration');parser.add_argument('--arm',choices=ARMS,default='fixed_three')
    parser.add_argument('--mode',choices=('individual','group'),default='individual')
    parser.add_argument('--max-seconds',type=float,default=600.);parser.add_argument('--deadline-utc')
    args=parser.parse_args(argv);path=Path(args.manifest).resolve();output=Path(args.output).resolve()
    manifest=json.loads(path.read_text());raw_settings=json.loads(Path(args.settings).read_text())
    if args.deadline_utc:manifest=with_hard_deadline(manifest,args.deadline_utc)
    validate_manifest(manifest,arm=args.arm,mode=args.mode);check_deadline(manifest)
    if output.exists():raise FileExistsError('V5 episode output already exists; no overwrite/resume')
    if args.command=='run'and not args.calibration:raise ValueError('same explicit calibration required for every arm')
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'invocation.json',{**vars(args),'manifest':str(path),'output':str(output)})
    if args.command=='check':
        snapshot=source_snapshot(manifest,path,args.source_commit)
        atomic_json(output/'source-pins-before.json',snapshot);atomic_json(output/'source-pins-after.json',snapshot)
        atomic_json(output/'result.json',{'complete':True,'scope':'CPU source check only; no evaluation'})
        return
    import fcntl
    signals=(signal.SIGTERM,signal.SIGINT,signal.SIGHUP);previous={s:signal.getsignal(s)for s in signals}
    def interrupted(number,frame):raise InterruptedError('V5 episode interrupted; owned cleanup required')
    with open('/tmp/localml-a10-feedback-pilot.lock','a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for number in signals:signal.signal(number,interrupted)
            result=run_episode(manifest,path,raw_settings,output,json.loads(Path(args.calibration).read_text()),
                arm=args.arm,mode=args.mode,max_seconds=args.max_seconds,source_commit=args.source_commit)
        except BaseException:
            atomic_json(output/'result.json',{'complete':False,'arm':args.arm,'mode':args.mode,'error':traceback.format_exc(),
                'planned_jobs':len(manifest['jobs']),'note':'Failed/unfinished jobs remain in the denominator.'})
            raise
        finally:
            for number,handler in previous.items():signal.signal(number,handler)
            fcntl.flock(lock,fcntl.LOCK_UN)
    print(json.dumps({'complete':result['complete'],'arm':args.arm,'mode':args.mode,'jobs':len(result['receipts']),
        'v5_flags':result['v5_flags'],'episode_duration_seconds':result['episode_duration_seconds']},sort_keys=True))

if __name__=='__main__':main()
