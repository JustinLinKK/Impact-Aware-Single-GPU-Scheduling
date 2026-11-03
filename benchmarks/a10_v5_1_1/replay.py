"""Independent same-path terminal replay of receipts, calibration and history."""
import argparse
import json
import math
from pathlib import Path

from .protocol import validate_plan, calibration_plan
from .campaign import coverage
from .audit import audit_episode
from .episode import source_snapshot
from ...scheduler.fifo_experience import ExperienceStore
from ...scheduler.workload_fingerprint import digest
from ..a10_feedback_latency.run import file_hash, atomic_json
from ..a10_v5_1_ablation.protocol import validate_table, validate_neural_table
from ..a10_resource_mechanisms.profiles import SOURCE, fit_templates
from ..a10_resource_mechanisms.calibrate import extract_row
from ..a10_v5_ablation.calibrate_predictor import _verify_receipt
from ..a10_v5_1_ablation.fresh_calibration import freshness, observation_range


def load(path): return json.loads(Path(path).read_text())


def calibration_audit(measured_path, neural_path, manifest_path, *, gpu_uuid):
    measured_path, neural_path, manifest_path = map(Path, (measured_path, neural_path, manifest_path))
    root = measured_path.parent
    table, neural, manifest = load(measured_path), load(neural_path), load(manifest_path)
    validate_table(table); validate_neural_table(neural, table)
    rows, audits = load(root / 'calibration-rows.json'), load(root / 'validity-audits.json')
    if len(rows) != 8 or len(manifest['calibration']) != 8 or table['gpu_uuid'] != gpu_uuid:
        raise ValueError('eight same-GPU real calibration receipts required')
    byid = {row['job_signature']: row for row in rows}
    pins = {str(p): file_hash(p) for p in (measured_path, neural_path, manifest_path)}
    from ..a10_feedback_latency.run import source_snapshot as calibration_snapshot
    before, after = root / 'source-snapshot.json', root / 'source-snapshot-after.json'
    if load(before) != load(after) or load(before) != calibration_snapshot(manifest, manifest_path):
        raise ValueError('dedicated calibration source identity changed')
    pins.update({str(p): file_hash(p) for p in (before, after)})
    from ..a10_resource_mechanisms import visibility_qualification as visibility
    qual_name = 'variant-qualification' if root.name == 'variant-calibration' else 'qualification'
    qual = root.parent / qual_name / 'qualification-proof.json'
    visibility.audit_qualification(load(qual), expected_source_sha256=file_hash(visibility.__file__), gpu_uuid=gpu_uuid)
    pins[str(qual)] = file_hash(qual)
    for spec in manifest['calibration']:
        work = root / 'jobs' / spec['name']; raw_path = work / 'raw-calibration-evidence.json'
        raw = load(raw_path)
        value = extract_row(spec, hardware_key=table['hardware_key'], gpu_uuid=gpu_uuid, **raw)
        if byid.get(spec['expected_runtime_signature']) != value or audits[spec['name']] != raw['final_audit']:
            raise ValueError('raw calibration replay mismatch')
        _verify_receipt(spec, raw, work); pins[str(raw_path)] = file_hash(raw_path)
    if fit_templates(rows, hardware_key=table['hardware_key'], gpu_uuid=gpu_uuid,
                     source_identity=table['source_identity']) != table:
        raise ValueError('template reconstruction mismatch')
    conversion = neural_path.parent / 'conversion-proof.json'; proof = load(conversion)
    if proof.get('complete') is not True or proof.get('cpu_only') is not True:
        raise ValueError('CPU-only released predictor conversion proof required')
    from ..a10_feedback_latency.run import STUDENT_SHA256
    selected={s['name'] for s in manifest['calibration'] if not s.get('ballast_mib',0)}
    excluded={s['name'] for s in manifest['calibration'] if s.get('ballast_mib',0)}
    if (proof.get('predictor_artifact_sha256')!=STUDENT_SHA256 or
            set(proof['selected_no_ballast_names'])!=selected or set(proof['excluded_ballast_names'])!=excluded
            or len(selected)!=6 or len(excluded)!=2):
        raise ValueError('unaltered released S_A10 conversion selection required')
    for name, expected in proof['inputs_sha256'].items():
        if file_hash(name) != expected: raise ValueError('conversion input changed')
        pins[name] = expected
    for name, expected in proof['output_sha256'].items():
        path = neural_path.parent / name
        if file_hash(path) != expected: raise ValueError('conversion output changed')
        pins[str(path)] = expected
    pins[str(conversion)] = file_hash(conversion)
    return {'valid': True, 'receipts': 8, 'input_sha256': pins}


def history_audit(path, namespace, episodes):
    replay = ExperienceStore(namespace)
    observations = queries = 0
    for line in (Path(path) / 'experience.jsonl').read_text().splitlines():
        event = json.loads(line); proof = event.pop('sha256')
        if (digest(event) != proof or event['parent_sha256'] != replay.parent
                or event['namespace'] != namespace or event['sequence'] != len(replay.events)):
            raise ValueError('history event chain, sequence or namespace mismatch')
        payload = event['payload']; kind = payload['kind']
        if kind == 'observation':
            row = payload['row']; record = episodes[row['episode']]
            result, offset = record['result'], record['offset']
            receipt = result['receipts'][row['job_id']]
            if (abs(row['accepted_unix'] - receipt['accepted_unix']) > .1
                    or not row['accepted_unix'] <= row['window_start_unix'] < row['observed_unix'] <= receipt['feedback_ready_unix'] + .1):
                raise ValueError('history observation outside actual accepted job lifetime')
            cp = row.get('counter_proof') or {}
            if (not isinstance(cp.get('start_step'), int) or not isinstance(cp.get('end_step'), int)
                    or not 0 < cp['start_step'] < cp['end_step'] <= receipt['expected_updates']
                    or cp['end_step'] - cp['start_step'] != row['updates']
                    or row['seconds'] < 1 or row['updates'] < 4):
                raise ValueError('history successful-update counters not proven')
            windows = [h for e in result['alpha_events'] if e['kind'] == 'valid_windows'
                       for h in e['halves'] if row['job_id'] in h['rates']]
            matches = [h for h in windows if abs(h['end'] + offset - row['observed_unix']) < .01
                       and h['start_steps'][row['job_id']] == cp['start_step']
                       and h['end_steps'][row['job_id']] == cp['end_step']
                       and h['stable_identity_sha256'] == cp['stable_identity_sha256']]
            if len(matches) != 1: raise ValueError('history does not bind one actual stable counter window')
            window = matches[0]
            phases = cp.get('resident_phases')
            if (not phases or phases != window.get('resident_phases')
                    or phases.get(row['job_id']) != 'training'
                    or set(window['rates']) != {j for j, phase in phases.items() if phase == 'training'}
                    or cp.get('peer_context') != row['context']):
                raise ValueError('history phase/peer proof differs from stable resident window')
            snapshots = [e for e in result['alpha_events'] if e['kind'] == 'phase_snapshot'
                         and e.get('identity_sha256') == cp['stable_identity_sha256']]
            for at, counter_key in ((window['start'], 'start_step'), (window['end'], 'end_step')):
                proof_snapshots = [e for e in snapshots if e['at'] == at
                    and digest(e['identity']) == e['identity_sha256']
                    and e['resident_phases'] == phases
                    and e['counters'].get(row['job_id']) == cp[counter_key]]
                if len(proof_snapshots) != 1:
                    raise ValueError('phase history lacks exact stable endpoint counter/identity snapshots')
            phase_counts = row['context']['phases']
            if (phase_counts['resident'] != len(phases) or any(
                    phase_counts[p] != list(phases.values()).count(p) for p in ('training', 'starting', 'finishing'))):
                raise ValueError('history resident phase count mismatch')
            for resident in phases:
                rr = result['receipts'][resident]
                if not rr['accepted_unix'] <= row['window_start_unix'] < row['observed_unix'] <= rr['feedback_ready_unix'] + .1:
                    raise ValueError('resident phase evidence outside accepted lifetime')
            observations += 1
        elif kind in {'decision', 'history_query'}:
            episode = episodes[payload['episode']]
            at = payload['at'] + episode['offset']
            if payload['history_parent_sha256'] != replay.parent:
                raise ValueError('decision did not read its declared preceding history state')
            if payload.get('reason') == 'compatible_initial_prefix_prior':
                prior = replay.control.get('context_priors', {}).get(payload['context_key'])
                if (prior != payload['prior'] or not prior or
                        prior['basis'] != 'two_clean_windows_at_actual_occupancy' or
                        not 0 <= at - prior['observed_unix'] <= 14400 or
                        abs(payload['alpha'] - payload['previous_alpha']) != 1 or
                        replay.control.get('pending') is not None):
                    raise ValueError('warm initial prior lacks preceding compatible observed evidence')
            for query in payload['history_queries']:
                receipt = episode['result']['receipts'][query['target_job']]
                if receipt['accepted_unix'] > at + .1: raise ValueError('future accepted job inspected')
                if len(query['neighbors']) > 16: raise ValueError('neighbor budget exceeded')
                independent = set()
                for neighbor in query['neighbors']:
                    row = replay.rows.get(neighbor['aggregate_key'])
                    if (not row or row['episode'] != neighbor['episode'] or
                            row['observed_unix'] != neighbor['observed_unix'] or
                            not 0 <= at - row['observed_unix'] <= 14400 or
                            not math.isclose(row['raw_rate'], neighbor['raw_rate'], rel_tol=1e-12)):
                        raise ValueError('future/stale/fabricated neighbor or measurement')
                    independent.add(row['episode'])
                if len(independent) != len(query['neighbors']) or (not query['exact'] and len(independent) < 3):
                    raise ValueError('history windows counted as independent episodes')
                queries += 1
        elif kind == 'control':
            control = payload['state']
            if len(control.get('tried', [])) > 6: raise ValueError('trace exploration budget exceeded')
        replay._apply(payload); replay.parent = proof; replay.events.append({**event, 'sha256': proof})
    saved = load(Path(path) / 'index.json')
    if saved['parent_sha256'] != replay.parent or saved['rows'] != replay.rows or saved['control'] != replay.control:
        raise ValueError('terminal history index differs from append-only reconstruction')
    return {'valid': True, 'namespace': namespace, 'observations': observations,
            'queried_estimates': queries, 'aggregates': len(replay.rows), 'parent_sha256': replay.parent}


def replay_campaign(root):
    root = Path(root).resolve()
    before = load(root / 'campaign-protocol-before.json'); plan = validate_plan(before['plan'])
    records, generations = load(root / 'campaign-results.json'), load(root / 'calibration-generations.json')
    if len(records) != len(plan['prepared']['rows']): raise ValueError('cohort denominator changed')
    inputs = {}; tables = {}; history = {}; calibration_checks = []
    for generation in generations:
        mp, np = Path(generation['measured_path']), Path(generation['neural_path'])
        if file_hash(mp) != generation['measured_sha256'] or file_hash(np) != generation['neural_sha256']:
            raise ValueError('calibration generation bytes changed')
        tables[generation['generation']] = (load(mp), load(np))
        oldest,newest=observation_range(*tables[generation['generation']])
        if (generation['oldest_observation_unix']!=oldest or generation['newest_observation_unix']!=newest
                or newest>generation['registered_unix'] or
                (generation['refresh_started_unix'] is not None and generation['refresh_started_unix']>oldest)
                or file_hash(generation['manifest_path'])!=generation['manifest_sha256']):
            raise ValueError('generation clock or original calibration manifest changed')
        inputs.update({str(mp): file_hash(mp), str(np): file_hash(np)})
        check = calibration_audit(mp, np, generation['manifest_path'], gpu_uuid=generation['gpu_uuid'])
        inputs.update(check['input_sha256']); calibration_checks.append(check)
    for row, record in zip(plan['prepared']['rows'], records):
        if any(record.get(k) != v for k, v in row.items()): raise ValueError('frozen row/seed/history scope changed')
        if record['status'] != 'completed': raise ValueError('full successful screen/development coverage required')
        output = root / record['output']; path = Path(record['manifest']); manifest = load(path)
        if file_hash(path) != record['manifest_sha256']: raise ValueError('sealed episode manifest changed')
        snapshot = source_snapshot(manifest, path, plan['source_commit'])
        if snapshot != record['source_snapshot']: raise ValueError('episode source changed')
        measured, neural = tables[record['calibration_generation']]
        check = freshness(measured, neural, now=record['freshness']['checked_unix'],
                          remaining_episodes=1, episode_seconds=3600)
        if check != record['freshness'] or not check['fresh']:
            raise ValueError('full-bound calibration freshness not proven')
        audit = audit_episode(output, row, manifest, snapshot,
            measured if manifest['profile_source'] == SOURCE else neural, development=plan['stage'] == 'development')
        if not audit['valid'] or audit != record['audit']: raise ValueError('independent raw receipt replay mismatch')
        result=load(output/'result.json')
        if not check['checked_unix']<=result['episode_started_unix']<=max(
                r['feedback_ready_unix'] for r in result['receipts'].values())<check['expires_unix']-300:
            raise ValueError('actual receipt interval exceeds its admitted freshness lease')
        if row['arm'] == 'v5_1_1':
            result = load(output / 'result.json'); state = load(output / 'alpha-state.json')
            if state['namespace'] != row['history_namespace'] or state['episode'] != row['id']:
                raise ValueError('trace state leaked across episodes/namespaces')
            history.setdefault(row['history_namespace'], {})[row['id']] = {'result': result, 'offset': state['wall_offset']}
        inputs[str(path)] = file_hash(path)
        for name in ('result.json', 'alpha-state.json', 'validity-audits.json'):
            p = output / name
            if p.exists(): inputs[str(p)] = file_hash(p)
        for p in output.rglob('*'):
            if p.is_file() and p.suffix in {'.json','.jsonl'}:
                if p.is_symlink():raise ValueError('symlink in raw episode evidence')
                inputs[str(p)]=file_hash(p)
    histories = [history_audit(root / 'history' / ns, ns, episodes) for ns, episodes in history.items()]
    for ns in history:
        for name in ('experience.jsonl','index.json'):
            p=root/'history'/ns/name;inputs[str(p)]=file_hash(p)
    if len(histories) != 4: raise ValueError('four independent cold-start history traces required')
    if coverage(records) != load(root / 'coverage.json'): raise ValueError('coverage replay mismatch')
    return {'schema': 'v511_saved_evidence_replay_v1', 'valid': True,
        'coverage': coverage(records), 'history': histories, 'input_sha256': inputs,
        'calibration_generations': len(calibration_checks),
        'calibration_receipts': sum(c['receipts'] for c in calibration_checks),
        'scope': 'same-path independent receipt/history and every dedicated calibration generation replay'}


def main(argv=None):
    p = argparse.ArgumentParser(__doc__); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args(argv)
    if a.output.exists(): raise FileExistsError('independent replay output already exists')
    result = replay_campaign(a.root); atomic_json(a.output, result)
    print(json.dumps({'valid': result['valid'], 'coverage': result['coverage']}))


if __name__ == '__main__': main()
