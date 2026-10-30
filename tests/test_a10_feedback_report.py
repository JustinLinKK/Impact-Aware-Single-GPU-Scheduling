"""CPU synthetic artifact tests; no GPU, prediction, or training imports."""
import copy
import json
from pathlib import Path
import pytest
from localml_scheduler.benchmarks.a10_feedback_latency import report


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def receipt(job='j', jct=10):
    return {'job_id':job,'worker_exit_code':0,'jct_seconds':jct,'expected_updates':64,'expected_epochs':2,
        'expected_runtime_signature':'a'*64,'auc_maximize':True,'selected_auc':.7,'selected_validation_bce':.6,
        'returned_result':{'global_step':64,'epochs':2,'signature':'a'*64}}


def cohort(arm, mode, value=10):
    return {'complete':True,'arm':arm,'mode':mode,'predictor_comparison_eligible':True,
        'receipts':{str(i):receipt(str(i),value+i) for i in range(6)},
        'request_latencies_seconds':{str(i):value for i in range(6 if mode=='individual' else 2)},
        'episode_duration_seconds':value*2,'chain_root_to_child_seconds':{str(i):value*2 for i in range(3)}}


def campaign(root, missing=None):
    order, rows = [], []
    for r in (1,2):
        for mode in report.MODES:
            for arm in report.ARMS:
                spec={'repeat':r,'mode':mode,'arm':arm}
                order.append(spec)
                if (r,mode,arm)==missing: continue
                name=f'r{r}-{mode}-{arm}'
                result=cohort(arm,mode,10 if arm=='reference_v4' else 8)
                rows.append({**spec,'output':name,'exit_code':0,'result':result})
                write(root/name/'result.json',result)
    write(root/'campaign-protocol-before.json',{'order':order})
    write(root/'campaign-results.json',rows)
    return rows


def test_complete_paired_exports_descriptive_not_winner(tmp_path):
    campaign(tmp_path)
    result=report.build_report(tmp_path)
    assert result['verified_receipts']==96
    assert result['expected_jobs']==96
    assert result['comparison_eligible_cohorts']==16
    pair=next(p for p in result['paired_comparisons'] if p['arm']=='calibrated_revised' and p['mode']=='individual')
    assert pair['paired_mean_delta']['average_job_jct_seconds']==-2
    assert pair['paired_mean_delta']['workflow_duration_seconds']==-4
    assert 'winner' not in result
    text=report.markdown(result)
    assert '96/96' in text and 'A10 student S_A10; not A100 parent P' in text


def test_unmatched_repeat_disables_aggregate_without_dropping_denominator(tmp_path):
    campaign(tmp_path,missing=(2,'individual','calibrated_revised'))
    result=report.build_report(tmp_path)
    assert result['expected_jobs']==96
    assert result['verified_receipts']==90
    pair=next(p for p in result['paired_comparisons'] if p['arm']=='calibrated_revised' and p['mode']=='individual')
    assert pair['pairs'][0]['eligible']
    assert not pair['pairs'][1]['eligible']
    assert pair['paired_mean_delta']['average_job_jct_seconds'] is None
    group=next(p for p in result['arm_mode_summary'] if p['arm']=='calibrated_revised' and p['mode']=='individual')
    assert group['expected_jobs']==12 and group['verified_receipts']==6
    assert group['mean_of_repeat_average_job_jct_seconds'] is None


def test_failed_required_job_remains_and_oom_is_explicit(tmp_path):
    rows=campaign(tmp_path)
    row=rows[0]; out=tmp_path/row['output']
    value=row['result']; value['complete']=False;value['receipts'].pop('5')
    value['error']='CUDA out of memory';value['request_latencies_seconds'].pop('5')
    write(out/'result.json',value)
    write(out/'jobs'/'last'/'job-final.json',{'job_id':'5','status':'FAILED','error':'CUDA out of memory'})
    result=report.build_report(tmp_path)
    rec=result['cohorts'][0]
    assert rec['expected_jobs']==6 and rec['verified_receipts']==5 and rec['unreturned_jobs']==1
    assert rec['job_jct_seconds'] is None
    assert rec['memory_failure_job_ids']==['5']
    assert rec['observed_successful_job_jct_seconds']['n']==5
    assert result['expected_jobs']==96 and result['verified_receipts']==95


def test_predictor_ineligible_repeat_never_compared(tmp_path):
    rows=campaign(tmp_path)
    row=next(r for r in rows if r['repeat']==2 and r['mode']=='group' and r['arm']=='calibrated_v4')
    row['result']['predictor_comparison_eligible']=False
    write(tmp_path/row['output']/'result.json',row['result'])
    result=report.build_report(tmp_path)
    pair=next(p for p in result['paired_comparisons'] if p['mode']=='group' and p['arm']=='calibrated_v4')
    assert pair['paired_mean_delta']['average_job_jct_seconds'] is None
    assert result['verified_receipts']==96


def calibration_row():
    return {'name':'b0','role':'calibration','complete':True,'runtime_completion_identity':'a'*64,
        'receipt':receipt(),'predicted_profile':{'hardware':'A10','step_cost':100,'peak_vram_mib':1000},
        'isolated_seconds_per_update':.2,'peak_reserved_mib':2000,
        'windows':[{'kind':'calibration','complete':True,'excludes_nonupdate_phases':True,'steps':8,'elapsed_seconds':1.6}]}


def test_calibration_rate_error_uses_reciprocal_and_no_full_solo_claim():
    row=report.calibration_evidence([calibration_row()],{'wall_multiplier':2})[0]
    assert row['clean_isolated_window_evidence']
    assert row['predicted_updates_per_second']==10
    assert row['measured_window_updates_per_second']==5
    assert row['window_rate_relative_error']==1 # (10-5)/5, not duration ARE .5
    assert row['window_duration_relative_error']==.5
    assert row['calibrated_window_rate_relative_error']==0
    assert row['full_sustained_solo_updates_per_second'] is None
    assert row['whole_job_update_equivalent_per_second']==6.4
    assert 'allocated' in row['memory_comparison_note']


@pytest.mark.parametrize('change',[{'role':'evaluation'},{'complete':False},
    {'windows':[{'kind':'calibration','complete':True,'excludes_nonupdate_phases':False,'steps':8,'elapsed_seconds':1.6}]}])
def test_colocated_or_incomplete_windows_not_solo_labels(change):
    row=calibration_row();row.update(change)
    output=report.calibration_evidence([row])[0]
    assert not output['clean_isolated_window_evidence']
    assert 'window_rate_relative_error' not in output


def test_invalid_receipt_no_completed_latency(tmp_path):
    rows=campaign(tmp_path)
    row=rows[0];row['result']['receipts']['0']['returned_result']['global_step']=63
    write(tmp_path/row['output']/'result.json',row['result'])
    rec=report.build_report(tmp_path)['cohorts'][0]
    assert not rec['complete'] and rec['invalid_receipts']==1
    assert rec['job_jct_seconds'] is None


def test_cli_empty_campaign_preserves_all_planned_denominators(tmp_path):
    report.main(['--campaign',str(tmp_path),'--output',str(tmp_path/'export')])
    result=json.loads((tmp_path/'export'/'feedback-report.json').read_text())
    assert result['expected_jobs']==96 and result['verified_receipts']==0
    assert (tmp_path/'export'/'feedback-report.md').exists()
    assert all(r['paired_mean_delta']['workflow_duration_seconds'] is None for r in result['paired_comparisons'])


def test_quantile_definition():
    assert report.summary([0,10])=={'n':2,'mean':5,'p95':9.5,'max':10}
    assert report.summary([float('nan')]) is None


def test_failure_snapshot_events_profiles_and_memory_denominator(tmp_path):
    campaign(tmp_path,missing=(2,'individual','calibrated_revised'))
    out=tmp_path/'15-r2-individual-calibrated_revised'
    write(out/'result.json',{'complete':False,'error':'worker failed','elapsed_seconds':60})
    write(out/'failure-jobs.json',[{'job_id':'bad','status':'FAILED','error':'CUDA out of memory'},
                                   {'job_id':'pending','status':'PENDING'}])
    write(out/'failure-events.json',[{'job_id':'bad','event_type':'job_failed','payload':{'error':'CUDA out of memory'}}])
    write(out/'jobs'/'example'/'predicted-profiles.json',{'16':{'step_cost':50,'artifact_sha256':'a'*64}})
    result=report.build_report(tmp_path)
    row=next(r for r in result['cohorts'] if r['repeat']==2 and r['mode']=='individual' and r['arm']=='calibrated_revised')
    assert row['incomplete_observation_seconds']==60
    assert row['memory_failure_job_ids']==['bad']
    assert row['failed_or_cancelled_job_ids']==['bad']
    assert row['unreturned_jobs']==6
    assert row['prepared_numeric_profiles']['example']['16']['step_cost']==50
    assert result['memory_failure_jobs']==1 and result['unreturned_jobs']==6
