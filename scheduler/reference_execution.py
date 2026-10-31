"""Executable continuation controller for the opt-in continuation ablation.

Compose ReferenceExecutionMixin before BracketBootstrapMixin. The root scorer
calls _v5_reference_begin(cohort, result, trial) only after an authoritative V5
score. Shadow mode never activates execution. This controller uses the forecast's
reference_continuation selector and the same _v5_placement_allowed predicate;
normal _dispatch_plan remains the execution transaction boundary.

No future measurement is assumed. Exact current group evidence can authorize a
packed refill; unknown/expired/restarted groups drain to ordinary solo service.
A restart preserves FIFO/reservations but invalidates every timing cache. New
arrivals or material identity changes explicitly return to online planning.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import time

from ..domain import BatchResolution, JobStatus, build_colocation_profile_key
from .cold_start import cold_group_blocked
from .continuation_forecast import reference_continuation
from .deferral_reservations import KEY, protected
from .feedback_objective import timestamp
from .planner_types import DispatchPlan

SCHEMA = 'v5_reference_execution_v1'
_ERRORS = (ValueError, TypeError, KeyError, OverflowError, AttributeError)


def _number(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
        raise ValueError('finite nonnegative resource/evidence value required')
    return float(value)


def _material(job, profile, backend):
    return {'submitted_at':job.submitted_at,'signature':job.packing.signature or job.job_id,
        'batch':BatchResolution.resolved_batch_size(job),'profile':deepcopy(profile),
        'recipe':deepcopy(job.metadata.get('perfseer_model')),
        'workload_sha256':job.metadata.get('pilot_frozen_workload_sha256'),
        'request':deepcopy(job.metadata.get('feedback_request')),
        'precision':job.metadata.get('precision_mode'),'backend':backend,
        'backend_config':_backend_config(job,backend)}


def _backend_config(job,backend):
    if backend!='mps_process':return deepcopy(job.metadata.get('placement_backend_config') or {})
    config=job.metadata.get('placement_backend_config') or {}
    allocations=config.get('allocation_percentages') or [100]
    return {'allocation_percentages':[max(1,min(100,int(job.metadata.get('mps_active_thread_pct',allocations[0]))))]}


class ReferenceExecutionMixin:
    _v5_reference_state = None
    _v5_reference_restored = False

    def _v5_reference_enabled(self):
        options=getattr(self,'v5_options',None)
        return getattr(options,'continuation',False)is True or getattr(options,'bracket',False)is True

    def _v5_reference_path(self):
        return Path(self.settings.runtime_root)/'v5-reference-state.json'

    def _v5_reference_emit(self,event,**payload):
        self.event_logger.emit(event,payload={'controller':'same_forecast_reference_fifo',**payload})

    def _v5_reference_save(self):
        state=self._v5_reference_state
        path=self._v5_reference_path();path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state or {'schema':SCHEMA,'status':'closed'},sort_keys=True,allow_nan=False)+'\n')
        temporary.replace(path)

    def _v5_reference_restore(self):
        if self._v5_reference_restored:return
        self._v5_reference_restored=True
        path=self._v5_reference_path()
        if not path.exists():return
        try:
            value=json.loads(path.read_text())
            if value.get('status')=='closed':return
            if (value.get('schema')!=SCHEMA or not isinstance(value.get('cohort'),list)
                    or len(set(value['cohort']))!=len(value['cohort'])
                    or set(value.get('order',[]))!=set(value['cohort'])
                    or set(value.get('identities',{}))!=set(value['cohort'])):
                raise ValueError('invalid persisted reference cohort/order')
            self._v5_reference_state=value
            value['reconciliation_deadline_at']=_number(value.get('reconciliation_deadline_at',
                _number(value['begun_at'])+self.settings.gpu_scheduler.colocation.cold_start_prepare_timeout_seconds))
            value.update(status='solo_drain_restored',prediction_complete=False,
                         restore_reason='process generation and timing caches invalid after restart')
            getattr(self,'_v5_rate_cache',{}).clear()
            self._v5_reference_save()
            self._v5_reference_emit('v5_reference_restored',cohort=value['cohort'],
                reservations=value.get('reservations',[]),action='drain_incumbents_then_reference_solo')
        except (OSError,json.JSONDecodeError,*_ERRORS) as exc:
            self._v5_reference_state=None
            self._v5_reference_emit('v5_reference_restore_invalid',reason=str(exc),action='fresh_online_planning')

    def _v5_reference_protected(self,jobs):
        by_id={j.job_id:j for j in jobs};ids=[]
        if getattr(getattr(self,'v5_options',None),'deferral',False):
            seconds=getattr(self.v5_options,'deferral_seconds',30.)
            bypasses=getattr(self.v5_options,'max_younger_bypasses',2)
            ids=[j.job_id for j in jobs if j.metadata.get(KEY)
                 and protected(j.metadata[KEY],seconds=seconds,bypasses=bypasses)]
        state=self._v5_reference_state or {}
        ids.extend(j for j in state.get('reservations',[]) if j in by_id)
        return sorted(set(ids),key=lambda j:(timestamp(by_id[j].submitted_at),j))

    def _v5_reference_begin(self,cohort,result,trial):
        # C-only enables the bootstrap controller, not A's add-only scorer.
        if getattr(getattr(self,'v5_options',None),'continuation',False)is not True:return False
        members=tuple(sorted({*getattr(trial,'preexisting_job_ids',()),
            *([trial.candidate_job_id] if getattr(trial,'candidate_job_id',None) else ())}))
        cleanup=getattr(trial,'cleanup_deadline_at',None)
        if cleanup is None:cleanup=(getattr(trial,'cold_start',None)or{}).get('cleanup_deadline_at')
        return self._v5_reference_install(cohort,result,trial_id=getattr(trial,'trial_id',None),
            members=members,cleanup=cleanup,origin='add_only_admission')

    def _v5_reference_begin_bootstrap(self,cohort,result,state):
        """Install the full-group continuation without newcomer/pre-add fiction."""
        if getattr(getattr(self,'v5_options',None),'bracket',False)is not True:return False
        jobs=list(cohort);members=tuple(sorted(state.job_ids));by_id={j.job_id:j for j in jobs}
        if not members or not set(members)<=set(by_id):
            self._v5_reference_emit('v5_reference_not_started',origin='group_bootstrap',
                reason='bootstrap group absent from complete receipt-aware cohort',action='bounded_bootstrap_solo_cleanup')
            return False
        from .experimental_v5 import update_state
        cleanup_members=tuple(j for j in members if not by_id[j].status.is_terminal
            and update_state(by_id[j])[1]!=0)if result.get('decision')!='accepted'else ()
        return self._v5_reference_install(jobs,result,trial_id=state.trial_id,members=members,
            cleanup=state.cleanup_deadline_at,origin='group_bootstrap',cleanup_members=cleanup_members,
            bootstrap_generation=state.generation)

    def _v5_reference_install(self,cohort,result,*,trial_id,members,cleanup,origin,
                              cleanup_members=(),bootstrap_generation=None):
        if not self._v5_reference_enabled():return False
        if result.get('decision')not in {'accepted','rejected','inconclusive'}:return False
        self._v5_reference_restore()
        jobs=list(cohort);ids=[j.job_id for j in jobs]
        if not ids or len(set(ids))!=len(ids):
            self._v5_reference_emit('v5_reference_not_started',reason='complete unique cohort required')
            return False
        if 'remaining_updates'in result and set(result['remaining_updates'])!=set(ids):
            self._v5_reference_emit('v5_reference_not_started',reason='scored cohort differs from concrete reference cohort')
            return False
        backend=self.settings.gpu_scheduler.packing_backend
        profiles=deepcopy(result.get('frozen_profiles') or {})
        try:
            for job in jobs:
                if job.job_id not in profiles:
                    profiles[job.job_id]=deepcopy(self.planner.estimator.predicted_profile(job))
                profile=profiles[job.job_id]
                if (not isinstance(profile,dict)or profile.get('kind')!='predicted_workload'
                        or profile.get('job_signature')!=(job.packing.signature or job.job_id)
                        or profile.get('batch_size')!=BatchResolution.resolved_batch_size(job)
                        or not profile.get('artifact_sha256')or not profile.get('hardware')
                        or _number(profile['step_cost'])<=0 or _number(profile['peak_vram_mib'])<=0):
                    raise ValueError('missing or incompatible frozen profile: '+job.job_id)
            if set(profiles)!=set(ids)or len({(p['artifact_sha256'],p['hardware'])for p in profiles.values()})!=1:
                raise ValueError('complete cohort needs one compatible frozen profile scale')
            identities={j.job_id:_material(j,profiles[j.job_id],backend)for j in jobs}
        except (ValueError,TypeError,KeyError,AttributeError,RuntimeError) as exc:
            self._v5_reference_emit('v5_reference_not_started',origin=origin,reason=str(exc),
                action='bounded_bootstrap_solo_cleanup'if origin=='group_bootstrap'else'safe_online_planning')
            return False
        halves=result.get('halves')or[]
        complete=(result.get('error')is None and len(halves)==2 and all(
            all(h.get(a+'_trace',{}).get('status')=='complete' and
                set(ids)<=set(h.get(a,{}).get('job_completions',{})) for a in ('admit','defer')) for h in halves))
        decision=result['decision'];complete=complete and decision!='inconclusive'
        action='admit' if decision=='accepted' else 'defer'
        now=time.time()
        reconciliation=now+_number(self.settings.gpu_scheduler.colocation.cold_start_prepare_timeout_seconds)
        if cleanup is not None:
            reconciliation=_number(cleanup)if origin=='group_bootstrap'else min(reconciliation,_number(cleanup))
        previous=self._v5_reference_state
        repeated=previous is not None and previous.get('trial_id')==trial_id
        if repeated:reconciliation=min(reconciliation,_number(previous['reconciliation_deadline_at']))
        state={'schema':SCHEMA,'status':('solo_drain_bootstrap'if origin=='group_bootstrap'and action=='defer'
                else'active'if complete else'solo_drain_inconclusive'),
            'cohort':ids,'order':sorted(ids,key=lambda j:(timestamp(next(k for k in jobs if k.job_id==j).submitted_at),j)),
            'identities':identities,
            'profiles':profiles,'hardware_key':self.store.hardware_key(),'backend':backend,
            'chosen_action':action,'score_decision':decision,'prediction_complete':complete,
            'score_error':result.get('error'),'trial_id':trial_id,
            'origin':origin,'group_members':list(members),'cleanup_members':list(cleanup_members),
            'bootstrap_generation':bootstrap_generation,
            'execution_concurrency_limit':1 if origin=='group_bootstrap'and action=='defer'else None,
            'avoid_groups':[list(members)] if action=='defer' and len(members)>1 else [],
            'predicate_context':deepcopy(result.get('context')),'begun_at':now,
            'reconciliation_deadline_at':reconciliation,
            'known_launch_ids':list(self._supervisor_active_job_ids()),'inflight_dispatch':[],
            'reservations':self._v5_reference_protected(jobs),'dispatches':[]}
        if repeated:
            state['begun_at']=previous['begun_at']
            state['known_launch_ids']=sorted(set(state['known_launch_ids'])|set(previous.get('known_launch_ids',[])))
            state['inflight_dispatch']=previous.get('inflight_dispatch',[])
            state['dispatches']=previous.get('dispatches',[])
        self._v5_reference_state=state;self._v5_reference_save()
        self._v5_reference_emit('v5_reference_started',cohort=ids,order=state['order'],
            origin=origin,group_members=list(members),cleanup_members=list(cleanup_members),
            chosen_action=action,score_decision=decision,prediction_complete=complete,
            reason='authoritative_forecast' if complete else 'explicit_unknown_reference_solo_fallback')
        return True

    def _v5_reference_invalidate(self,reason):
        state=self._v5_reference_state
        self._v5_reference_emit('v5_reference_invalidated',reason=reason,
            cohort=state.get('cohort',[])if state else [],action='fresh_online_planning',
            difference='new information; frozen continuation no longer predicts this online path')
        self._v5_reference_state=None;self._v5_reference_save()

    def _v5_reference_context(self):
        state=self._v5_reference_state
        if state:
            return self._v5_snapshot_context(state['profiles'])
        from .v5_lifecycle import forecast_cohort
        from .feedback_trials import profile_snapshot
        cohort,_,_=forecast_cohort(self)
        return self._v5_snapshot_context(profile_snapshot(self,cohort))

    def _v5_reference_memory(self,job):
        backend=self.settings.gpu_scheduler.packing_backend
        estimated=_number(self.planner.estimator.estimate_peak_vram_mb(job,BatchResolution.resolved_batch_size(job),backend))
        profile=self.planner.estimator.predicted_profile(job)or{}
        predicted=_number(profile.get('peak_vram_mib',0.))
        return max(estimated,predicted)

    def _v5_reference_group_key(self,jobs):
        backend=self.settings.gpu_scheduler.packing_backend
        descriptors=[self.planner.time_objective.member_descriptor(job,backend_name=backend,
            batch_size=BatchResolution.resolved_batch_size(job),backend_config=_backend_config(job,backend)) for job in jobs]
        return build_colocation_profile_key(self.store.hardware_key(),descriptors)

    def _v5_placement_allowed(self,group,*,context=None):
        """Shared forecast/execution predicate; no cached timing is a trial permit.

        Root may pass its frozen complete-cohort ``context`` explicitly during
        scoring. Otherwise construct current context with the same root helper.
        Only both clean halves with exact identity/generation/actual observed
        time are accepted. Every failed/retry-prohibited group stays prohibited.
        """
        self._v5_reference_last_block=None
        def block(reason):
            self._v5_reference_last_block=reason
            return False
        try:
            ids=tuple(sorted(group))
            if not ids or len(set(ids))!=len(ids):return block('ambiguous_group')
            jobs=[self.store.get_job(j)for j in ids]
            if any(j is None or j.hold for j in jobs):return block('missing_or_held_member')
            backend=self.settings.gpu_scheduler.packing_backend;gpu=self.settings.gpu_scheduler
            cap=gpu.parallel_job_cap
            if cap is not None and len(ids)>cap:return block('configured_concurrency_cap')
            if not gpu.enabled and len(ids)>1:return block('concurrency_disabled')
            if not self.planner.compatibility.compatible_group(jobs,backend_name=backend):return block('compatibility')
            memory=[self._v5_reference_memory(j) for j in jobs]
            if any(v<=0 for v in memory)or sum(memory)>_number(self.planner.estimator.safe_budget_mb()):return block('admission_memory')
            if len(ids)==1:return True
            # Use the same optional protected-placement authorization as the
            # physical DeferralReservationMixin transaction; sorting alone is
            # insufficient when a protected job's current contract is solo.
            waiting=self.store.runnable_jobs()
            reservations=self._v5_reference_protected(waiting)
            if reservations:
                reserved=self.store.get_job(reservations[0])
                hook=getattr(self,'_v5_deferral_legal_placement',None)
                proposed=DispatchPlan(mode='concurrent_group',backend_name=backend,job_ids=ids,reason='shared reference predicate')
                if (reserved.job_id not in ids or reserved.metadata.get('cold_start_solo_required')
                        or hook is None or not hook(reserved,proposed)):
                    return block('protected_service_placement')
            state=self._v5_reference_state
            if state and (state['status']!='active' or list(ids)in state.get('avoid_groups',[])):
                return block('reference_solo_drain_or_rejected_pack')
            if any(not self.planner.compatibility.pack_eligible(j,backend_name=backend)for j in jobs):return block('packing_ineligible')
            key=self._v5_reference_group_key(jobs);now=time.time()
            for job in jobs:
                attempt=(job.metadata.get('cold_start_group_attempts')or{}).get(key)or{}
                _number(job.metadata.get('cold_start_retry_after',0.));_number(attempt.get('retry_after',0.))
                if cold_group_blocked(job,key,gpu.colocation,now)or int(attempt.get('failures',0))>0:
                    return block('failed_group_or_retry_restriction')
            record=getattr(self,'_v5_rate_cache',{}).get(ids)
            if not isinstance(record,dict):return block('unknown_exact_group')
            context=self._v5_reference_context()if context is None else context
            cached=record.get('context')or{}
            for k in ('evidence_identity','membership_generation'):
                if k not in context or cached.get(k)!=context[k]:return block('evidence_context_or_process_generation')
            halves=record.get('halves')or[]
            if len(halves)!=2:return block('two_clean_halves_required')
            ttl=_number(self.v5_options.evidence_ttl_seconds)
            for half in halves:
                if (half.get('identity')!=context['evidence_identity']or half.get('membership_generation')!=context['membership_generation']
                        or set(half.get('rates',{}))!=set(ids)or half.get('valid',True)is not True):return block('exact_half_identity')
                age=now-_number(half['observed_at'])
                if not 0<=age<=min(ttl,_number(half['max_age_seconds'])):return block('stale_actual_measurement')
                if any(_number(v)<=0 for v in half['rates'].values()):return block('invalid_rate')
            return True
        except _ERRORS as exc:
            return block('unknown_placement_identity:'+str(exc))

    def _v5_reference_check(self):
        state=self._v5_reference_state;all_jobs=self.store.list_jobs();by_id={j.job_id:j for j in all_jobs}
        pending={j.job_id for j in all_jobs if not j.status.is_terminal or
            (j.status==JobStatus.COMPLETED and not j.metadata.get('feedback_ready_at'))}
        if set(self._supervisor_active_job_ids())-set(state['cohort']):return 'unmodeled_active_worker'
        if pending-set(state['cohort']):return 'new_submitted_outstanding_arrival'
        if self.store.hardware_key()!=state['hardware_key']or self.settings.gpu_scheduler.packing_backend!=state['backend']:
            return 'hardware_or_backend_changed'
        for name in state['cohort']:
            job=by_id.get(name)
            if job is None:return 'cohort_job_missing'
            if job.status.is_terminal and job.status!=JobStatus.COMPLETED:return 'required_job_failed_or_cancelled'
            if job.status==JobStatus.COMPLETED and job.metadata.get('feedback_ready_at'):continue
            profile=self.planner.estimator.predicted_profile(job)
            if _material(job,profile,state['backend'])!=state['identities'][name]:return 'material_recipe_profile_request_or_submission_changed'
        return None

    def _v5_reference_wait_release(self,pending,reason):
        state=self._v5_reference_state
        deadline=_number(state['reconciliation_deadline_at'])
        if time.time()>=deadline:
            failure={'kind':'reference_cleanup_reconciliation_timeout','pending_job_ids':list(pending),
                'reason':reason,'reconciliation_deadline_at':deadline,'trial_id':state.get('trial_id')}
            self.v5_failure=failure
            state['failure']=failure;self._v5_reference_save()
            self._v5_reference_emit('v5_reference_cleanup_timeout',**failure)
            raise TimeoutError('reference cleanup/ownership reconciliation exceeded original finite deadline')
        self._v5_reference_emit('v5_reference_waiting_release',pending_job_ids=list(pending),
            reason=reason,reconciliation_deadline_at=deadline)

    def _v5_reference_has_release(self,name):
        observed=getattr(self.supervisor,'observed_exit_code',None)
        return callable(observed)and observed(name)==0

    def _dispatch_pending_work(self):
        if not self._v5_reference_enabled():return super()._dispatch_pending_work()
        self._v5_reference_restore()
        state=self._v5_reference_state
        if state is None:return super()._dispatch_pending_work()
        reason=self._v5_reference_check()
        if reason:
            self._v5_reference_invalidate(reason)
            return super()._dispatch_pending_work()
        by_id={j:self.store.get_job(j)for j in state['cohort']}
        active=tuple(sorted(self._supervisor_active_job_ids()))
        if not active and all(j.status==JobStatus.COMPLETED and j.metadata.get('feedback_ready_at') for j in by_id.values()):
            self._v5_reference_emit('v5_reference_complete',cohort=state['cohort'])
            self._v5_reference_state=None;self._v5_reference_save()
            return super()._dispatch_pending_work()
        # Existing lifecycle owns checkpoint/exit cleanup. Never pause incumbents
        # or dispatch into PAUSING/old-supervisor ownership to realize a forecast.
        if getattr(self,'_colocation_trial',None)is not None:
            self._v5_reference_wait_release(active,'authoritative_trial_cleanup_pending')
            return
        bootstrap=getattr(self,'_bracket_bootstrap',None)
        if bootstrap is not None and getattr(bootstrap,'phase','closed')!='closed':
            pending_hook=getattr(self,'_bracket_cleanup_pending',None)
            pending=pending_hook()if callable(pending_hook)else list(getattr(bootstrap,'live_identities',{}))
            self._v5_reference_wait_release(pending,'owned_bootstrap_cleanup_pending')
            return super()._dispatch_pending_work()
        if time.monotonic()<getattr(self,'_next_gpu_dispatch_attempt_at',0.):return
        if not getattr(self._admission_gate,'is_open',False):return
        pausing=[j for j in active if by_id[j].status==JobStatus.PAUSING]
        if pausing:
            self._v5_reference_wait_release(pausing,'owned_pause_checkpoint_exit_pending')
            return
        known=set(state.get('known_launch_ids',[]))|set(state.get('inflight_dispatch',[]))
        uncertain=[j for j in known if j not in active and not (by_id[j].status==JobStatus.COMPLETED
            and by_id[j].metadata.get('feedback_ready_at'))and not self._v5_reference_has_release(j)]
        if uncertain:
            self._v5_reference_wait_release(uncertain,'absence_from_supervisor_is_not_release_proof')
            return
        if getattr(self,'_exclusive_probe_job_id',None)is not None:
            self._v5_reference_invalidate('exclusive_probe_reservation_requires_online_planning')
            return super()._dispatch_pending_work()
        runnable={j.job_id:j for j in self._runnable_jobs()if j.job_id not in active and j.job_id in by_id}
        if not runnable:return
        queue=[j for j in state['order']if j in runnable]
        jobs=list(runnable.values());reservations=self._v5_reference_protected(jobs)
        state['reservations']=reservations
        backend=state['backend'];gpu=self.settings.gpu_scheduler
        if not self.supervisor.available_backends().get(backend,False):return
        try:
            budget=_number(self.planner.estimator.safe_budget_mb())
            memory={j:self._v5_reference_memory(by_id[j]) for j in (*active,*queue)}
            used=_number(self._active_vram_occupancy())
            # Live device/process reservation is an additional admission floor;
            # the selector charges any excess beyond predicted resident memory.
            effective=max(0.,budget-max(0.,used-sum(memory[j]for j in active)))
            blocks=[]
            def allowed(group):
                valid=self._v5_placement_allowed(group)
                if not valid:blocks.append({'group':list(group),'reason':self._v5_reference_last_block})
                return valid
            choice=reference_continuation(queue,active,memory=memory,memory_capacity=effective,
                concurrency_limit=state.get('execution_concurrency_limit')or(gpu.parallel_job_cap if gpu.enabled else 1),
                protected=reservations,compatible=lambda group:self.planner.compatibility.compatible_group(
                    [by_id[j]for j in group],backend_name=backend),placement_allowed=allowed)
            self._v5_reference_emit('v5_reference_selection',cohort=state['cohort'],queue=queue,
                residents=list(active),selected=list(choice['selected']),reservations=reservations,
                blocked=choice['blocked'],predicate_blocks=blocks,safe_budget_mib=budget,
                active_memory_mib=used,effective_budget_mib=effective)
            self._v5_reference_save()
            selected=choice['selected']
            if not selected:return
            # Select once, dispatch as one known feasible group. Starting workers
            # changes process generation; never recursively reuse old group rates.
            configs=[_backend_config(by_id[j],backend)for j in selected]
            if any(c!=configs[0]for c in configs):return
            plan=DispatchPlan(mode='concurrent_group'if active or len(selected)>1 else 'stack_anchor',
                backend_name=backend,job_ids=selected,reason='authoritative V5 executable reference continuation',
                backend_config=configs[0],trial_metadata={'cooperative_trial':True,'requires_live_trial':False,
                    'reference_execution':True,'memory_limit_mb':max(0.,budget-used)},
                objective_breakdown={'reference_execution':True,'chosen_action':state['chosen_action'],
                    'prediction_complete':state['prediction_complete'],'preexisting_job_ids':list(active),
                    'reservations':reservations},objective_version=gpu.objective.objective_version)
            state['inflight_dispatch']=list(selected);self._v5_reference_save()
            if self._dispatch_plan(plan):
                state['known_launch_ids']=sorted(set(state.get('known_launch_ids',[]))|set(selected))
                state['inflight_dispatch']=[]
                state['dispatches'].append({'job_ids':list(selected),'at':time.time(),'resident_ids':list(active)})
                self._v5_reference_save()
            else:
                state['inflight_dispatch']=[];self._v5_reference_save()
        except _ERRORS as exc:
            self._v5_reference_emit('v5_reference_blocked',reason='unknown_resource_or_placement',error=str(exc))
