"""Jointly choose ready jobs and disjoint CPU reservations using measured co-runs.

GPU interference changes predicted service rates. CPU/GPU complementarity is a
bounded tie-break inside the forecast tolerance, never permission to ignore a
measured slowdown. No future arrivals, preemption or invented GPU shares.
"""
from itertools import combinations,product
import math,time

def positive(x):return type(x) in (int,float) and math.isfinite(x) and x>0

def partitions(ids,total=6,minimum=2):
 ids=tuple(ids)
 if not ids:return []
 return [dict(zip(ids,counts)) for counts in product(range(minimum,total+1),repeat=len(ids)) if sum(counts)==total]

def allocate_counts(pool,current,counts):
 pool=sorted(set(pool));keys=list(current)+[k for k in counts if k not in current]
 if set(current)-set(counts) or set(keys)!=set(counts) or any(type(c) is not int or c<2 for c in counts.values()) or sum(counts.values())!=len(pool):raise ValueError('invalid target reservations')
 seen=set()
 for cores in current.values():
  if not set(cores)<=set(pool) or seen&set(cores):raise ValueError('invalid prior ownership')
  seen.update(cores)
 result={k:sorted(current.get(k,[]))[:counts[k]] for k in keys}
 free=sorted(set(pool)-{c for cs in result.values() for c in cs})
 for k in keys:
  n=counts[k]-len(result[k]);result[k]+=free[:n];del free[:n];result[k].sort()
 assert not free
 return result

class JointCosts:
 def __init__(self,reports):
  self.profiles={};self.groups={};self.signatures={}
  for key,r in reports.items():
   if r.get('status')!='estimated':continue
   self.signatures[key]=r['source_sha256'];self.profiles[key]={int(c):v for c,v in r['cpu_profiles'].items()}
  reverse={s:k for k,s in self.signatures.items()}
  for r in reports.values():
   for group in r.get('co_runs',[]):
    if not all(m['source_sha256'] in reverse for m in group['members']):continue
    pair=tuple(sorted((reverse[m['source_sha256']],m['cpu_count']) for m in group['members']))
    self.groups[pair]={reverse[m['source_sha256']]:{'point':m['overlap_factor'],'upper':m['upper_factor']} for m in group['members']}
 def memory(self,key):return max((v['planning_memory_mib'] for v in self.profiles.get(key,{}).values()),default=math.inf)
 def rates(self,counts,scenario='point'):
  if not counts:return {}
  if any(c not in self.profiles.get(k,{}) for k,c in counts.items()):return None
  group=self.groups.get(tuple(sorted(counts.items()))) if len(counts)>1 else None
  if len(counts)>1 and group is None:return None
  return {k:1/(self.profiles[k][c]['runtime_seconds']*(group[k][scenario] if group else 1.)) for k,c in counts.items()}
 def complementarity(self,counts):
  demands=[]
  for k,c in counts.items():
   p=self.profiles.get(k,{}).get(c,{})
   cpu=p.get('cpu_pressure_lower_bound');gpu=p.get('gpu_busy_fraction')
   if not all(type(x) in (int,float) and math.isfinite(x) and 0<=x<=1 for x in (cpu,gpu)):return None
   demands.append((cpu,gpu))
  if len(demands)<2:return 0.
  scores=[max(0.,a-b)*max(0.,d-c)+max(0.,b-a)*max(0.,c-d) for (a,b),(c,d) in combinations(demands,2)]
  return sum(scores)/len(scores)

class WorkClock:
 def __init__(self):self.remaining={};self.last=None;self.overruns=set()
 def advance(self,now,counts,costs):
  if self.last is not None and now<self.last:raise ValueError('clock moved backwards')
  dt=0 if self.last is None else now-self.last;self.last=now
  rates=costs.rates(counts)
  if counts and rates is None:raise ValueError('actual allocation has no matched interference evidence')
  for k in counts:
   value=self.remaining[k]-dt*rates[k]
   if value<.03:self.overruns.add(k)
   self.remaining[k]=max(.03,value)
  for k in list(self.remaining):
   if k not in counts:del self.remaining[k]
 def launched(self,key):self.remaining[key]=1.

def _forecast(ready,remaining,costs,first,first_counts,capacity,external,scenario,deadline):
 memo={};visits=0
 def visit(waiting,work,initial=False):
  nonlocal visits
  visits+=1
  if time.perf_counter()>deadline:raise TimeoutError('planner budget')
  if not waiting and not work:return (0.,{})
  key=(waiting,tuple(sorted((k,round(v,6)) for k,v in work.items())))
  if not initial and key in memo:return memo[key]
  choices=[tuple(first)] if initial else (c for n in range(min(len(waiting),max(0,3-len(work)))+1) for c in combinations(waiting,n))
  best=None
  for add in choices:
   active={**work,**{k:1. for k in add}}
   if not active or sum(costs.memory(k) for k in active)+external>capacity:continue
   plans=[first_counts] if initial else partitions(active)
   for counts in plans:
    if time.perf_counter()>deadline:raise TimeoutError('planner budget')
    rates=costs.rates(counts,scenario)
    if rates is None or set(counts)!=set(active):continue
    dt=min(active[k]/rates[k] for k in active)
    left={k:max(0.,v-dt*rates[k]) for k,v in active.items()}
    finished=[k for k,v in left.items() if v<1e-6]
    after={k:v for k,v in left.items() if k not in finished}
    tail=visit(tuple(k for k in waiting if k not in add),after)
    if tail is None:continue
    finish={k:dt for k in finished};finish.update({k:dt+v for k,v in tail[1].items()})
    value=(dt+tail[0],finish)
    if best is None or value[0]<best[0]:best=value
  if not initial:memo[key]=best
  return best
 result=visit(tuple(ready),dict(remaining),True)
 return result,visits

def choose_joint(ready,remaining,current_counts,costs,*,live_used_mib,capacity_mib,budget_ms=80.,delay_penalty=.25,min_gain=.01):
 start=time.perf_counter();deadline=start+budget_ms/1000
 ids=list(ready)+list(remaining)
 if len(set(ids))!=len(ids) or set(remaining)!=set(current_counts) or any(not positive(v) for v in remaining.values()):raise ValueError('invalid resident/ready state')
 if not positive(capacity_mib) or type(live_used_mib) not in (int,float) or not math.isfinite(live_used_mib) or live_used_mib<0:raise ValueError('invalid memory state')
 base=dict(admit=[],cpu_counts=dict(current_counts),reason='wait',future_arrivals_used=False,forecasts=[],predicted_phase_known=False,units='normalized_whole_program_work',complementarity_role='tie_break_within_0.5_seconds',gpu_share_control=False,resident_delay_penalty=delay_penalty)
 def fallback(reason):
  if not remaining and ready and costs.rates({ready[0]:6}) is not None and costs.memory(ready[0])+live_used_mib<=capacity_mib:
   return {**base,'admit':[ready[0]],'cpu_counts':{ready[0]:6},'reason':reason+'_exclusive_head'}
  plans=[p for p in partitions(remaining) if costs.rates(p) is not None]
  counts=current_counts if sum(current_counts.values())==6 or not plans else min(plans,key=lambda p:max(remaining[k]/costs.rates(p)[k] for k in p))
  return {**base,'cpu_counts':dict(counts),'reason':reason}
 if not ready and not remaining:return {**base,'reason':'idle'}
 if live_used_mib>=capacity_mib:return fallback('live_memory_gate')
 if any(k not in costs.profiles for k in ids):return {**base,'reason':'missing_profile_exclusive_fallback_required'}
 external=max(0.,live_used_mib-sum(costs.memory(k) for k in remaining))
 anchor=() if remaining else (ready[0],)
 options=[]
 try:
  for n in range(min(len(ready),max(0,3-len(remaining)))+1):
   for add in combinations(ready,n):
    keys=[*remaining,*add]
    if not keys:continue
    for counts in partitions(keys):
     point,visits=_forecast(ready,remaining,costs,add,counts,capacity_mib,external,'point',deadline)
     if point is None:continue
     upper,_=_forecast(ready,remaining,costs,add,counts,capacity_mib,external,'upper',deadline)
     if upper is None:continue
     options.append(dict(admit=list(add),cpu_counts=counts,point_seconds=point[0],upper_scenario_seconds=upper[0],finish=point[1],complementarity=costs.complementarity(counts),events=visits))
 except TimeoutError:return fallback('planner_time_budget')
 refs=[o for o in options if o['admit']==list(anchor)]
 if not refs:return fallback('no_evidenced_feasible_reference')
 reference=min(refs,key=lambda o:o['point_seconds'])
 for o in options:
  delay=sum(max(0.,o['finish'][k]-reference['finish'][k]) for k in remaining)
  o.update(added_resident_delay=delay,objective=o['point_seconds']+delay_penalty*delay)
 minimum=min(o['objective'] for o in options)
 # Complementarity can break a genuinely near-equal forecast, not force a bad pair.
 best=min(options,key=lambda o:o['objective'])
 # Missing duty measurements are unknown, not evidence of poor complementarity.
 if best['complementarity'] is not None:
  tied=[o for o in options if o['objective']<=minimum+.5 and o['complementarity'] is not None]
  best=max(tied,key=lambda o:(o['complementarity'],-o['objective']))
 threshold=max(.5,reference['point_seconds']*min_gain)
 gain=reference['point_seconds']-best['point_seconds'];upper_gain=reference['upper_scenario_seconds']-best['upper_scenario_seconds']
 if best['admit']!=list(anchor) and not (gain>=threshold and upper_gain>=threshold and reference['objective']-best['objective']>=threshold):best=reference
 return {**base,'admit':best['admit'],'cpu_counts':best['cpu_counts'],'reason':'joint_predicted_gain' if best['admit']!=list(anchor) else ('joint_hold_or_reallocate' if remaining else 'joint_exclusive_head'),'forecasts':options,'selected':best,'reference':reference,'elapsed_ms':1000*(time.perf_counter()-start),'upper_scenario_is_statistical_bound':False}
