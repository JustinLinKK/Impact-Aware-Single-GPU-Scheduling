import unittest
from localml_scheduler.scheduler.ready_set import Costs, choose, actions

class BudgetAdmissionTests(unittest.TestCase):
    def costs(self,guard=True,budget=150):
        reports={}
        for k in ['a','b','c']:
            reports[k]=dict(status='estimated',source_sha256=k,
                cpu_profiles={str(c):dict(runtime_seconds=100,planning_memory_mib=1000,gpu_busy_fraction=.8,cpu_pressure_lower_bound=.3) for c in [2,3,4,6]},
                native_training_budget=dict(status='inferred',source_sha256=k,seconds=budget))
        return Costs(dict(profiles=[],co_runs=[]),reports,budget_guard=guard)

    def decide(self,costs,ready,work=None,policy='lookahead_drain'):
        return choose(ready,work or {},costs,accepted={k:0 for k in ready},now=0,used_mib=500,capacity_mib=15000,policy=policy,max_wait=None,finite_cohort=True)

    def test_memory_fit_can_still_exceed_predicted_training_budget(self):
        c=self.costs();d=self.decide(c,['a','b','c'])
        self.assertEqual(len(d['admit']),1)
        self.assertEqual(len(self.decide(self.costs(False),['a','b','c'],policy='fifo')['admit']),3)

    def test_resident_never_stopped_for_predicted_budget(self):
        d=self.decide(self.costs(),['b','c'],{'a':.5})
        self.assertEqual(d['admit'],[]);self.assertEqual(d['cpu_counts'],{'a':6})

    def test_singleton_not_deadlocked_when_estimate_exceeds_budget(self):
        self.assertEqual(self.decide(self.costs(budget=10),['a'])['admit'],['a'])

    def test_source_mismatch_hint_not_trusted(self):
        c=self.costs();c.reports['a']['native_training_budget']['source_sha256']='wrong'
        d=Costs(dict(profiles=[],co_runs=[]),c.reports,budget_guard=True)
        self.assertNotIn('a',d.native_budgets)

    def test_future_rollout_actions_respect_guard(self):
        c=self.costs();opts=list(actions(['a','b','c'],{},c,15000,500))
        self.assertTrue(opts);self.assertTrue(all(len(o[0])==1 for o in opts))

    def test_original_cost_rates_not_changed_by_guard(self):
        self.assertEqual(self.costs().rates(['a','b']),self.costs(False).rates(['a','b']))

    def test_invalid_margin_rejected(self):
        with self.assertRaises(ValueError):Costs(dict(profiles=[],co_runs=[]),{},budget_fraction=0)
