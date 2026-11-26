import unittest
from localml_scheduler.scheduler.arrival_order import ready_order

class ArrivalTests(unittest.TestCase):
    def test_late_manifest_predecessor_does_not_pass_earlier_arrival(self):
        sequence={'late_C4':2,'early_C5':1,'later_C5':3}
        self.assertEqual(ready_order(sequence,sequence,{}, {},fifo=True),['early_C5','late_C4','later_C5'])
    def test_precheck_completion_does_not_define_fifo_order(self):
        sequence={'first':1,'second':2}
        self.assertEqual(ready_order(sequence,{'second':{}},{},{},fifo=True),[])
        self.assertEqual(ready_order(sequence,{'second':{}},{},{},fifo=False),['second'])
    def test_finished_reports_recover_original_acceptance_order(self):
        self.assertEqual(ready_order({'first':1,'second':2},{'second':{},'first':{}},{},{},fifo=True),['first','second'])
    def test_actual_residents_and_exits_removed(self):
        q={'a':1,'b':2,'c':3,'d':4}
        self.assertEqual(ready_order(q,q,{'a':{}},{'b':{}},fifo=True),['c','d'])
    def test_stable_event_tie_sequence_is_preserved(self):
        self.assertEqual(ready_order({'b':2,'a':1},{'b':{},'a':{}},{},{},fifo=False),['a','b'])
    def test_no_future_reports(self):
        with self.assertRaises(ValueError):ready_order({'a':1},{'a':{},'future':{}},{},{},fifo=False)
    def test_bad_sequence_rejected(self):
        for q in [{'a':1,'b':1},{'a':0},{'a':True}]:
            with self.assertRaises(ValueError):ready_order(q,q,{},{},fifo=True)

if __name__=='__main__':unittest.main()
