"""Contract tests on synthetic descriptors, not performance experiments."""
import copy
import unittest
from variance_delta import SCHEMA, compare_descriptors, histogram_delta, neighborhood_histogram, scalar_delta


def fixture():
    result = {"schema_version": 1,
              "structure": {"complete": True, "histogram": {"conv": 2, "relu": 2}}}
    for component, fields in SCHEMA.items():
        result[component] = {"complete": True, "values": {
            key: "unchanged" if floor is None else 100.0 for key, floor in fields.items()}}
    return result


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.ref = fixture()
        self.new = copy.deepcopy(self.ref)

    def test_identical_descriptors_zero_is_not_an_admission_decision(self):
        result = compare_descriptors(self.new, self.ref)
        self.assertEqual(result["delta"], 0)
        self.assertNotIn("reuse", result)
        self.assertNotIn("admit", result)

    def test_width_change_with_same_graph_is_visible_in_scale(self):
        self.new["scale"]["values"]["activation_bytes"] = 400
        result = compare_descriptors(self.new, self.ref)
        self.assertEqual(result["components"]["structure"]["delta"], 0)
        self.assertEqual(result["components"]["scale"]["delta"], .75)

    def test_pipeline_change_without_graph_change(self):
        self.new["pipeline"]["values"]["preprocess_revision"] = "expensive-augmentation"
        self.assertEqual(compare_descriptors(self.new, self.ref)["delta"], 1)

    def test_optimizer_change_without_forward_change(self):
        self.new["recipe"]["values"]["optimizer"] = "adam"
        self.assertEqual(compare_descriptors(self.new, self.ref)["components"]["recipe"]["delta"], 1)

    def test_missing_field_never_zero(self):
        del self.new["scale"]["values"]["optimizer_bytes"]
        result = compare_descriptors(self.new, self.ref)
        self.assertIsNone(result["delta"])
        self.assertIn("scale.optimizer_bytes", result["unknown"])

    def test_absent_observation_is_unknown_static_features_still_visible(self):
        del self.new["observed"]
        self.new["scale"]["values"]["parameters"] = 200
        result = compare_descriptors(self.new, self.ref)
        self.assertIsNone(result["delta"])
        self.assertEqual(result["components"]["scale"]["delta"], .5)

    def test_incomplete_histogram_cannot_assume_missing_label_zero(self):
        self.new["structure"]["complete"] = False
        self.assertIsNone(compare_descriptors(self.new, self.ref)["components"]["structure"]["known_max"])

    def test_incomplete_component_reports_partial_values_only(self):
        self.new["scale"]["complete"] = False
        self.new["scale"]["values"]["parameters"] = 200
        result = compare_descriptors(self.new, self.ref)["components"]["scale"]
        self.assertIsNone(result["delta"])
        self.assertEqual(result["known_max"], .5)

    def test_unknown_schema_or_extra_feature_rejected(self):
        self.new["schema_version"] = 2
        with self.assertRaises(ValueError): compare_descriptors(self.new, self.ref)
        self.new = fixture()
        self.new["scale"]["values"]["uncaptured_operator"] = 1
        with self.assertRaises(ValueError): compare_descriptors(self.new, self.ref)

    def test_opposing_changes_do_not_cancel(self):
        self.new["scale"]["values"]["parameters"] = 200
        self.new["scale"]["values"]["activation_bytes"] = 50
        self.assertEqual(compare_descriptors(self.new, self.ref)["delta"], .5)

    def test_signed_change_distinguishes_growth_and_shrinkage(self):
        self.assertEqual(scalar_delta(200, 100, 1), {"distance": .5, "signed_relative": 1})
        self.assertEqual(scalar_delta(100, 200, 1), {"distance": .5, "signed_relative": -.5})

    def test_zero_values_and_floor(self):
        self.assertEqual(scalar_delta(0, 0, 1)["distance"], 0)
        self.assertEqual(scalar_delta(1, 0, 1)["distance"], 1)
        with self.assertRaises(ValueError): scalar_delta(0, 0, 0)

    def test_extreme_finite_inputs_do_not_silently_produce_nan(self):
        self.assertEqual(histogram_delta({"a": 1e308}, {"a": 1e308}), 0)
        self.assertEqual(histogram_delta({"a": 1e308}, {"b": 1e308}), 1)
        with self.assertRaises(ValueError): scalar_delta(1e308, 0, 1e-6)

    def test_invalid_numbers_rejected(self):
        for value in [-1, float("inf"), float("nan"), True, "10", 10**1000]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                scalar_delta(value, 1, 1)

    def test_histogram_counts_preserve_repetition(self):
        self.assertAlmostEqual(histogram_delta({"op": 2}, {"op": 1}), 1/3)
        self.assertEqual(histogram_delta({}, {}), 0)
        self.assertEqual(histogram_delta({"a": 1}, {"b": 1}), 1)

    def test_graph_node_renaming_is_invariant(self):
        a = neighborhood_histogram({"a": "conv", "b": "relu"}, [("a", "b", "data")])
        b = neighborhood_histogram({"y": "relu", "x": "conv"}, [("x", "y", "data")])
        self.assertEqual(a, b)

    def test_rewired_graph_with_same_call_counts_is_visible(self):
        nodes = {"a": "op", "b": "op", "c": "op"}
        chain = neighborhood_histogram(nodes, [("a", "b", "data"), ("b", "c", "data")])
        fork = neighborhood_histogram(nodes, [("a", "b", "data"), ("a", "c", "data")])
        self.assertGreater(histogram_delta(chain, fork), 0)

    def test_edge_types_and_direction_are_visible(self):
        nodes = {"a": "conv", "b": "relu"}
        a = neighborhood_histogram(nodes, [("a", "b", "data")])
        b = neighborhood_histogram(nodes, [("b", "a", "data")])
        c = neighborhood_histogram(nodes, [("a", "b", "gradient")])
        self.assertGreater(histogram_delta(a, b), 0)
        self.assertGreater(histogram_delta(a, c), 0)

    def test_invalid_graph_and_work_limits(self):
        for edges, rounds in [([("a", "missing", "data")], 2), ([], 4)]:
            with self.assertRaises(ValueError): neighborhood_histogram({"a": "op"}, edges, rounds)


if __name__ == "__main__":
    unittest.main()
