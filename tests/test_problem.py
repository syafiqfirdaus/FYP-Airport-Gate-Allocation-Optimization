import numpy as np
import unittest

from agap_opt.problem import load_klia_case


class ProblemTests(unittest.TestCase):
    def test_case_dimensions_and_fixed_schedule(self) -> None:
        problem = load_klia_case()
        self.assertEqual(problem.dimension, 31)
        self.assertEqual(problem.n_gates, 16)
        self.assertEqual(len(problem.fixed_flights), 9)

    def test_repair_produces_valid_schedule(self) -> None:
        problem = load_klia_case()
        evaluation = problem.evaluate_preferences(np.zeros(problem.dimension, dtype=int))
        self.assertTrue(np.isfinite(evaluation.score))
        self.assertTrue(problem.validate_assignment(evaluation.assignment))
