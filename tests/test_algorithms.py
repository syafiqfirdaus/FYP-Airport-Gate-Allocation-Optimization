import numpy as np
import unittest

from agap_opt.algorithms import ALGORITHMS
from agap_opt.problem import load_klia_case


class AlgorithmSmokeTests(unittest.TestCase):
    def test_all_algorithms(self) -> None:
        problem = load_klia_case()
        for name in sorted(ALGORITHMS):
            with self.subTest(algorithm=name):
                result = ALGORITHMS[name](problem, population=6, iterations=3, seed=7)
                self.assertEqual(result.algorithm, name.upper())
                self.assertEqual(result.history.shape, (3,))
                self.assertTrue(np.isfinite(result.best_score))
                self.assertTrue(problem.validate_assignment(result.best_assignment))
                self.assertTrue(np.all(np.diff(result.history) <= 0))
