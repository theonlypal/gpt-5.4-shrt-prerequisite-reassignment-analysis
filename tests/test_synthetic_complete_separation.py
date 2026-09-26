import math
import unittest

import numpy as np

from analysis import firth_interaction, interaction_estimate


class CompleteSeparationTests(unittest.TestCase):
    def test_predicted_checkerboard_complete_separation(self):
        result = firth_interaction([0, 200, 200, 0], [200] * 4)
        self.assertTrue(result["converged"])
        self.assertTrue(all(math.isfinite(value) for value in result["profile_ci95"]))
        self.assertLess(result["profile_ci95"][1], 0)
        self.assertAlmostEqual(result["coefficient"], -4 * math.log(401))
        self.assertLess(result["p_two_sided"], .05)
        self.assertEqual(interaction_estimate([0, 200, 200, 0], [200] * 4)["interaction"], 2)

    def test_reversed_checkerboard_flips_coefficient_and_interval(self):
        forward = firth_interaction([0, 200, 200, 0], [200] * 4)
        reverse = firth_interaction([200, 0, 0, 200], [200] * 4)
        self.assertAlmostEqual(forward["coefficient"], -reverse["coefficient"], places=8)
        self.assertAlmostEqual(forward["profile_ci95"][0], -reverse["profile_ci95"][1], places=5)
        self.assertAlmostEqual(forward["profile_ci95"][1], -reverse["profile_ci95"][0], places=5)
        self.assertAlmostEqual(forward["p_two_sided"], reverse["p_two_sided"], places=8)

    def test_null_effect_all_zero_all_one_half_and_one_factor(self):
        for y in ([0] * 4, [200] * 4, [100] * 4, [20, 20, 100, 100], [20, 100, 20, 100]):
            with self.subTest(y=y):
                result = firth_interaction(y, [200] * 4)
                self.assertAlmostEqual(result["coefficient"], 0, places=8)
                self.assertLess(result["profile_ci95"][0], 0)
                self.assertGreater(result["profile_ci95"][1], 0)
                self.assertAlmostEqual(result["p_two_sided"], 1, places=5)

    def test_factor_swaps_preserve_or_reverse_as_algebra_requires(self):
        # Unequal counts/totals prevent a symmetric fixture from hiding cell errors.
        y, n = np.array([5, 140, 175, 25]), np.array([170, 180, 190, 200])
        base = firth_interaction(y, n)
        for permutation, sign in (([2, 3, 0, 1], -1), ([1, 0, 3, 2], -1),
                                  ([3, 2, 1, 0], 1), ([0, 2, 1, 3], 1)):
            with self.subTest(permutation=permutation):
                result = firth_interaction(y[permutation], n[permutation])
                self.assertAlmostEqual(result["coefficient"], sign * base["coefficient"], places=8)
                expected = base["profile_ci95"] if sign == 1 else [-base["profile_ci95"][1], -base["profile_ci95"][0]]
                np.testing.assert_allclose(result["profile_ci95"], expected, atol=1e-5)
                self.assertAlmostEqual(result["penalized_lr"], base["penalized_lr"], places=6)
                self.assertAlmostEqual(result["p_two_sided"], base["p_two_sided"], places=8)


if __name__ == "__main__":
    unittest.main()
