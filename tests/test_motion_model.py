"""Model-only head-motion regression checks; no camera/model downloads."""
import unittest
from types import SimpleNamespace

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from head_tracking import HeadAwareEstimator


class FakeModel:
    def train(self, x, y, variable_scaling):
        self.scaler = StandardScaler()
        self.weights = variable_scaling
        self.targets = y.copy()
        self.regression = Ridge(alpha=1).fit(
            self.scaler.fit_transform(x) * self.weights, y)

    def predict(self, x):
        return self.regression.predict(self.scaler.transform(x) * self.weights)


def estimator():
    result = HeadAwareEstimator.__new__(HeadAwareEstimator)
    upstream = SimpleNamespace(model=FakeModel())
    upstream.train = lambda x, y, variable_scaling: upstream.model.train(x, y, variable_scaling)
    upstream.predict = lambda x: upstream.model.predict(x)
    result.estimator = upstream
    return result


def samples():
    target = np.repeat([-1., 1.], 20)
    base = np.zeros((40, 10))
    base[:, 0] = target
    labels = np.column_stack([target * 100, target * 40])
    motion = np.zeros((120, 10))
    motion[:, -3] = np.linspace(-.2, .2, 120)
    motion[:, 0] = -motion[:, -3]
    return base, labels, np.vstack([base, motion]), np.vstack([labels, np.zeros((120, 2))])


class MotionModelTests(unittest.TestCase):
    def test_one_motion_target_is_not_diluted_by_ten_stationary_targets(self):
        model = estimator()
        base = np.zeros((200, 10))
        base[:, 0] = np.repeat(np.linspace(-1, 1, 10), 20)
        targets = np.column_stack([base[:, 0] * 100, base[:, 0] * 40])
        motion = np.zeros((40, 10))
        motion[:, -3] = np.linspace(-.08, .08, 40)
        motion[:, 0] = -motion[:, -3]
        model.train(np.vstack([base, motion]), np.vstack([targets, np.zeros((40, 2))]))
        self.assertEqual(model.pose_weights[-3], 1)
        self.assertLess(np.mean(np.abs(model.estimator.predict(motion)[:, 0])), 2)

    def test_resampling_does_not_turn_five_frames_into_motion_evidence(self):
        model = estimator()
        base, labels, _, _ = samples()
        motion = np.zeros((5, 10))
        motion[:, -3] = np.linspace(-.2, .2, 5)
        model.train(np.vstack([base, motion]), np.vstack([labels, np.zeros((5, 2))]))
        self.assertEqual(model.pose_weights[-3], 0)

    def test_reject_restores_predictions_and_guards(self):
        model = estimator()
        base, labels, combined, targets = samples()
        model.train(base, labels)
        before = model.estimator.predict(combined)
        guards = [value.copy() for value in (model.pose_weights, model.pose_low, model.pose_high)]
        model.pose_in_range = False
        model.begin_candidate(combined, targets)
        old, candidate = model.comparison(combined)
        np.testing.assert_allclose(old, before)
        self.assertTrue(np.isfinite(candidate).all())
        self.assertLess(np.mean(np.abs(candidate[40:])), np.mean(np.abs(old[40:])))
        model.finish_candidate(False)
        np.testing.assert_allclose(model.estimator.predict(combined), before)
        for current, expected in zip((model.pose_weights, model.pose_low, model.pose_high), guards):
            np.testing.assert_array_equal(current, expected)
        self.assertFalse(model.pose_in_range)

    def test_accept_and_reload_reproduce_balanced_candidate(self):
        model = estimator()
        base, labels, combined, targets = samples()
        model.train(base, labels)
        model.begin_candidate(combined, targets)
        _, candidate = model.comparison(combined)
        _, counts = np.unique(model.estimator.model.targets, axis=0, return_counts=True)
        np.testing.assert_array_equal(counts, [120, 120, 120])
        model.finish_candidate(True)
        np.testing.assert_allclose(model.estimator.predict(combined), candidate)
        reloaded = estimator()
        reloaded.train(combined, targets)
        np.testing.assert_allclose(reloaded.estimator.predict(combined), candidate)
        with self.assertRaises(RuntimeError):
            model.comparison(combined)

    def test_failed_candidate_restores_baseline(self):
        model = estimator()
        base, labels, combined, targets = samples()
        model.train(base, labels)
        before = model.estimator.predict(base)
        with self.assertRaises(ValueError):
            model.begin_candidate(combined, targets[:-1])
        np.testing.assert_allclose(model.estimator.predict(base), before)
        self.assertIsNone(model._baseline)


if __name__ == '__main__':
    unittest.main()
