from pathlib import Path
import copy
import io
import sys
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from suite.data import synthetic_data
from suite.final_interval import POINT_METHODS, run_task, verify_frames, _validate_oof_audit
from suite.optimized_uncertainty import finite_quantile

class MeanBundle:

    def __init__(self, y):
        self.center = np.mean(y, axis=0)
        self.audit = {'fixture': 'training mean only'}

    def predict_methods(self, x, methods=None):
        n = len(next(iter(x.values())))
        return {name: np.tile(self.center, (n, 1)) + (0.01 if name.endswith('ensemble') else 0) for name in methods or POINT_METHODS}

def fake_fit(x, y, *args, **kwargs):
    return MeanBundle(y)

def fake_crossfit(x, y, profile, seed, methods=None, options=None):
    predictions = {name: np.empty_like(y) for name in methods or POINT_METHODS}
    audit = []
    for train, validation in KFold(n_splits=2, shuffle=True, random_state=seed).split(y):
        fitted = fake_fit({k: v[train] for k, v in x.items()}, y[train])
        values = fitted.predict_methods({k: v[validation] for k, v in x.items()}, methods)
        for name in predictions:
            predictions[name][validation] = values[name]
        audit.append({'train': train.tolist(), 'validation': validation.tolist(), 'entire_model_selection_and_weights_refitted': True})
    return (predictions, audit)

class FinalIntervalTests(unittest.TestCase):

    def setUp(self):
        self.data = synthetic_data(45, 3, seed=229)
        self.task = {'id': 'synthetic', 'kind': 'regression', 'repeat': 0, 'fold': 0, 'seed': 701, 'train': list(range(27)), 'calibration': list(range(27, 36)), 'test': list(range(36, 45))}
        self.profile = {'is_smoke': True, 'inner_folds': 2, 'scale_crossfit_folds': 2, 'candidates_per_family': 1, 'trees': 8}
        self.cfg = {'nominal_coverage': 0.9, 'ensemble_options': {'ensemble_penalty': 0.1}}

    def run_fixture(self, data=None, task=None):
        with patch('suite.final_interval.fit_final_bundle', side_effect=fake_fit), patch('suite.final_interval.strict_crossfit_final', side_effect=fake_crossfit):
            return run_task(self.data if data is None else data, self.task if task is None else task, self.profile, self.cfg)

    def test_same_point_pipeline_and_saved_equations(self):
        frames, audit = self.run_fixture()
        self.assertEqual(verify_frames(frames, 0.1), [])
        self.assertEqual(frames['regression'].method.nunique(), len(POINT_METHODS))
        self.assertEqual(frames['interval_regression'].method.nunique(), 4 * len(POINT_METHODS))
        self.assertEqual(frames['calibration_regression'].method.nunique(), 2 * len(POINT_METHODS))
        self.assertTrue(audit['full_procedure_oof_includes_candidate_selection_and_weights'])
        for receipt in audit['strict_full_pipeline_crossfit']:
            self.assertFalse(set(receipt['fit_bottle_ids']) & set(receipt['held_out_bottle_ids']))
            self.assertFalse(set(receipt['held_out_bottle_ids']) & set(audit['calibration_partition_ids']))
        changed = {k: a.copy() for k, a in frames.items()}
        changed['interval_regression'].loc[0, 'upper'] += 1
        self.assertTrue(verify_frames(changed, 0.1))

    def test_calibration_labels_only_change_quantiles(self):
        frames, _ = self.run_fixture()
        changed = copy.deepcopy(self.data)
        changed['Y'][self.task['calibration']] *= 10
        other, _ = self.run_fixture(changed)
        for key, cols in [('regression', ['y_pred']), ('interval_regression', ['y_pred', 'interval_scale']), ('calibration_regression', ['y_pred', 'interval_scale']), ('risk_rules', ['threshold']), ('risk_predictions', ['y_pred', 'threshold', 'uncertainty_score', 'accepted'])]:
            np.testing.assert_array_equal(frames[key][cols], other[key][cols])
        self.assertFalse(np.allclose(frames['interval_regression']['quantile'], other['interval_regression']['quantile']))

    def test_test_responses_never_change_predictions_bounds_or_selection(self):
        frames, audit = self.run_fixture()
        changed = copy.deepcopy(self.data)
        changed['Y'][self.task['test']] += 100
        other, other_audit = self.run_fixture(changed)
        for key, cols in [('interval_regression', ['y_pred', 'interval_scale', 'quantile', 'lower', 'upper']), ('risk_rules', ['threshold']), ('risk_predictions', ['y_pred', 'threshold', 'uncertainty_score', 'accepted'])]:
            np.testing.assert_array_equal(frames[key][cols], other[key][cols])
        self.assertEqual(audit, other_audit)

    def test_test_abv_is_posthoc_diagnostic_only(self):
        frames, _ = self.run_fixture()
        changed = copy.deepcopy(self.data)
        changed['ethanol'][self.task['test']] += 100
        other, _ = self.run_fixture(changed)
        np.testing.assert_array_equal(frames['interval_regression'][['y_pred', 'lower', 'upper']], other['interval_regression'][['y_pred', 'lower', 'upper']])

    def test_all_partition_indices_must_be_unique_and_disjoint(self):
        for bad in [dict(self.task, calibration=[0, 27, 28]), dict(self.task, calibration=[27, 27, 28]), dict(self.task, calibration=[-1, 27]), dict(self.task, calibration=[27.5, 28.5])]:
            with self.assertRaises(ValueError):
                self.run_fixture(task=bad)

    def test_complete_pipeline_crossfit_receipt_is_required(self):
        with self.assertRaisesRegex(ValueError, 'exclude'):
            _validate_oof_audit([{'train': [0, 1, 2], 'validation': [1]}], 3)
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            _validate_oof_audit([{'train': [1, 2], 'validation': [0]}], 3)
        with patch('suite.final_interval.fit_final_bundle', side_effect=fake_fit), patch('suite.final_interval.strict_crossfit_final', return_value=({name: np.zeros((27, 3)) for name in POINT_METHODS}, [{'train': list(range(27)), 'validation': [0]}])):
            with self.assertRaisesRegex(ValueError, 'exclude'):
                run_task(self.data, self.task, self.profile, self.cfg)

    def test_csv_roundtrip_infinity_and_finite_order_statistics(self):
        task = dict(self.task, calibration=list(range(27, 35)), test=list(range(35, 45)))
        frames, _ = self.run_fixture(task=task)
        self.assertTrue(np.isposinf(frames['interval_regression']['quantile']).all())
        loaded = {k: pd.read_csv(io.StringIO(a.to_csv(index=False)), dtype={'sample_id': str}, float_precision='round_trip') for k, a in frames.items()}
        self.assertEqual(verify_frames(loaded, 0.1), [])
        p = np.array([1 - 7e-16, 1 - 6e-16, 1 - 5e-16, 1 - 4e-16, 1 - 3e-16, 1 - 2e-16, 1 - 1e-16, 1.0, 1.0])
        restored = pd.read_csv(io.StringIO(pd.DataFrame({'score': p}).to_csv(index=False)), float_precision='round_trip').score.to_numpy()
        np.testing.assert_array_equal(restored, p)
        self.assertEqual(float(finite_quantile(restored, 0.1)), float(finite_quantile(p, 0.1)))

    def test_aggregate_repeat_folds_verify_independently(self):
        frames, _ = self.run_fixture()
        other = {k: a.assign(repeat=1) for k, a in frames.items()}
        combined = {k: pd.concat([a, other[k]], ignore_index=True) for k, a in frames.items()}
        self.assertEqual(verify_frames(combined, 0.1), [])
        duplicated = {k: a.copy() for k, a in frames.items()}
        duplicated['calibration_regression'] = pd.concat([duplicated['calibration_regression'], duplicated['calibration_regression'].iloc[[0]]])
        self.assertTrue(verify_frames(duplicated, 0.1))

    def test_full_retention_never_drops_samples_and_rule_tampering_fails(self):
        frames, _ = self.run_fixture()
        full = frames['risk_predictions'].query('retained_fraction == 1')
        self.assertTrue(full.accepted.all())
        self.assertTrue(np.isposinf(full.threshold).all())
        self.assertTrue(np.isfinite(full.uncertainty_score).all())
        bad = {k: a.copy() for k, a in frames.items()}
        bad['risk_rules'].loc[0, 'threshold'] += 10
        self.assertTrue(verify_frames(bad, 0.1))
if __name__ == '__main__':
    unittest.main()
