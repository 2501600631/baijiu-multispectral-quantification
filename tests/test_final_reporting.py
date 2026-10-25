import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from suite.common import digest, iqr, save_csv, save_json, sha, csv_read
from suite.final_ensemble import FINAL_METHODS, STRATEGY_METHODS
from suite.final_reporting import generate, verify, _verify_router, _paired, _verify_local

class FinalReportingTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='V10 report test ', dir=str(Path(__file__).resolve().parent))
        self.root = Path(self.tmp.name)
        self.out = self.root / 'results'
        self.task = {'id': 'reg_r00_f00', 'train': list(range(8)), 'calibration': [], 'test': list(range(8, 13)), 'repeat': 0, 'fold': 0, 'seed': 9}
        y = np.column_stack([np.arange(13) / 7.0, (np.arange(13) + 1.0) ** 1.3])
        self.data = {'ids': np.array(['b%02d' % i for i in range(13)]), 'Y': y, 'target_names': ['a', 'b']}
        self.cfg = {'nominal_coverage': 0.9, 'seed': 12}
        self.plan = {'mode': 'ablation', 'tasks': [self.task], 'parameters': {'bootstrap': 30}, 'evidence_tier': 'synthetic_only'}
        self.plan['plan_hash'] = digest(self.plan)
        self.cp = self.out / 'checkpoints' / self.task['id']
        self.cp.mkdir(parents=True)
        scale = iqr(y[:8])
        records = []
        for k, method in enumerate(FINAL_METHODS):
            for index in self.task['test']:
                for j, target in enumerate(self.data['target_names']):
                    records.append({'sample_id': self.data['ids'][index], 'target': target, 'repeat': 0, 'fold': 0, 'method': method, 'y_true': y[index, j], 'y_pred': y[index, j] + 0.1 + 0.02 * k, 'train_iqr': scale[j], 'fit_n': 8})
        save_csv(self.cp / 'regression.csv', pd.DataFrame(records))
        candidate = 'candidate01'
        bundle = {'training_bottles': 8, 'candidate_order': [candidate], 'reference_candidate': candidate, 'shared_catalog_candidate': candidate, 'minimum_routes': [candidate] * 2, 'conservative_routes': [candidate] * 2, 'penalties_by_method': {'v10_unanchored_ensemble': 0.0, 'v10_anchored_ensemble': 0.1}, 'inner_splits': [{'fit': list(range(4, 8)), 'validation': list(range(4))}, {'fit': list(range(4)), 'validation': list(range(4, 8))}], 'oof_membership_in_supplied_training_order': [0] * 4 + [1] * 4, 'weights_by_method': {m: [[1.0], [1.0]] for m in ('v10_equal_weight', 'v10_unanchored_ensemble', 'v10_anchored_ensemble')}, 'strategy_router': {'enabled': False, 'routes': ['v10_pls2_reference'] * 2, 'reason': 'synthetic internal procedure fixture'}, 'candidates': [{'id': candidate, 'spec': {'family': 'ridge', 'modes': ['uv']}, 'oof_predictions': y[:8].tolist(), 'fold_scores_targetwise': [[0.0, 0.0], [0.0, 0.0]]}]}
        save_json(self.cp / 'audit.json', {'bundle': bundle})
        self.receipt()

    def tearDown(self):
        self.tmp.cleanup()

    def receipt(self):
        save_json(self.cp / 'receipt.json', {'task_id': self.task['id'], 'plan_hash': self.plan['plan_hash'], 'files': {name: sha(self.cp / name) for name in ['regression.csv', 'audit.json']}})

    def test_readonly_generate_complete_tables_and_cluster_scope(self):
        with patch('suite.final_ensemble.fit_model', side_effect=AssertionError('No model fitting in report')):
            result = generate(self.root, self.out, self.data, self.cfg, self.plan)
        self.assertTrue(result['ok'], result)
        self.assertFalse(result['model_fitting'])
        summary = csv_read(self.out / 'tables' / 'regression_summary.csv')
        self.assertEqual(len(summary), len(FINAL_METHODS) * 2)
        paired = csv_read(self.out / 'tables' / 'paired_comparisons.csv')
        self.assertIn('__macro__', set(paired.target))
        self.assertTrue(paired.uncertainty_scope.str.contains('conditional_on_saved').all())
        self.assertTrue((paired.unique_bottles == 5).all())
        self.assertTrue((self.out / 'tables' / 'weight_stability.csv').is_file())
        self.assertTrue((self.out / 'report_receipt.json').is_file())
        selection = csv_read(self.out / 'tables' / 'strategy_selection_by_fold.csv')
        self.assertEqual(set(selection.selected_procedure), {'v10_pls2_reference'})
        mappings = csv_read(self.out / 'tables' / 'real_winning_algorithm_mappings.csv')
        self.assertEqual(len(mappings), len(FINAL_METHODS) * 2)
        self.assertFalse(mappings.historical_best_scores_spliced.any())

    def test_reference_tampering_detected_even_after_receipt_recalculated(self):
        frame = csv_read(self.cp / 'regression.csv', dtype={'sample_id': str})
        frame.loc[0, 'y_true'] += 1.0
        save_csv(self.cp / 'regression.csv', frame)
        self.receipt()
        result = verify(self.root, self.out, self.data, self.cfg, self.plan)
        self.assertFalse(result['ok'])
        self.assertTrue(any(('Reference values' in e for e in result['errors'])))

    def test_truncated_bottle_grid_and_modified_source_receipt_fail(self):
        frame = csv_read(self.cp / 'regression.csv', dtype={'sample_id': str})
        save_csv(self.cp / 'regression.csv', frame.iloc[1:])
        result = verify(self.root, self.out, self.data, self.cfg, self.plan)
        self.assertFalse(result['ok'])
        self.assertTrue(any(('digest mismatch' in e for e in result['errors'])))
        self.receipt()
        result = verify(self.root, self.out, self.data, self.cfg, self.plan)
        self.assertFalse(result['ok'])
        self.assertTrue(any(('Incomplete bottle grid' in e for e in result['errors'])))

    def strategy_fixture(self):
        Y = np.column_stack([np.arange(30) / 9.0, np.arange(30) ** 1.2 + 0.4])
        data = {'Y': Y, 'target_names': ['a', 'b']}
        task = {'train': list(range(24)), 'test': list(range(24, 30))}
        values = {name: (Y[:24] + 0.1 * (i + 1)).tolist() for i, name in enumerate(STRATEGY_METHODS)}
        scores = np.column_stack([np.mean(((np.asarray(values[name]) - Y[:24]) / iqr(Y[:24])) ** 2, axis=0) for name in STRATEGY_METHODS])
        folds = []
        for fit, va in [(list(range(12)), list(range(12, 24))), (list(range(12, 24)), list(range(12)))]:
            folds.append({'train': fit, 'validation': va, 'training_bottles': len(fit), 'all_selection_and_weights_exclude_validation': True, 'inner_splits': [{'fit': list(range(6)), 'validation': list(range(6, 12))}, {'fit': list(range(6, 12)), 'validation': list(range(6))}], 'training_audit_digest': 'synthetic-digest', 'selected_reference': 'one', 'candidate_order': ['one']})
        bundle = {'strategy_router': {'enabled': True, 'fallback': False, 'routes': ['v10_pls2_reference'] * 2, 'methods': list(STRATEGY_METHODS), 'strict_procedure_oof': values, 'scores_targetwise': scores.tolist(), 'crossfit': folds, 'route_winners_chosen_using_outer_test': False, 'history_metric_splicing': False}}
        return (data, task, bundle)

    def test_router_scores_and_fold_membership_recomputed_without_fitting(self):
        data, task, bundle = self.strategy_fixture()
        self.assertEqual(_verify_router(data, task, bundle), [])
        changed = dict(data, Y=data['Y'].copy())
        changed['Y'][task['test']] += 999
        self.assertEqual(_verify_router(changed, task, bundle), [])
        bundle['strategy_router']['scores_targetwise'][0][0] += 1
        self.assertTrue(any(('scores differ' in x for x in _verify_router(data, task, bundle))))

    def test_router_rejects_test_selected_route_and_overlap_but_allows_internal_disable(self):
        data, task, bundle = self.strategy_fixture()
        bundle['strategy_router']['routes'][0] = 'v10_equal_weight'
        self.assertTrue(any(('routes differ' in x for x in _verify_router(data, task, bundle))))
        bundle['strategy_router']['crossfit'][0]['train'].append(12)
        self.assertTrue(any(('exclusion mismatch' in x for x in _verify_router(data, task, bundle))))
        bundle['strategy_router'] = {'enabled': False, 'routes': ['v10_pls2_reference'] * 2}
        self.assertEqual(_verify_router(data, task, bundle, internal=True), [])
        self.assertTrue(any(('disabled' in x for x in _verify_router(data, task, bundle))))

    def test_optional_ethanol_metrics_keep_source_unit_and_complete_method_inventory(self):
        records = csv_read(self.cp / 'regression.csv', dtype={'sample_id': str})
        records = records[records.target == 'a'].copy()
        records['target'] = 'ethanol'
        source = dict(self.data, target_names=['ethanol'], Y=self.data['Y'][:, :1])
        self.assertEqual(_verify_local(source, self.task, {'regression': records}, 'ethanol', 0.1), [])
        paired = _paired(records, 10, 7)
        self.assertEqual(set(paired[paired.target == 'ethanol'].metric), {'RMSE_source_unit'})
        from suite.final_plotting import _base, ANCHORED, WEIGHT_METHOD
        self.assertEqual(_base('v10_v9_unanchored_reference__fixed__marginal'), 'v10_v9_unanchored_reference')
        self.assertEqual(_base('v10_strategy_router__adaptive__joint'), 'v10_strategy_router')
        self.assertNotEqual(ANCHORED, WEIGHT_METHOD)
if __name__ == '__main__':
    unittest.main()
