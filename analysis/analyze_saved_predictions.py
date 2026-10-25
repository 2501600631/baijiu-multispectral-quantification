import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis'
OUT.mkdir(parents=True, exist_ok=True)
V10 = ROOT
OLD = ROOT / 'archives/comparison'
CN = dict(zip(['acetaldehyde', 'methanol', 'n_propanol', 'ethyl_acetate', 'isobutanol', 'acetal', 'n_butanol', 'isoamyl_alcohol', 'ethyl_butyrate', 'ethyl_lactate', 'ethyl_hexanoate'], ['Acetaldehyde', 'Methanol', 'n-Propanol', 'Ethyl acetate', 'Isobutanol', 'Acetal', 'n-Butanol', 'Isoamyl alcohol', 'Ethyl butyrate', 'Ethyl lactate', 'Ethyl hexanoate']))
INPUTS = []

def read(p):
    if not p.exists() and p.with_suffix(p.suffix + '.gz').exists():
        p = p.with_suffix(p.suffix + '.gz')
    INPUTS.append(p)
    return pd.read_csv(p, encoding='utf-8-sig', dtype={'sample_id': str}, low_memory=False, float_precision='round_trip')

def dump(name, data):
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    frame.to_csv(OUT / (name + '.csv'), index=False, encoding='utf-8-sig')
    return frame

def jsonsave(name, data):

    def convert(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x))
    (OUT / (name + '.json')).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=convert), encoding='utf-8')

def bools(x):
    return x.astype(str).str.lower().isin(['true', '1'])

def main():
    samples = read(V10 / 'data/aligned/samples.csv').set_index('sample_id')
    interval_plan_path = V10 / 'archives/main/interval/plan.json'
    INPUTS.append(interval_plan_path)
    plan = json.loads(interval_plan_path.read_text(encoding='utf-8-sig'))
    tasks = {(t['repeat'], t['fold']): t for t in plan['tasks']}
    ids = plan['data_ids']
    assert len(ids) == 139 and set(ids) == set(samples.index)
    descriptors = []
    for target in ['ethanol'] + list(CN):
        x = samples[target]
        descriptors.append({'target': target, 'target_label': CN.get(target, 'Measured alcohol strength'), 'bottles': len(x), 'minimum': x.min(), 'maximum': x.max(), 'mean': x.mean(), 'sd': x.std(ddof=1), 'median': x.median(), 'q25': x.quantile(0.25), 'q75': x.quantile(0.75), 'pearson_r_with_ethanol': x.corr(samples.ethanol), 'spearman_r_with_ethanol': x.corr(samples.ethanol, method='spearman')})
    dump('sample_descriptive_statistics', descriptors)
    train_info = {}
    for key, t in tasks.items():
        fit = t.get('fit', t.get('train'))
        if fit is None:
            raise KeyError('No fit partition')
        train_info[key] = samples.loc[[ids[i] for i in fit]]
    points = read(OLD / 'all_method_component_metrics.csv')
    dump('all_method_component_metrics', points)
    dump('primary_component_metrics', points[(points.version == 'V10') & (points.method == 'v10_strategy_router')].copy())
    point_macro = read(OLD / 'all_method_macro_ranking.csv')
    dump('method_macro_metrics', point_macro)
    abv = read(V10 / 'archives/main/measured_abv/tables/records_regression.csv')
    pairs = [('v10_strategy_router', 'v10_measured_abv_only'), ('v10_v9_unanchored_reference', 'v10_measured_abv_only'), ('v10_measured_abv_residual', 'v10_measured_abv_only'), ('v10_measured_abv_guarded', 'v10_measured_abv_only')]
    rng = np.random.default_rng(20261024)
    sorted_ids = sorted(samples.index)
    draws = rng.integers(0, len(sorted_ids), size=(4000, len(sorted_ids)))
    abv_rows = []
    for target in CN:
        g = abv[abv.target == target].copy()
        assert g.groupby(['method', 'sample_id']).size().eq(5).all()
        err = {}
        metrics = {}
        for method, s in g.groupby('method'):
            s = s.copy()
            s['se'] = (s.y_pred - s.y_true) ** 2
            err[method] = s.groupby('sample_id').se.mean().reindex(sorted_ids).to_numpy()
            metrics[method] = {'RMSE': np.sqrt(s.se.mean()), 'R2': 1 - s.se.sum() / ((s.y_true - s.y_true.mean()) ** 2).sum(), 'nRMSE': np.sqrt((s.se / s.train_iqr ** 2).mean())}
        for method, base in pairs:
            a, b = (err[method], err[base])
            gain = 100 * (1 - np.sqrt(a.mean() / b.mean()))
            delta = 100 * (1 - np.sqrt(a[draws].mean(axis=1) / b[draws].mean(axis=1)))
            lo, hi = np.quantile(delta, [0.025, 0.975])
            abv_rows.append({'target': target, 'target_label': CN[target], 'method': method, 'baseline': base, 'method_RMSE': metrics[method]['RMSE'], 'baseline_RMSE': metrics[base]['RMSE'], 'method_R2': metrics[method]['R2'], 'baseline_R2': metrics[base]['R2'], 'method_nRMSE': metrics[method]['nRMSE'], 'baseline_nRMSE': metrics[base]['nRMSE'], 'gain_pct': gain, 'conditional_bottle_CI_low_pct': lo, 'conditional_bottle_CI_high_pct': hi, 'bottles': len(sorted_ids), 'repeats': 5, 'bootstrap_draws': len(draws)})
    abv_results = dump('measured_alcohol_strength_paired_gains', abv_rows)
    intervals = read(V10 / 'archives/main/interval/tables/records_interval_regression.csv')
    intervals = intervals[intervals.base_model.eq('v10_strategy_router')].copy()
    intervals['covered'] = (intervals.y_true >= intervals.lower) & (intervals.y_true <= intervals.upper)
    intervals['under_upper'] = intervals.y_true > intervals.upper
    intervals['over_lower'] = intervals.y_true < intervals.lower
    intervals['error'] = intervals.y_pred - intervals.y_true
    intervals['ethanol'] = intervals.sample_id.map(samples.ethanol)
    intervals['high_concentration'] = bools(intervals.above_train_q75)
    for (rep, fold, target), s in intervals.groupby(['repeat', 'fold', 'target']):
        assert np.allclose(s.y_true.to_numpy(), s.sample_id.map(samples[target]).to_numpy(), rtol=0, atol=1e-13)
        threshold = train_info[rep, fold][target].quantile(0.75)
        assert np.array_equal(s.high_concentration.to_numpy(), s.y_true.to_numpy() > threshold)
    tail_rows = []
    for (scale, cov, target, high), s in intervals.groupby(['scale_type', 'coverage_type', 'target', 'high_concentration']):
        tail_rows.append({'scale_type': scale, 'coverage_type': cov, 'target': target, 'target_label': CN[target], 'high_concentration': bool(high), 'records': len(s), 'unique_bottles': s.sample_id.nunique(), 'coverage': s.covered.mean(), 'truth_above_upper_rate': s.under_upper.mean(), 'truth_below_lower_rate': s.over_lower.mean(), 'signed_bias_g_L': s.error.mean(), 'RMSE_g_L': np.sqrt((s.error ** 2).mean()), 'MAE_g_L': s.error.abs().mean(), 'mean_width_g_L': (s.upper - s.lower).mean(), 'mean_width_iqr': ((s.upper - s.lower) / s.train_iqr).mean(), 'mean_true_g_L': s.y_true.mean(), 'mean_predicted_g_L': s.y_pred.mean(), 'mean_ethanol': s.ethanol.mean()})
    tails = dump('concentration_boundary_bias_and_misses', tail_rows)
    risks = read(V10 / 'archives/main/interval/tables/records_risk_predictions.csv')
    risks = risks[risks.base_model.eq('v10_strategy_router')].copy()
    assert risks.method.eq('v10_strategy_router__adaptive').all()
    risks['accepted'] = bools(risks.accepted)
    risks['ethanol'] = risks.sample_id.map(samples.ethanol)
    risks['se'] = (risks.y_pred - risks.y_true) ** 2
    risks['nse'] = risks.se / risks.train_iqr ** 2
    assert risks.groupby(['method', 'target', 'retained_fraction', 'sample_id']).size().eq(5).all()
    marginal = intervals[(intervals.scale_type == 'adaptive') & (intervals.coverage_type == 'marginal')]
    risks = risks.merge(marginal[['repeat', 'fold', 'sample_id', 'target', 'covered', 'lower', 'upper']], on=['repeat', 'fold', 'sample_id', 'target'], validate='many_to_one')
    risks['y_quartile'] = -1
    risks['y_median'] = -1
    risks['abv_median'] = -1
    for (rep, fold, target), index in risks.groupby(['repeat', 'fold', 'target']).groups.items():
        tr = train_info[rep, fold]
        yedges = np.quantile(tr[target], [0.25, 0.5, 0.75])
        eedge = np.median(tr.ethanol)
        risks.loc[index, 'y_quartile'] = np.searchsorted(yedges, risks.loc[index, 'y_true'], side='right')
        risks.loc[index, 'y_median'] = (risks.loc[index, 'y_true'] >= yedges[1]).astype(int)
        risks.loc[index, 'abv_median'] = (risks.loc[index, 'ethanol'] >= eedge).astype(int)
    risks['joint_stratum'] = risks.y_median * 2 + risks.abv_median
    risk_rows = []
    base_rows = []
    repeat_rows = []
    n_draws = 2000
    u = np.random.default_rng(20261025).random((n_draws, len(sorted_ids)))
    idmap = {sid: i for i, sid in enumerate(sorted_ids)}
    null_names = ['uniform_same_count', 'truth_quartile_matched', 'truth_abv_median_matched']
    sim_cache = {}
    for (target, frac), g in risks.groupby(['target', 'retained_fraction']):
        accepted = g[g.accepted]
        rejected = g[~g.accepted]
        all_rmse = np.sqrt(g.se.mean())
        acc_rmse = np.sqrt(accepted.se.mean())
        acc_nrmse = np.sqrt(accepted.nse.mean())
        row = {'target': target, 'target_label': CN[target], 'retained_fraction': frac, 'actual_retention': g.accepted.mean(), 'accepted_records': len(accepted), 'rejected_records': len(rejected), 'unique_accepted_bottles': accepted.sample_id.nunique(), 'all_RMSE': all_rmse, 'accepted_RMSE': acc_rmse, 'rejected_RMSE': np.sqrt(rejected.se.mean()) if len(rejected) else np.nan, 'all_nRMSE': np.sqrt(g.nse.mean()), 'accepted_nRMSE': acc_nrmse, 'accepted_marginal_coverage': accepted.covered.mean(), 'rejected_marginal_coverage': rejected.covered.mean() if len(rejected) else np.nan, 'accepted_mean_truth': accepted.y_true.mean(), 'rejected_mean_truth': rejected.y_true.mean() if len(rejected) else np.nan, 'accepted_mean_ethanol': accepted.ethanol.mean(), 'rejected_mean_ethanol': rejected.ethanol.mean() if len(rejected) else np.nan, 'gain_vs_all_pct': 100 * (1 - acc_rmse / all_rmse), 'rejected_to_accepted_MSE_ratio': rejected.se.mean() / accepted.se.mean() if len(rejected) else np.nan}
        risk_rows.append(row)
        for rep, s in g.groupby('repeat'):
            a = s[s.accepted]
            repeat_rows.append({'target': target, 'target_label': CN[target], 'retained_fraction': frac, 'repeat': rep, 'actual_retention': s.accepted.mean(), 'accepted_RMSE': np.sqrt(a.se.mean()), 'all_RMSE': np.sqrt(s.se.mean()), 'gain_pct': 100 * (1 - np.sqrt(a.se.mean() / s.se.mean())), 'accepted_nRMSE': np.sqrt(a.nse.mean()), 'all_nRMSE': np.sqrt(s.nse.mean())})
        for null_name in null_names:
            sse = np.zeros(n_draws)
            nsse = np.zeros(n_draws)
            ncovered = np.zeros(n_draws)
            total = 0
            variable_groups = 0
            groups = 0
            keys = ['repeat', 'fold']
            if null_name == 'truth_quartile_matched':
                keys.append('y_quartile')
            elif null_name == 'truth_abv_median_matched':
                keys.append('joint_stratum')
            for _, s in g.groupby(keys):
                k = int(s.accepted.sum())
                groups += 1
                if not k:
                    continue
                n = len(s)
                total += k
                if k == n:
                    sse += s.se.sum()
                    nsse += s.nse.sum()
                    ncovered += s.covered.sum()
                    continue
                variable_groups += 1
                orders = np.argpartition(u[:, [idmap[x] for x in s.sample_id]], k - 1, axis=1)[:, :k]
                sse += s.se.to_numpy()[orders].sum(axis=1)
                nsse += s.nse.to_numpy()[orders].sum(axis=1)
                ncovered += s.covered.to_numpy()[orders].sum(axis=1)
            assert total == len(accepted)
            rmses = np.sqrt(sse / total)
            nrmses = np.sqrt(nsse / total)
            coverages = ncovered / total
            sim_cache[target, float(frac), null_name] = (nrmses, coverages)
            lo, hi = np.quantile(rmses, [0.025, 0.975])
            covlo, covhi = np.quantile(coverages, [0.025, 0.975])
            base_rows.append({'target': target, 'target_label': CN[target], 'retained_fraction': frac, 'null_policy': null_name, 'draws': n_draws, 'observed_RMSE': acc_rmse, 'random_mean_RMSE': rmses.mean(), 'random_RMSE_q025': lo, 'random_RMSE_q975': hi, 'observed_gain_vs_random_mean_pct': 100 * (1 - acc_rmse / rmses.mean()), 'fraction_random_RMSE_le_observed': np.mean(rmses <= acc_rmse), 'observed_accepted_coverage': accepted.covered.mean(), 'random_mean_coverage': coverages.mean(), 'random_coverage_q025': covlo, 'random_coverage_q975': covhi, 'observed_below_random_q025': bool(acc_rmse < lo), 'observed_above_random_q975': bool(acc_rmse > hi), 'within_fold_groups': groups, 'randomizable_groups': variable_groups, 'same_bottle_random_priority_across_repeats': True, 'truth_matching_is_posthoc_diagnostic': null_name != 'uniform_same_count'})
    risk_table = dump('selective_review_metrics', risk_rows)
    dump('selective_review_by_repeat', repeat_rows)
    null_table = dump('same_count_random_review_controls', base_rows)
    macro_rows = []
    for frac, g in risk_table.groupby('retained_fraction'):
        for null_name in null_names:
            simulations = np.mean([sim_cache[t, float(frac), null_name][0] for t in CN], axis=0)
            actual = g.accepted_nRMSE.mean()
            lo, hi = np.quantile(simulations, [0.025, 0.975])
            macro_rows.append({'retained_fraction': frac, 'null_policy': null_name, 'actual_mean_retention': g.actual_retention.mean(), 'actual_macro_nRMSE': actual, 'all_macro_nRMSE': g.all_nRMSE.mean(), 'gain_vs_all_pct': 100 * (1 - actual / g.all_nRMSE.mean()), 'random_macro_nRMSE_mean': simulations.mean(), 'random_macro_nRMSE_q025': lo, 'random_macro_nRMSE_q975': hi, 'gain_vs_random_mean_pct': 100 * (1 - actual / simulations.mean()), 'fraction_random_macro_le_observed': np.mean(simulations <= actual), 'draws': n_draws})
    macro = dump('random_review_macro_comparison', macro_rows)
    risk_audit = risks.copy()
    dump('selective_review_records', risk_audit)
    dump('interval_diagnostic_records', intervals)
    sim_path = OUT / 'random_review_simulations.npz'
    np.savez_compressed(sim_path, **{'%s__%.1f__%s' % (t, f, n): x[0] for (t, f, n), x in sim_cache.items()})
    manifest = []
    for path in dict.fromkeys(INPUTS):
        manifest.append({'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    dump('read_only_input_manifest', manifest)
    metrics = {'model_fitting': False, 'originals_modified': False, 'evidence_tier': 'retrospective_internal_only', 'new_analysis_status': 'exploratory_saved_prediction_reanalysis', 'independent_bottles': 139, 'repeat_count': 5, 'fold_count': 5, 'input_count': len(manifest), 'primary_method': 'v10_strategy_router', 'ABV_baseline': 'fixed quadratic features plus Ridge alpha 1; measured test ABV required', 'CI_interpretation': 'paired bottle-cluster bootstrap of saved predictions; conditional on saved fits; no refitting, no redevelopment uncertainty; no multiplicity adjustment', 'random_reference_interpretation': 'Monte Carlo distribution of random policies with exactly matched accepted counts in every outer test fold; shared bottle random priorities across repeats; not a generalization CI or a hypothesis-test p value', 'truth_matching_interpretation': 'posthoc diagnostic; test true concentration is not available as a deployment selector', 'abv_gain_rows': abv_results.to_dict('records'), 'tail_rows': tails[(tails.scale_type == 'adaptive') & (tails.coverage_type == 'marginal')].to_dict('records'), 'risk_macro': macro.to_dict('records'), 'risk70': risk_table[risk_table.retained_fraction.eq(0.7)].to_dict('records'), 'random70': null_table[null_table.retained_fraction.eq(0.7)].to_dict('records')}
    jsonsave('results_numbers', metrics)
    print(json.dumps({'status': 'PASS', 'model_fitting': False, 'bottles': 139, 'output': str(OUT), 'random_draws': n_draws, 'random_macro_70': macro[macro.retained_fraction.eq(0.7)].to_dict('records')}, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
