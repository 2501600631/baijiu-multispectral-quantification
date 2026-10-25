from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from .common import save_csv, save_json, sha
TARGETS = ['acetaldehyde', 'methanol', 'n_propanol', 'ethyl_acetate', 'isobutanol', 'acetal', 'n_butanol', 'isoamyl_alcohol', 'ethyl_butyrate', 'ethyl_lactate', 'ethyl_hexanoate']
LABELS = {'acetaldehyde': 'Acetaldehyde', 'methanol': 'Methanol', 'n_propanol': 'n-Propanol', 'ethyl_acetate': 'Ethyl acetate', 'isobutanol': 'Isobutanol', 'acetal': 'Acetal', 'n_butanol': 'n-Butanol', 'isoamyl_alcohol': 'Isoamyl alcohol', 'ethyl_butyrate': 'Ethyl butyrate', 'ethyl_lactate': 'Ethyl lactate', 'ethyl_hexanoate': 'Ethyl hexanoate', 'ethanol': 'Ethanol (source unit unconfirmed)'}
REFERENCE = 'v10_pls2_reference'
LEGACY = 'v10_v9_unanchored_reference'
ANCHORED = 'v10_strategy_router'
WEIGHT_METHOD = 'v10_anchored_ensemble'
COLORS = {REFERENCE: '#717B84', 'v10_equal_weight': '#BD8C39', 'v10_unanchored_ensemble': '#A46187', ANCHORED: '#176B92', LEGACY: '#77844A', WEIGHT_METHOD: '#528FA8', 'v10_measured_abv_only': '#BD8C39', 'v10_measured_abv_residual': '#A46187', 'v10_measured_abv_guarded': '#525B63'}
MARKERS = {REFERENCE: 'o', 'v10_equal_weight': 's', 'v10_unanchored_ensemble': '^', ANCHORED: 'D', LEGACY: 'P', 'v10_measured_abv_only': 's', 'v10_measured_abv_residual': '^', 'v10_measured_abv_guarded': 'X'}
SHORT = {REFERENCE: 'PLS2 reference', 'v10_shared_catalog_best': 'Shared catalog', 'v10_target_adaptive': 'Target adaptive', 'v10_conservative_adaptive': 'Conservative adaptive', 'v10_equal_weight': 'Equal weights', 'v10_unanchored_ensemble': 'Expanded unanchored', ANCHORED: 'Training-selected router', LEGACY: 'Original V9 unanchored', WEIGHT_METHOD: 'Anchored ensemble', 'v10_single_view_fusion': 'Single-view fusion', 'v10_measured_abv_only': 'Measured ethanol only', 'v10_measured_abv_residual': 'Ethanol-adjusted spectra', 'v10_measured_abv_guarded': 'Ethanol-assisted guard'}
CAPTIONS = {'Fig01_ablation': {'zh': 'Matched-partition comparison of integrated procedures. Left: target-wise training-IQR-normalized RMSE. Right: RMSE reduction of the training-selected strategy router against PLS2; the original V9 unanchored procedure is the reference when only the measured-ethanol module is available. Positive values indicate improvement. Original and expanded candidate pools are distinguished. All selection and weights use training bottles; measured-ethanol-assisted procedures are labeled separately. Intervals are descriptive 95% bottle-cluster bootstrap intervals conditional on saved fits and partitions, excluding redevelopment/refitting uncertainty and multiplicity correction. V10 remains retrospective internal redevelopment after viewing historical results.', 'en': 'Matched-partition comparison of integrated procedures. Left: target-wise training-IQR-normalized RMSE. Right: RMSE reduction of the training-selected strategy router against PLS2; the original V9 unanchored procedure is the reference when only the measured-ethanol module is available. Positive values indicate improvement. Original and expanded candidate pools are distinguished. All selection and weights use training bottles; measured-ethanol-assisted procedures are labeled separately. Intervals are descriptive 95% bottle-cluster bootstrap intervals conditional on saved fits and partitions, excluding redevelopment/refitting uncertainty and multiplicity correction. V10 remains retrospective internal redevelopment after viewing historical results.'}, 'Fig02_intervals': {'zh': 'Coverage versus normalized interval width using matched fit/calibration/test bottles. Left: mean marginal component coverage. Right: simultaneous coverage of the 11-component prediction box. These coverage definitions are distinct. Width is normalized by fitting-set target IQR and averaged equally over components. Color identifies the point model; circles and diamonds indicate fixed and adaptive scales. The dashed line marks nominal 90% coverage, not a conditional guarantee for tails, selected subsets or future batches. Unbounded records are counted separately and retained in source data.', 'en': 'Coverage versus normalized interval width using matched fit/calibration/test bottles. Left: mean marginal component coverage. Right: simultaneous coverage of the 11-component prediction box. These coverage definitions are distinct. Width is normalized by fitting-set target IQR and averaged equally over components. Color identifies the point model; circles and diamonds indicate fixed and adaptive scales. The dashed line marks nominal 90% coverage, not a conditional guarantee for tails, selected subsets or future batches. Unbounded records are counted separately and retained in source data.'}, 'Fig03_selective_risk': {'zh': 'Selective prediction risk. Panels show component nRMSE among retained test bottles versus actual retained fraction, with a common vertical scale. Lines connect prespecified operating points only. Risk models and acceptance thresholds use fitting bottles; test references are used only to calculate reported error. These curves must not be reused to select a favorable test threshold. Rejected samples require an alternative measurement pathway; retained-subset error and coverage cannot be claimed for all bottles.', 'en': 'Selective prediction risk. Panels show component nRMSE among retained test bottles versus actual retained fraction, with a common vertical scale. Lines connect prespecified operating points only. Risk models and acceptance thresholds use fitting bottles; test references are used only to calculate reported error. These curves must not be reused to select a favorable test threshold. Rejected samples require an alternative measurement pathway; retained-subset error and coverage cannot be claimed for all bottles.'}, 'Fig04_numerical_stress': {'zh': 'Numerical sensitivity of frozen models. One modality is changed at a time, and both point models receive the identical perturbed input. Noise and offsets use coordinate-wise training standard deviations; intensity factors multiply the full trace. Values average component-wise RMSE changes against the same unperturbed model (positive means worse). No model is refitted or selected using challenged test references. These simulated challenges are not measured instrument drift, batch transfer or interference validation.', 'en': 'Numerical sensitivity of frozen models. One modality is changed at a time, and both point models receive the identical perturbed input. Noise and offsets use coordinate-wise training standard deviations; intensity factors multiply the full trace. Values average component-wise RMSE changes against the same unperturbed model (positive means worse). No model is refitted or selected using challenged test references. These simulated challenges are not measured instrument drift, batch transfer or interference validation.'}, 'Fig05_window_dependence': {'zh': 'Contiguous-window replacement. Each modality is divided into six prespecified segments. One test segment is replaced by the corresponding outer-training mean, leaving other coordinates and modalities unchanged. Color and labels show the equal-component mean RMSE change, on a shared scale. Numerical coordinates are copied from the input axis; unspecified units are not inferred. Replacement disturbs local covariance and indicates model dependence only, not molecular assignment, chemical mechanism or causal importance.', 'en': 'Contiguous-window replacement. Each modality is divided into six prespecified segments. One test segment is replaced by the corresponding outer-training mean, leaving other coordinates and modalities unchanged. Color and labels show the equal-component mean RMSE change, on a shared scale. Numerical coordinates are copied from the input axis; unspecified units are not inferred. Replacement disturbs local covariance and indicates model dependence only, not molecular assignment, chemical mechanism or causal importance.'}, 'Fig06_weight_stability': {'zh': 'Descriptive stability of anchored ensemble weights. The ten candidates with the largest overall mean training weights are shown for readability; the complete candidate table is retained in source CSV. Panels show the mean and standard deviation across overlapping outer fits. Standard deviations are not confidence intervals from independent replicates. Candidate labels identify family, view set and a short ID. A multiview candidate weight cannot be decomposed into independent modality contributions or interpreted as chemical peak importance. Candidate absence/failure handling follows the report table definition.', 'en': 'Descriptive stability of anchored ensemble weights. The ten candidates with the largest overall mean training weights are shown for readability; the complete candidate table is retained in source CSV. Panels show the mean and standard deviation across overlapping outer fits. Standard deviations are not confidence intervals from independent replicates. Candidate labels identify family, view set and a short ID. A multiview candidate weight cannot be decomposed into independent modality contributions or interpreted as chemical peak importance. Candidate absence/failure handling follows the report table definition.'}}
CAPTIONS['Fig02_intervals']['en'] = 'Matched fit/calibration/test comparison of three spectra-only point procedures: PLS2, original V9 unanchored fusion and the V10 training-selected strategy router. Entire model selection and weighting are refitted inside scale-training OOF folds; calibration bottles determine quantiles only. Left: mean marginal component coverage. Right: simultaneous all-component coverage. Width is normalized by fitting-set IQR and averaged equally over targets. Circles and diamonds denote fixed and adaptive scales. The dashed line marks nominal90% coverage, not conditional coverage of tails, selected subsets or future batches. Unbounded records are retained and counted. No measured-ethanol-assisted intervals are supplied.'
CAPTIONS['Fig04_numerical_stress']['en'] = CAPTIONS['Fig04_numerical_stress']['en'].replace('both point models', 'all three spectra-only point models')
CAPTIONS['Fig06_weight_stability']['zh'] += ' Abbreviations are defined in the English caption and full source tables.'
CAPTIONS['Fig06_weight_stability']['en'] += ' Abbreviations: PLS, partial least squares; KRR, kernel ridge regression; Block KRR, weighted block kernel ridge regression; RF, random forest; EN, elastic net. Full family names and parameters remain in the source tables.'
CAPTIONS['Fig07_strategy_selection'] = {'zh': 'Descriptive stability of per-target procedure selection. Each cell gives the fraction of outer-training tasks selecting a fixed prediction procedure. Entire-procedure predictions are obtained by strict training-only cross-fitting and scored by fitting-IQR-normalized MSE; outer test bottles never select routes. Fractions sum to100% per target. Overlapping training sets make these descriptive frequencies, not independent replication probabilities, significance tests or new validation samples. Procedures are chosen using training predictions rather than stitched historical test minima.', 'en': 'Descriptive stability of per-target procedure selection. Each cell gives the fraction of outer-training tasks selecting a fixed prediction procedure. Entire-procedure predictions are obtained by strict training-only cross-fitting and scored by fitting-IQR-normalized MSE; outer test bottles never select routes. Fractions sum to100% per target. Overlapping training sets make these descriptive frequencies, not independent replication probabilities, significance tests or new validation samples. Procedures are chosen using training predictions rather than stitched historical test minima.'}

class MissingFigure(Exception):
    pass

def _read(out, name, required):
    file = Path(out) / 'tables' / name
    if not file.exists():
        raise MissingFigure('Missing report table: ' + name)
    try:
        frame = pd.read_csv(file, encoding='utf-8-sig', float_precision='round_trip', low_memory=False)
    except pd.errors.EmptyDataError:
        raise MissingFigure('Empty report table: ' + name)
    if frame.empty:
        raise MissingFigure('Empty report table: ' + name)
    absent = set(required) - set(frame)
    if absent:
        raise ValueError('Figure source schema mismatch ' + name + ': ' + repr(sorted(absent)))
    return (frame, file)

def _targets(frame):
    values = set(frame['target'].astype(str)) - {'__macro__'}
    return [t for t in TARGETS if t in values] + sorted(values - set(TARGETS))

def _label(target):
    return LABELS.get(target, target.replace('_', ' '))

def _base(method):
    for base in (REFERENCE, LEGACY, ANCHORED):
        if str(method).startswith(base):
            return base
    raise ValueError('Unknown interval/risk point-model identity: ' + str(method))

def _style():
    return {'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11, 'axes.titleweight': 'semibold', 'axes.edgecolor': '#525B63', 'axes.linewidth': 0.7, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.facecolor': 'white', 'figure.facecolor': 'white', 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'xtick.color': '#343B42', 'ytick.color': '#343B42', 'text.color': '#202830', 'axes.labelcolor': '#202830', 'grid.color': '#E4E8EB', 'grid.linewidth': 0.6, 'legend.frameon': False, 'legend.fontsize': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'}

def _grid(ax, axis='x'):
    ax.grid(axis=axis, zorder=0)
    ax.set_axisbelow(True)

def _ablation(out):
    frame, source = _read(out, 'regression_summary.csv', ['method', 'target', 'nRMSE'])
    gain, gain_source = _read(out, 'paired_comparisons.csv', ['method', 'reference', 'target', 'relative_RMSE_gain_pct'])
    targets = _targets(frame)
    methods = [m for m in [REFERENCE, LEGACY, 'v10_equal_weight', 'v10_unanchored_ensemble', ANCHORED, 'v10_measured_abv_only', 'v10_measured_abv_residual', 'v10_measured_abv_guarded'] if m in set(frame.method)]
    if not targets or not methods:
        raise MissingFigure('No ablation methods available')
    fig, axes = plt.subplots(1, 2, figsize=(12.4, max(5.2, 0.38 * len(targets) + 2)), gridspec_kw={'width_ratios': [1.32, 1.0]})
    y = np.arange(len(targets))
    for i, method in enumerate(methods):
        rows = frame[frame.method == method].set_index('target')
        values = rows.reindex(targets)['nRMSE'].to_numpy(float)
        axes[0].scatter(values, y + (i - (len(methods) - 1) / 2) * 0.115, s=36, color=COLORS[method], marker=MARKERS[method], label=SHORT[method], zorder=3)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([_label(t) for t in targets])
    axes[0].invert_yaxis()
    axes[0].set_xlim(left=0)
    axes[0].set_xlabel('Training-IQR-normalized RMSE')
    axes[0].set_title('A  Component-wise prediction error', loc='left', pad=16)
    _grid(axes[0])
    comparison = REFERENCE if REFERENCE in set(frame.method) else LEGACY
    focal = gain[(gain.method == ANCHORED) & (gain.reference == comparison)]
    for j, target in enumerate(targets):
        row = focal[focal.target == target]
        if row.empty:
            continue
        if len(row) != 1:
            raise ValueError('Duplicate component gain source')
        row = row.iloc[0]
        if 'gain_ci_low_pct' in row and 'gain_ci_high_pct' in row:
            lo, hi = (float(row.gain_ci_low_pct), float(row.gain_ci_high_pct))
            if np.isfinite([lo, hi]).all():
                axes[1].hlines(j, lo, hi, color=COLORS[ANCHORED], linewidth=1.8)
        axes[1].scatter(float(row.relative_RMSE_gain_pct), j, marker='D', s=35, color=COLORS[ANCHORED], zorder=3)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].set_ylim(axes[0].get_ylim())
    axes[1].axvline(0, color='#525B63', ls='--', lw=0.9)
    axes[1].set_xlabel('RMSE reduction versus ' + SHORT[comparison] + ' (%)')
    axes[1].set_title('B  Router gain and descriptive interval', loc='left', pad=16)
    _grid(axes[1])
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3, bbox_to_anchor=(0.55, 1.005))
    fig.subplots_adjust(left=0.16, right=0.985, top=0.81, bottom=0.16, wspace=0.12)
    merged = frame.assign(source_table=source.name)
    merged = pd.concat([merged, gain.assign(source_table=gain_source.name)], ignore_index=True, sort=False)
    return (fig, merged, [source, gain_source], {})

def _intervals(out):
    component, file1 = _read(out, 'interval_summary.csv', ['method', 'target', 'coverage', 'mean_width_iqr'])
    joint, file2 = _read(out, 'interval_joint_summary.csv', ['method', 'joint_coverage', 'mean_width_iqr'])
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.3))
    all_widths, all_coverage = ([], [])
    notes = []
    for ax, kind in zip(axes, ['marginal', 'joint']):
        methods = sorted((m for m in set(component.method) if str(m).endswith('__' + kind)))
        for method in methods:
            rows = component[component.method == method]
            base = _base(method)
            finite_field = 'finite_mean_width_iqr' if 'finite_mean_width_iqr' in rows else 'mean_width_iqr'
            unbounded = int(pd.to_numeric(rows.get('infinite_width_count', pd.Series(0, index=rows.index)), errors='coerce').fillna(0).sum())
            if kind == 'joint':
                selected = joint[joint.method == method]
                if len(selected) != 1:
                    if selected.empty:
                        continue
                    raise ValueError('Duplicate joint interval method summary')
                value = selected.iloc[0]
                width = float(value.get(finite_field, value['mean_width_iqr']))
                coverage = float(value.joint_coverage)
            else:
                width = float(pd.to_numeric(rows[finite_field], errors='coerce').mean())
                coverage = float(pd.to_numeric(rows.coverage, errors='coerce').mean())
            notes.append((method, unbounded))
            if not np.isfinite(width) or not np.isfinite(coverage):
                continue
            all_widths.append(width)
            all_coverage.append(coverage * 100)
            adaptive = '__adaptive__' in method
            label = SHORT[base] + (' / adaptive' if adaptive else ' / fixed')
            ax.scatter(width, coverage * 100, s=70, marker='D' if adaptive else 'o', facecolor=COLORS[base] if adaptive else 'white', edgecolor=COLORS[base], linewidth=1.6, label=label, zorder=4)
        ax.axhline(90, color='#525B63', ls='--', lw=0.9)
        ax.set_xlabel('Mean normalized width (finite records)')
        ax.set_ylim(0, 103)
        ax.set_ylabel('Coverage (%)')
        _grid(ax, 'both')
    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(legend_handles, legend_labels, loc='upper center', fontsize=8, ncol=3, bbox_to_anchor=(0.53, 1.005))
    if not all_widths:
        plt.close(fig)
        raise MissingFigure('No finite interval widths available for display')
    upper = max(all_widths) * 1.15
    coverage_lower = max(0.0, min(85.0, np.floor(min(all_coverage) / 5.0) * 5.0 - 5.0))
    for ax in axes:
        ax.set_xlim(0, upper)
        ax.set_ylim(coverage_lower, 101.0)
    axes[0].set_title('A  Marginal component coverage', loc='left', pad=13)
    axes[1].set_title('B  Simultaneous all-component coverage', loc='left', pad=13)
    ninf = sum((value for _, value in notes))
    fig.text(0.5, 0.025, 'Unbounded component-interval records: %d; all retained in source data.' % ninf, ha='center', fontsize=9, color='#525B63')
    fig.subplots_adjust(left=0.075, right=0.985, top=0.79, bottom=0.18, wspace=0.23)
    data = pd.concat([component.assign(source_table=file1.name), joint.assign(source_table=file2.name)], ignore_index=True, sort=False)
    return (fig, data, [file1, file2], {'unbounded_component_interval_records': ninf, 'coverage_axis_percent_range': [coverage_lower, 101.0], 'unbounded_by_method': dict(notes)})

def _risk(out):
    frame, source = _read(out, 'risk_summary.csv', ['method', 'target', 'actual_retention', 'nRMSE', 'retained_fraction'])
    targets = _targets(frame)
    if not targets:
        raise MissingFigure('No component selective-risk rows')
    columns = 3
    rows = int(np.ceil(len(targets) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(11.8, 2.25 * rows + 0.6), squeeze=False, sharex=True, sharey=True)
    numeric = pd.to_numeric(frame.loc[frame.target.isin(targets), 'nRMSE'], errors='coerce')
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        plt.close(fig)
        raise MissingFigure('No finite selective-risk errors')
    handles, names = ([], [])
    for index, target in enumerate(targets):
        ax = axes.flat[index]
        for method, part in frame[frame.target == target].groupby('method', sort=True):
            part = part.sort_values('actual_retention')
            base = _base(method)
            line, = ax.plot(part.actual_retention.to_numpy(float) * 100, part.nRMSE.to_numpy(float), color=COLORS[base], marker=MARKERS[base], markersize=4, lw=1.4, label=SHORT[base])
            if SHORT[base] not in names:
                names.append(SHORT[base])
                handles.append(line)
        ax.set_title(_label(target), loc='left', fontsize=10, pad=8)
        ax.set_xlim(0, 103)
        ax.set_ylim(0, max(float(finite.max()) * 1.1, 0.01))
        _grid(ax, 'both')
        if index // columns == rows - 1:
            ax.set_xlabel('Actual retention (%)')
        if index % columns == 0:
            ax.set_ylabel('nRMSE')
    for index in range(len(targets), rows * columns):
        axes.flat[index].axis('off')
    fig.legend(handles, names, loc='upper center', ncol=3, bbox_to_anchor=(0.52, 1.005))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.8 if rows == 1 else 0.9, bottom=0.18 if rows == 1 else 0.075, hspace=0.48, wspace=0.17)
    return (fig, frame, [source], {'vertical_scales': 'shared', 'test_threshold_selected': False})

def _challenge_label(meta):
    mode = {'nir': 'NIR', 'raman': 'Raman', 'uv': 'UV'}.get(str(meta['changed_mode']), str(meta['changed_mode']))
    kind = str(meta['challenge_type'])
    value = float(meta['magnitude'])
    if kind == 'gaussian_noise':
        effect = 'noise %.1f%% SD' % (100 * value)
    elif kind == 'training_std_offset':
        effect = 'offset %+.1f%% SD' % (100 * value)
    elif kind == 'intensity_multiplier':
        effect = 'intensity x%.2f' % value
    else:
        effect = kind + ' ' + format(value, '.3g')
    return mode + '  |  ' + effect

def _stress(out):
    frame, source = _read(out, 'stress_summary.csv', ['method', 'target', 'scenario', 'delta_RMSE_pct_vs_clean', 'changed_mode', 'challenge_type', 'magnitude'])
    finite = frame[frame.target != '__macro__'].copy()
    values = finite.groupby(['method', 'scenario'], as_index=False)['delta_RMSE_pct_vs_clean'].mean()
    scenarios = list(dict.fromkeys(finite.sort_values(['changed_mode', 'challenge_type', 'magnitude']).scenario))
    if not scenarios:
        raise MissingFigure('No numerical stress scenarios')
    fig, ax = plt.subplots(figsize=(11.5, max(5.0, 0.31 * len(scenarios) + 1.9)))
    y = np.arange(len(scenarios))
    methods = [m for m in (REFERENCE, LEGACY, ANCHORED) if m in set(values.method)]
    for i, method in enumerate(methods):
        part = values[values.method == method].set_index('scenario').reindex(scenarios)
        ax.scatter(part.delta_RMSE_pct_vs_clean, y + (i - (len(methods) - 1) / 2) * 0.18, s=38, color=COLORS[method], marker=MARKERS[method], label=SHORT[method], zorder=3)
    meta = finite.drop_duplicates('scenario').set_index('scenario')
    ax.set_yticks(y)
    ax.set_yticklabels([_challenge_label(meta.loc[s]) for s in scenarios])
    ax.invert_yaxis()
    ax.axvline(0, color='#525B63', ls='--', lw=0.9)
    _grid(ax)
    ax.set_xlabel('Mean component RMSE change versus clean input (%)')
    ax.set_title('Prediction-only numerical challenges', loc='left', pad=17)
    fig.legend(*ax.get_legend_handles_labels(), loc='upper center', ncol=3, bbox_to_anchor=(0.59, 1.005))
    fig.subplots_adjust(left=0.32, right=0.98, top=0.8, bottom=0.12)
    return (fig, frame, [source], {'equal_weight_over_components': True, 'real_instrument_drift_validation': False})

def _occlusion(out):
    frame, source = _read(out, 'occlusion_summary.csv', ['method', 'target', 'scenario', 'delta_RMSE_pct_vs_clean', 'changed_mode', 'window_index', 'coordinate_start', 'coordinate_end'])
    valid = frame[frame.target != '__macro__'].copy()
    modes = [m for m in ['nir', 'raman', 'uv'] if m in set(valid.changed_mode)]
    if not modes:
        raise MissingFigure('No coordinate-window scenarios')
    values = valid.groupby(['method', 'changed_mode', 'window_index'], as_index=False)['delta_RMSE_pct_vs_clean'].mean()
    numeric = values.delta_RMSE_pct_vs_clean.to_numpy(float)
    finite = numeric[np.isfinite(numeric)]
    if not len(finite):
        raise MissingFigure('No finite coordinate-window response')
    bound = max(np.max(np.abs(finite)), 0.1)
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    methods = [m for m in (REFERENCE, LEGACY, ANCHORED) if m in set(valid.method)]
    fig, axes = plt.subplots(len(modes), 1, figsize=(11.8, 2.4 * len(modes) + 0.5), squeeze=False)
    for ax, mode in zip(axes.flat, modes):
        selected = values[values.changed_mode == mode]
        windows = sorted(selected.window_index.unique())
        matrix = selected.pivot(index='method', columns='window_index', values='delta_RMSE_pct_vs_clean').reindex(index=methods, columns=windows)
        im = ax.imshow(matrix.to_numpy(float), cmap='PuOr_r', norm=norm, aspect='auto', interpolation='nearest')
        ax.set_yticks(np.arange(len(methods)))
        ax.set_yticklabels([SHORT[m] for m in methods])
        metadata = valid[valid.changed_mode == mode].drop_duplicates('window_index').set_index('window_index')
        ticks = ['%.4g to %.4g' % (metadata.loc[w, 'coordinate_start'], metadata.loc[w, 'coordinate_end']) for w in windows]
        ax.set_xticks(np.arange(len(windows)))
        ax.set_xticklabels(ticks, fontsize=8)
        ax.set_title({'nir': 'NIR', 'raman': 'Raman', 'uv': 'UV'}[mode] + ' | numerical coordinate windows', loc='left', fontsize=10)
        for i in range(len(methods)):
            for j in range(len(windows)):
                value = matrix.iloc[i, j]
                if np.isfinite(value):
                    ax.text(j, i, '%+.1f' % value, ha='center', va='center', fontsize=9, color='white' if abs(value) > bound * 0.6 else '#202830')
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(left=0.24, right=0.86, top=0.93, bottom=0.11, hspace=0.8)
    cax = fig.add_axes([0.9, 0.19, 0.018, 0.62])
    fig.colorbar(im, cax=cax, label='Mean component RMSE change (%)')
    return (fig, frame, [source], {'causal_assignment': False, 'axis_units_inferred': False, 'common_symmetric_color_range_pct': [-float(bound), float(bound)]})

def _weights(out):
    frame, source = _read(out, 'weight_stability.csv', ['method', 'target', 'candidate_id', 'mean_weight', 'std_weight', 'family', 'modes'])
    anchored = frame[(frame.method == WEIGHT_METHOD) & (frame.target != '__macro__')].copy()
    if anchored.empty:
        raise MissingFigure('No anchored candidate weights')
    ordered = anchored.groupby('candidate_id').mean_weight.mean().sort_values(ascending=False)
    chosen = list(ordered.index[:10])
    targets = _targets(anchored)
    mean = anchored.pivot(index='target', columns='candidate_id', values='mean_weight').reindex(index=targets, columns=chosen)
    std = anchored.pivot(index='target', columns='candidate_id', values='std_weight').reindex(index=targets, columns=chosen)
    meta = anchored.drop_duplicates('candidate_id').set_index('candidate_id')
    names = []
    family_labels = {'pls2': 'PLS2', 'pls1': 'PLS1', 'ridge': 'Ridge', 'kernel_ridge': 'KRR', 'weighted_block_kernel': 'Block KRR', 'graph_ridge': 'Graph ridge', 'multitask_elasticnet': 'Multi-task EN', 'hist_gradient_boosting': 'Hist. boosting', 'random_forest': 'RF', 'extra_trees': 'ExtraTrees', 'reduced_rank': 'Reduced rank', 'regressor_chain': 'Regressor chain', 'distilled': 'Distillation', 'svr': 'SVR', 'pcr': 'PCR'}
    for candidate in chosen:
        row = meta.loc[candidate]
        mode = str(row['modes']).replace('nir', 'NIR').replace('raman', 'Raman').replace('uv', 'UV')
        family = family_labels.get(str(row['family']), str(row['family']).replace('_', ' '))
        names.append(family + ' | ' + mode + '\n' + str(candidate)[:6])
    fig, axes = plt.subplots(1, 2, figsize=(max(12.5, len(chosen) * 1.2), max(6.0, len(targets) * 0.34 + 2.6)))
    vmax_std = max(float(np.nanmax(std.to_numpy(float))), 0.001)
    for ax, matrix, vmax, title in [(axes[0], mean, 1.0, 'A  Mean candidate weight'), (axes[1], std, vmax_std, 'B  Across-fit standard deviation')]:
        image = ax.imshow(matrix.to_numpy(float), cmap='Blues', vmin=0, vmax=vmax, aspect='auto', interpolation='nearest')
        ax.set_xticks(np.arange(len(chosen)))
        ax.set_xticklabels(names, rotation=55, ha='right', fontsize=7.5)
        ax.set_yticks(np.arange(len(targets)))
        ax.set_yticklabels([_label(t) for t in targets] if ax is axes[0] else [])
        ax.set_title(title, loc='left', pad=13)
        fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.9, bottom=0.29, wspace=0.13)
    frame = frame.copy()
    frame['displayed_candidate'] = frame.candidate_id.isin(chosen) & (frame.method == WEIGHT_METHOD)
    return (fig, frame, [source], {'displayed_top_candidates': chosen, 'selection_for_display_only': 'overall_mean_training_weight', 'standard_deviation_is_confidence_interval': False})

def _strategy_selection(out):
    frame, source = _read(out, 'strategy_selection_stability.csv', ['target', 'selected_procedure', 'selection_fraction', 'selection_count'])
    targets = _targets(frame)
    methods = [m for m in [REFERENCE, LEGACY, 'v10_single_view_fusion', 'v10_equal_weight', 'v10_unanchored_ensemble', WEIGHT_METHOD, 'v10_conservative_adaptive', 'v10_target_adaptive', 'v10_shared_catalog_best'] if m in set(frame.selected_procedure)]
    if not targets or not methods:
        raise MissingFigure('No saved training strategy routes')
    matrix = frame.pivot(index='target', columns='selected_procedure', values='selection_fraction').reindex(index=targets, columns=methods).fillna(0.0)
    fig, ax = plt.subplots(figsize=(max(9.0, len(methods) * 1.25 + 3.0), max(4.6, len(targets) * 0.4 + 2.5)))
    image = ax.imshow(matrix.to_numpy(float) * 100, cmap='Blues', vmin=0, vmax=100, aspect='auto', interpolation='nearest')
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels([SHORT[m].replace(' ', '\n', 1) for m in methods], fontsize=8)
    ax.set_yticks(np.arange(len(targets)))
    ax.set_yticklabels([_label(t) for t in targets])
    ax.set_title('Training-only procedure selection across overlapping outer fits', loc='left', pad=16)
    for i in range(len(targets)):
        for j in range(len(methods)):
            value = float(matrix.iloc[i, j]) * 100
            if value > 0:
                ax.text(j, i, '%.0f%%' % value, ha='center', va='center', color='white' if value > 55 else '#202830', fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025, label='Outer-training selection fraction (%)')
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.23, right=0.94, bottom=0.22, top=0.86)
    return (fig, frame, [source], {'selection_uses_test_reference': False, 'independent_replicate_frequencies': False, 'historical_test_scores_spliced': False})

def draw_all(out):
    out = Path(out)
    dest = out / 'figures'
    dest.mkdir(parents=True, exist_ok=True)
    source_dir = dest / 'source_data'
    source_dir.mkdir(parents=True, exist_ok=True)
    definitions = [('Fig01_ablation', _ablation), ('Fig02_intervals', _intervals), ('Fig03_selective_risk', _risk), ('Fig04_numerical_stress', _stress), ('Fig05_window_dependence', _occlusion), ('Fig06_weight_stability', _weights), ('Fig07_strategy_selection', _strategy_selection)]
    result = {'model_fitting': False, 'input_directory': str(out.resolve()), 'style': 'static scientific vector PDF and 600dpi PNG', 'figures': [], 'skipped': [], 'code_sha256': {}}
    root = Path(__file__).resolve().parents[1]
    for file in [Path(__file__), root / 'figures_code' / 'redraw_all.py']:
        if file.exists():
            result['code_sha256'][str(file.relative_to(root)).replace('\\', '/')] = sha(file)
    with plt.rc_context(_style()):
        for name, builder in definitions:
            try:
                fig, source_data, input_files, details = builder(out)
            except MissingFigure as exc:
                result['skipped'].append({'figure': name, 'reason': str(exc)})
                continue
            png, pdf = (dest / (name + '.png'), dest / (name + '.pdf'))
            try:
                fig.savefig(png, dpi=600, bbox_inches='tight', pad_inches=0.13)
                fig.savefig(pdf, bbox_inches='tight', pad_inches=0.13)
            finally:
                plt.close(fig)
            file = source_dir / (name + '_source.csv')
            save_csv(file, source_data)
            result['figures'].append({'id': name, 'png': str(png.relative_to(out)), 'pdf': str(pdf.relative_to(out)), 'source_csv': str(file.relative_to(out)), 'caption': CAPTIONS[name], 'details': details, 'input_sha256': {str(f.relative_to(out)).replace('\\', '/'): sha(f) for f in input_files}, 'output_sha256': {str(f.relative_to(out)).replace('\\', '/'): sha(f) for f in [png, pdf, file]}})
    result['generated_figure_count'] = len(result['figures'])
    result['skipped_figure_count'] = len(result['skipped'])
    save_json(dest / 'figure_manifest.json', result)
    caption_lines = ['# V10 figure captions', '', 'Saved results only; no model fitting.', '']
    for fig in result['figures']:
        caption_lines.extend(['## ' + fig['id'], '', fig['caption']['zh'], '', fig['caption']['en'], ''])
    for skipped in result['skipped']:
        caption_lines.extend(['Skipped ' + skipped['figure'] + ': ' + skipped['reason'], ''])
    (dest / 'figure_captions.md').write_text('\n'.join(caption_lines), encoding='utf-8')
    return result
