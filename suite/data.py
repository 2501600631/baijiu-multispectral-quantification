from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from .common import csv_read
import openpyxl
from .common import MODES, sha, digest, save_csv, save_json
TARGETS = ['acetaldehyde', 'methanol', 'n_propanol', 'ethyl_acetate', 'isobutanol', 'acetal', 'n_butanol', 'isoamyl_alcohol', 'ethyl_butyrate', 'ethyl_lactate', 'ethyl_hexanoate']
TARGET_LABELS = ['Acetaldehyde', 'Methanol', 'n-Propanol', 'Ethyl acetate', 'Isobutanol', 'Acetal', 'n-Butanol', 'Isoamyl alcohol', 'Ethyl butyrate', 'Ethyl lactate', 'Ethyl hexanoate']
FILES = {'chemistry': 'chemistry.xlsx', 'nir': 'nir.xlsx', 'raman': 'raman.xlsx', 'uv': 'uv.xlsx'}
RANGES = {'nir': [4000.0, 10000.0], 'raman': [200.0, 1800.0], 'uv': [220.0, 400.0]}
METADATA_COLUMNS = {'Source grade': 'original_grade', 'Coefficient(actual)': 'coefficient_actual', 'Coefficient(60.5)': 'coefficient_60_5', 'Weight': 'weight', 'Adjusted weight': 'adjusted_weight', 'Physical grade': 'physical_grade', 'Supplied acetate to hexanoate ratio': 'ethyl_acetate_hexanoate_ratio_supplied', 'Sensory grade': 'sensory_grade', 'Combined grade': 'combined_grade', 'Retest flag': 'retest_flag'}
GRADE_COLUMNS = ['original_grade', 'physical_grade', 'sensory_grade', 'combined_grade', 'retest_flag']

def _header(value):
    return re.sub('\\s+', ' ', str(value or '')).strip()

def _number(value):
    if isinstance(value, (bool, np.bool_)):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan

def _bottle(value):
    n = _number(value)
    return str(int(n)) if np.isfinite(n) and n > 0 and n.is_integer() else None

def _read_rows(path, sheet=None):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=False)
    try:
        ws = wb[sheet] if sheet else wb.active
        rows = list(ws.values)
        title = ws.title
    finally:
        wb.close()
    return (rows, title)

def parse_chemistry(path):
    rows, sheet = _read_rows(path)
    header_i = None
    for i, row in enumerate(rows[:12]):
        h = [_header(v) for v in row]
        if 'Bottle ID' in h and 'Measured alcohol strength' in h and ('Acetaldehyde(g/L)' in h):
            header_i = i
            break
    if header_i is None:
        raise ValueError('Cannot identify chemistry header by names; no fixed column fallback.')
    raw_headers = list(rows[header_i])
    headers = [_header(v) for v in raw_headers]
    nonempty = [v for v in headers if v]
    if len(set(nonempty)) != len(nonempty):
        raise ValueError('Duplicate chemistry headers after whitespace normalization.')
    columns = {'sample_id': headers.index('Bottle ID'), 'ethanol': headers.index('Measured alcohol strength')}
    aliases = []
    for target, label in zip(TARGETS, TARGET_LABELS):
        accepted = [label + '(g/L)']
        matched = [h for h in accepted if h in headers]
        if len(matched) != 1:
            raise ValueError('Missing or ambiguous analyte header / unit: ' + label)
        columns[target] = headers.index(matched[0])
        if matched[0] != accepted[0]:
            aliases.append({'raw_header': raw_headers[columns[target]], 'canonical_label': label, 'canonical_field': target, 'basis': 'author-confirmed historical typo; numeric values unchanged'})
    for source, field in METADATA_COLUMNS.items():
        if source not in headers:
            raise ValueError('Required author-supplied metadata header missing: ' + source)
        columns[field] = headers.index(source)
    unknown = [h for j, h in enumerate(headers) if h and j not in columns.values()]
    if unknown:
        raise ValueError('Unmapped source columns must be reviewed explicitly: ' + str(unknown))
    records = []
    for row_i, row in enumerate(rows[header_i + 1:], header_i + 2):
        if not any((v is not None for v in row)):
            continue
        if any((isinstance(v, str) and v.startswith('=') for v in row)):
            raise ValueError('Formula found in chemistry row %d; supply verified numeric values.' % row_i)
        sid = _bottle(row[columns['sample_id']])
        if sid is None:
            raise ValueError('Nonempty chemistry row has invalid bottle ID at row %d.' % row_i)
        record = {'sample_id': sid, 'chemistry_row': row_i}
        for field, col in columns.items():
            if field == 'sample_id':
                continue
            value = row[col]
            if field in TARGETS + ['ethanol']:
                value = _number(value)
                if not np.isfinite(value):
                    raise ValueError('Missing/nonnumeric reference value: bottle %s, %s.' % (sid, field))
                if value < 0:
                    raise ValueError('Negative chemical reference value: bottle %s, %s.' % (sid, field))
            elif value is None:
                raise ValueError('Missing author metadata: bottle %s, %s; labels are never imputed.' % (sid, field))
            elif field in GRADE_COLUMNS:
                value = str(value).strip()
                if not value:
                    raise ValueError('Empty metadata category at bottle ' + sid)
            record[field] = value
        records.append(record)
    if not records:
        raise ValueError('No chemistry records.')
    frame = pd.DataFrame(records)
    if frame.sample_id.duplicated().any():
        raise ValueError('Duplicate chemistry bottle IDs: ' + ', '.join(frame.loc[frame.sample_id.duplicated(False), 'sample_id']))
    frame = frame.sort_values('sample_id', key=lambda s: s.astype(int)).reset_index(drop=True)
    supplied_ratio = pd.to_numeric(frame['ethyl_acetate_hexanoate_ratio_supplied'], errors='coerce').to_numpy(float)
    denominator = frame['ethyl_hexanoate'].to_numpy(float)
    calculated_ratio = np.divide(frame['ethyl_acetate'].to_numpy(float), denominator, out=np.full(len(frame), np.nan), where=denominator > 0)
    ratio_difference = supplied_ratio - calculated_ratio
    ratio_mismatch = ~np.isfinite(ratio_difference) | (np.abs(ratio_difference) > 1e-09)
    audit = {'workbook': str(Path(path).name), 'sheet': sheet, 'header_row': header_i + 1, 'raw_headers': raw_headers, 'normalized_headers': headers, 'field_to_source_header': {field: raw_headers[col] for field, col in columns.items()}, 'aliases': aliases, 'formula_count': 0, 'n_bottles': len(frame), 'missing_reference_values': 0, 'duplicate_bottle_ids': 0, 'metadata_distributions': {c: frame[c].value_counts(dropna=False).to_dict() for c in GRADE_COLUMNS}, 'grade_order': None, 'grade_order_inferred': False, 'unknown_grade_provenance_not_inferred': True, 'supplied_ratio_audit': {'formula_for_comparison_only': 'ethyl_acetate / ethyl_hexanoate', 'absolute_tolerance': 1e-09, 'n_not_equal_to_calculated_ratio': int(ratio_mismatch.sum()), 'bottle_ids': frame.loc[ratio_mismatch, 'sample_id'].tolist(), 'max_absolute_difference': float(np.nanmax(np.abs(ratio_difference))), 'raw_source_never_corrected': True, 'supplied_ratio_excluded_from_predictors': True, 'not_a_verified_independent_reference': True}, 'sensory_combined_disagreements': frame.loc[frame.sensory_grade != frame.combined_grade, ['sample_id', 'sensory_grade', 'combined_grade']].to_dict('records'), 'metadata_excluded_from_spectral_predictors': list(METADATA_COLUMNS.values())}
    return (frame, audit)

def parse_spectra(path, sheet=None):
    rows, _ = _read_rows(path, sheet)
    if len(rows) < 12:
        raise ValueError('Insufficient spectral rows: ' + str(path))
    cols = [(j, _bottle(v)) for j, v in enumerate(rows[0]) if j and _bottle(v)]
    ids = [sid for _, sid in cols]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('No valid or duplicate spectral bottle IDs: ' + str(path))
    numeric = []
    for row_i, row in enumerate(rows[1:], 2):
        if any((isinstance(row[j], str) and row[j].startswith('=') for j in [0] + [j for j, _ in cols])):
            raise ValueError('Spectral formula at row %d: %s' % (row_i, path))
        if np.isfinite(_number(row[0])):
            numeric.append(row)
        elif any((row[j] is not None for j, _ in cols)):
            raise ValueError('Nonempty spectral row with missing coordinate: ' + str(path))
    axis = np.array([_number(r[0]) for r in numeric], dtype=float)
    X = np.array([[_number(r[j]) for r in numeric] for j, _ in cols], dtype=float)
    if len(axis) < 11 or len(np.unique(axis)) != len(axis):
        raise ValueError('Duplicate or insufficient spectral coordinates: ' + str(path))
    order = np.argsort(axis)
    axis, X = (axis[order], X[:, order])
    if not np.isfinite(X).all() or np.any(np.std(X, axis=1) < 1e-12):
        raise ValueError('Missing/nonnumeric/flat spectrum; no cross-bottle imputation: ' + str(path))
    return (ids, axis, X)

def parse_raw(raw):
    raw = Path(raw)
    chemistry, chem_audit = parse_chemistry(raw / FILES['chemistry'])
    blocks = {k: parse_spectra(raw / FILES[k], 'Absorbance' if k == 'uv' else None) for k in MODES}
    sets = {'chemistry': set(chemistry.sample_id)}
    sets.update({k: set(blocks[k][0]) for k in MODES})
    ids = sorted(set.intersection(*sets.values()), key=int)
    if not ids:
        raise ValueError('No four-way matched bottle cohort.')
    samples = chemistry.set_index('sample_id').loc[ids].reset_index()
    X = {k: blocks[k][2][[blocks[k][0].index(s) for s in ids]] for k in MODES}
    axes = {k: blocks[k][1] for k in MODES}
    missing = []
    for sid in sorted(set.union(*sets.values()), key=int):
        flags = {k + '_present': sid in v for k, v in sets.items()}
        if not all(flags.values()):
            missing.append(dict(sample_id=sid, **flags, missing_sources=';'.join((k for k, v in sets.items() if sid not in v)), external_validation=False))
    columns = ['sample_id'] + [k + '_present' for k in sets] + ['missing_sources', 'external_validation']
    missing_frame = pd.DataFrame(missing, columns=columns)
    audit = {'experimental_unit': 'bottle_id', 'n_bottles': len(ids), 'n_targets': len(TARGETS), 'n_chemistry_all': len(chemistry), 'chemistry': chem_audit, 'source_bottle_counts': {k: len(v) for k, v in sets.items()}, 'matched_bottle_ids': ids, 'incomplete_source_union_n': len(missing_frame), 'chemistry_without_complete_modalities': sorted(sets['chemistry'] - set(ids), key=int), 'validation_tier': 'retrospective_internal_only', 'no_inferred_dates_or_jar_groups': True, 'no_deleted_or_imputed_labels': True, 'metadata_used_only_as_explicit_targets_or_descriptions': True, 'spectra': {k: {'raw_points': len(axes[k]), 'raw_min': float(axes[k][0]), 'raw_max': float(axes[k][-1]), 'n_bottles': len(blocks[k][0])} for k in MODES}}
    return (chemistry, samples, X, axes, missing_frame, audit)

def build_aligned(root, previous_samples=None):
    root = Path(root)
    chemistry, samples, X, axes, missing, audit = parse_raw(root / 'data/raw')
    if previous_samples is not None:
        old = csv_read(previous_samples, dtype={'sample_id': str})
        merged = old[['sample_id', 'ethanol'] + TARGETS].merge(samples[['sample_id', 'ethanol'] + TARGETS], on='sample_id', suffixes=('_v6', '_v7'), how='inner')
        counts = {c: int(np.count_nonzero(merged[c + '_v6'].to_numpy() != merged[c + '_v7'].to_numpy())) for c in ['ethanol'] + TARGETS}
        audit['previous_v6_comparison'] = {'n_shared_bottles': len(merged), 'old_only_ids': sorted(set(old.sample_id) - set(samples.sample_id), key=int), 'new_only_ids': sorted(set(samples.sample_id) - set(old.sample_id), key=int), 'n_numeric_values_checked': len(merged) * 12, 'difference_counts': counts, 'total_differing_numeric_values': sum(counts.values()), 'cleaning_did_not_create_new_measurements': True}
    aligned = root / 'data/aligned'
    aligned.mkdir(parents=True, exist_ok=True)
    save_csv(aligned / 'samples.csv', samples)
    save_csv(aligned / 'chemistry_all.csv', chemistry)
    save_csv(aligned / 'missing_modalities.csv', missing)
    ratio = chemistry[['sample_id', 'ethyl_acetate', 'ethyl_hexanoate', 'ethyl_acetate_hexanoate_ratio_supplied']].copy()
    ratio['ethyl_acetate_hexanoate_ratio_calculated'] = np.divide(ratio.ethyl_acetate, ratio.ethyl_hexanoate, out=np.full(len(ratio), np.nan), where=ratio.ethyl_hexanoate.to_numpy(float) > 0)
    ratio['supplied_minus_calculated'] = ratio.ethyl_acetate_hexanoate_ratio_supplied - ratio.ethyl_acetate_hexanoate_ratio_calculated
    ratio['equal_within_1e_9'] = np.isfinite(ratio.supplied_minus_calculated) & (ratio.supplied_minus_calculated.abs() <= 1e-09)
    save_csv(aligned / 'supplied_ratio_audit.csv', ratio)
    np.savez_compressed(aligned / 'spectra.npz', sample_id=samples.sample_id.to_numpy(str), **X, **{k + '_axis': v for k, v in axes.items()})
    save_json(aligned / 'audit.json', audit)
    paths = sorted((root / 'data/raw').glob('*.xlsx')) + sorted(aligned.iterdir())
    manifest = {'version': 'V7', 'experimental_unit': 'bottle_id', 'chemistry_source': FILES['chemistry'], 'raw_workbooks_preserved_byte_for_byte': False, 'files': {str(p.relative_to(root)).replace('\\', '/'): sha(p) for p in paths if p.is_file()}, 'independent_bottles': len(samples), 'n_chemistry_all': len(chemistry), 'no_jar_group_inferred': True, 'validation_tier': 'retrospective_internal_only', 'author_confirmed_name_correction': 'Author-confirmed historical analyte spelling corrected during English header translation; numeric values unchanged'}
    save_json(root / 'data/source_manifest.json', manifest)
    return audit

def load_data(root, verify=True):
    root = Path(root)
    manifest = json.loads((root / 'data/source_manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        if sha(root / name) != expected:
            raise ValueError('Frozen input changed: %s. Create another version instead of editing this package.' % name)
    samples = csv_read(root / 'data/aligned/samples.csv', dtype={'sample_id': str})
    chemistry = csv_read(root / 'data/aligned/chemistry_all.csv', dtype={'sample_id': str})
    audit = json.loads((root / 'data/aligned/audit.json').read_text(encoding='utf-8'))
    with np.load(root / 'data/aligned/spectra.npz', allow_pickle=False) as z:
        ids = z['sample_id'].astype(str)
        raw_X = {k: z[k].copy() for k in MODES}
        raw_axes = {k: z[k + '_axis'].copy() for k in MODES}
    if not np.array_equal(ids, samples.sample_id.to_numpy(str)) or not samples.sample_id.is_unique:
        raise ValueError('Chemical/spectral bottle order mismatch or duplicate ID.')
    if verify:
        raw_chem, raw_samples, X0, axes0, missing0, audit0 = parse_raw(root / 'data/raw')
        pd.testing.assert_frame_equal(raw_chem, chemistry, check_dtype=False, atol=1e-12, rtol=0)
        pd.testing.assert_frame_equal(raw_samples, samples, check_dtype=False, atol=1e-12, rtol=0)
        missing_saved = csv_read(root / 'data/aligned/missing_modalities.csv', dtype={'sample_id': str})
        if missing0.to_dict('records') != missing_saved.to_dict('records'):
            raise ValueError('Raw/aligned source coverage differs.')
        for k in MODES:
            if not np.array_equal(axes0[k], raw_axes[k]) or not np.array_equal(X0[k], raw_X[k]):
                raise ValueError('Raw/aligned spectra differ: ' + k)
        if audit0['matched_bottle_ids'] != ids.tolist():
            raise ValueError('Raw/aligned bottle IDs differ.')
    selected, axes = ({}, {})
    for k in MODES:
        low, high = RANGES[k]
        mask = (raw_axes[k] >= low) & (raw_axes[k] <= high)
        a, x = (raw_axes[k][mask], raw_X[k][:, mask])
        if len(a) < 11:
            raise ValueError('Prespecified spectral interval too short: ' + k)
        grid = np.linspace(a[0], a[-1], len(a))
        selected[k] = np.vstack([np.interp(grid, a, row) for row in x])
        axes[k] = grid
        audit['spectra'][k].update({'retained_points': len(grid), 'numeric_interval': [low, high], 'axis_physical_unit': 'pending_author_confirmation', 'interpolation': 'deterministic within-bottle linear to equal-spacing grid; no labels fitted'})
    Y = samples[TARGETS].to_numpy(float)
    ethanol = samples.ethanol.to_numpy(float)
    if not np.isfinite(Y).all() or not np.isfinite(ethanol).all():
        raise ValueError('Invalid reference concentrations; never imputed.')
    if len(ids) != manifest['independent_bottles']:
        raise ValueError('Frozen cohort count mismatch.')
    audit.update({'source_hashes_verified': True, 'aligned_and_raw_equal': bool(verify), 'fingerprint': digest(manifest['files']), 'raw_spectral_interpolation_only': True})
    return {'X': selected, 'Y': Y, 'ethanol': ethanol, 'ids': ids, 'axes': axes, 'target_names': list(TARGETS), 'target_labels': list(TARGET_LABELS), 'data_audit': audit, 'chemistry_all': chemistry}

def synthetic_data(n=60, m=3, seed=731):
    if n < 21 or m < 1:
        raise ValueError('Synthetic tests need n >= 21 and m >= 1.')
    rng = np.random.RandomState(seed)
    latent = rng.normal(size=(n, 5))
    X, axes = ({}, {})
    for k, p in zip(MODES, [31, 25, 37]):
        a = np.linspace(0, 1, p)
        basis = np.vstack([np.sin((j + 1) * np.pi * a) + 0.1 * (j + 1) * a for j in range(5)])
        X[k] = latent @ basis + rng.normal(scale=0.08, size=(n, p))
        axes[k] = a
    Y = np.exp(0.7 + 0.2 * (latent @ rng.normal(size=(5, m))) + rng.normal(scale=0.05, size=(n, m)))
    ids = np.array(['S%03d' % i for i in range(n)])
    names = list(TARGETS) if m == len(TARGETS) else ['synthetic_%d' % j for j in range(m)]
    ethanol = 70 + latent[:, 0]
    grades = np.array(['G%d' % i for i in range(3)])[np.arange(n) % 3]
    chemistry = pd.DataFrame(Y, columns=names)
    chemistry.insert(0, 'sample_id', ids)
    chemistry['ethanol'] = ethanol
    for c in GRADE_COLUMNS:
        chemistry[c] = grades if c != 'retest_flag' else 'synthetic_no_retest'
    chemistry['physical_grade'] = np.array(['P%d' % i for i in range(3)])[(np.arange(n) + 1) % 3]
    return {'X': X, 'Y': Y, 'ethanol': ethanol, 'ids': ids, 'axes': axes, 'target_names': names, 'target_labels': names, 'chemistry_all': chemistry, 'data_audit': {'synthetic': True, 'physical_interpretation': False, 'n_bottles': n, 'n_targets': m, 'validation_tier': 'synthetic_test_only'}}
