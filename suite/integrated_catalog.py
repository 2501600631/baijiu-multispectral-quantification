from __future__ import annotations
import copy
from .common import COMBINATIONS, digest
from .integrated_models import validate_spec
from .optimized_regression import build_catalog as v9_catalog
from .search import candidates

def build_integrated_catalog(profile, options=None):
    options = {} if options is None else dict(options)
    library, reference_ids = v9_catalog(profile)
    origin = {digest(s): ['exact_V9_specification'] for s in library}

    def add(spec, provenance):
        spec = validate_spec(spec)
        key = digest(spec)
        origin.setdefault(key, []).append(provenance)
        library.append(spec)
    smoke = bool(profile.get('is_smoke'))
    if not smoke:
        expanded_profile = dict(profile, candidates_per_family=max(2, int(profile['candidates_per_family'])))
        existing = candidates(COMBINATIONS['all'], expanded_profile, full=True)
        for family in sorted({s['family'] for s in existing}):
            for spec in [s for s in existing if s['family'] == family][:2]:
                add(spec, 'V7_existing_family_common_protocol')
        for modes in (COMBINATIONS['all'], ['nir']):
            for variant in range(2):
                pre = ['smooth', 'snv'][variant]
                base = {'modes': list(modes), 'pre': pre, 'representation': 'direct'}
                extras = [('elasticnet', {'alpha': [0.01, 0.1][variant], 'l1_ratio': 0.5, 'max_iter': 20000, 'pca_components': 24}), ('gpr', {'length_scale': [1.0, 0.3][variant], 'noise_level': [0.05, 0.1][variant], 'pca_components': 12}), ('knn', {'n_neighbors': [3, 7][variant], 'weights': 'distance', 'pca_components': 12}), ('mlp', {'hidden_layer_sizes': [[32], [32, 16]][variant], 'alpha': [0.1, 1.0][variant], 'max_iter': 5000, 'max_fun': 100000, 'pca_components': 12})]
                for family, params in extras:
                    add(dict(base, family=family, params=params), 'V4_missing_family_common_protocol')
        for modes in COMBINATIONS.values():
            for kernel in ('linear', 'rbf'):
                add({'family': 'weighted_block_kernel', 'modes': list(modes), 'pre': 'smooth', 'representation': 'direct', 'params': {'kernel': kernel, 'alpha': 0.1, 'gamma': 1.0, 'weights': [1.0 / len(modes)] * len(modes)}}, 'V2_V3_additive_kernel_concept_common_protocol')
        for weights in ([0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]):
            add({'family': 'weighted_block_kernel', 'modes': list(COMBINATIONS['all']), 'pre': 'smooth', 'representation': 'direct', 'params': {'kernel': 'rbf', 'alpha': 0.1, 'gamma': 1.0, 'weights': weights}}, 'V2_V3_prespecified_dominant_block_kernel_concept')
        for modes in (COMBINATIONS['all'], COMBINATIONS['raman_uv']):
            for family in ('pls2', 'ridge', 'kernel_ridge'):
                add({'family': family, 'modes': list(modes), 'pre': 'smooth', 'representation': 'direct', 'target_transform': 'log1p', 'params': {'n_components': 4, 'alpha': 1.0, 'pca_components': 24, 'gamma': 0.01, 'kernel': 'rbf'}}, 'V3_V8_positive_target_log_transform_concept')
        for alpha in (0.1, 1.0, 10.0, 100.0):
            add({'family': 'ridge', 'modes': list(COMBINATIONS['nir_uv']), 'pre': 'smooth', 'representation': 'direct', 'params': {'alpha': alpha, 'pca_components': 8}}, 'history_verified_NIR_UV_eight_PC_ridge')
        for modes, pc in ((COMBINATIONS['nir_uv'], 16), (COMBINATIONS['raman_uv'], 24)):
            add({'family': 'svr', 'modes': list(modes), 'pre': 'snv', 'representation': 'direct', 'params': {'C': 10.0, 'gamma': 0.1, 'pca_components': pc, 'epsilon': 0.05}}, 'history_verified_pair_SVR_concept_current_UV_policy')
        for retained, pc, c in ((2, 16, 10.0), (4, 24, 30.0)):
            add({'family': 'band_svr', 'modes': list(COMBINATIONS['raman_uv']), 'pre': 'smooth', 'representation': 'direct', 'params': {'n_components': 8, 'windows_per_mode': 20, 'selected_windows': retained, 'pca_components': pc, 'C': c, 'gamma': 0.1, 'epsilon': 0.05}}, 'historical_PLS_band_selection_SVR_concept_not_exact_old_selection')
        for teacher_modes in (COMBINATIONS['nir_uv'], COMBINATIONS['all']):
            for strength in (0.25, 0.5):
                add({'family': 'distilled_nir', 'modes': ['nir'], 'pre': 'smooth', 'representation': 'direct', 'params': {'teacher_modes': list(teacher_modes), 'distillation_strength': strength}}, 'V5_V6_privileged_view_teacher_OOF_NIR_student_concept')
    else:
        add({'family': 'weighted_block_kernel', 'modes': list(COMBINATIONS['all']), 'pre': 'smooth', 'params': {'kernel': 'linear', 'alpha': 0.1, 'weights': [1.0 / 3] * 3}}, 'smoke_additive_kernel')
    historical = options.get('historical_specs', [])
    if not isinstance(historical, list):
        raise ValueError('historical_specs must be an explicitly frozen JSON list')
    for spec in historical:
        add(spec, 'explicit_historical_specification_under_current_protocol')
    unique = {digest(s): validate_spec(s) for s in library}
    catalog = [unique[key] for key in sorted(unique)]
    families = sorted({s['family'] for s in catalog})
    manifest = {'version': '10.0.0', 'catalog_size': len(catalog), 'families': families, 'family_counts': {f: sum((s['family'] == f for s in catalog)) for f in families}, 'library_fingerprint': digest(catalog), 'candidate_ids': [digest(s) for s in catalog], 'reference_ids': sorted(reference_ids), 'provenance': {key: sorted(set(origin[key])) for key in sorted(unique)}, 'historical_specs_explicitly_added': len(historical), 'outer_targets_used_to_build_library': False, 'calibration_targets_used_to_build_library': False, 'historical_V1_V9_heldout_results_informed_library_design': True, 'label_access_field_scope': 'False above means this builder API receives no current outer-test/calibration array. Historical design was informed by already viewed outcomes from the same bottles; it is not independent confirmation.', 'library_changes_when_optional_package_installed': False, 'optional_boosting_in_default_library': False, 'measured_ethanol_in_spectral_candidate_pool': False, 'old_V3_preprocessing_exactly_reproduced': False, 'interpretation': 'Algorithm concepts re-evaluated under one common retrospective protocol; not stitched old test predictions', 'fit_budget_per_bundle': {'inner_fits': len(catalog) * int(profile['inner_folds']), 'full_refits_upper_bound': len(catalog), 'count_unit': 'candidate fit calls; multioutput and teacher internal fits add cost', 'distilled_candidates': sum((s['family'] == 'distilled_nir' for s in catalog)), 'teacher_subfits_per_distilled_candidate_call': '3, or 2 when fitting n < 12'}}
    return (catalog, set(reference_ids), manifest)
