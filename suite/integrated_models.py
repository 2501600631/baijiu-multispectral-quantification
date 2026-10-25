from __future__ import annotations
import copy
import warnings
import numpy as np
from scipy.linalg import solve
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits
from . import models as legacy
ADDITIONAL_FAMILIES = ('elasticnet', 'gpr', 'knn', 'mlp', 'weighted_block_kernel', 'band_svr', 'distilled_nir')
SUPPORTED_FAMILIES = tuple(legacy.FAMILIES) + ADDITIONAL_FAMILIES

def validate_spec(spec, allow_measured_ethanol=False):
    if not isinstance(spec, dict):
        raise ValueError('A candidate must be a plain specification dictionary')
    allowed = {'family', 'modes', 'pre', 'representation', 'params', 'target_transform', 'use_ethanol'}
    if set(spec) - allowed:
        raise ValueError('Unknown candidate fields: ' + repr(sorted(set(spec) - allowed)))
    family = spec.get('family')
    if family not in SUPPORTED_FAMILIES:
        raise ValueError('Unsupported fixed-library family: ' + str(family))
    if family in ('abv_only', 'abv_augmented') or spec.get('use_ethanol', False):
        if not allow_measured_ethanol:
            raise ValueError('Measured ethanol candidates require a separate input track')
    modes = spec.get('modes', ['nir'])
    if not isinstance(modes, list) or (not modes and family != 'abv_only') or len(modes) != len(set(modes)):
        raise ValueError('Candidate modalities must be a nonempty unique list')
    if set(modes) - {'nir', 'raman', 'uv'}:
        raise ValueError('Only the three supplied spectral modalities are supported')
    if spec.get('representation', 'direct') not in ('direct', 'paired_pls', 'paired_shuffled', 'paired_cca'):
        raise ValueError('Unsupported representation')
    if spec.get('target_transform', 'identity') not in ('identity', 'log1p'):
        raise ValueError('Unsupported target transformation')
    pre = spec.get('pre', 'raw')
    preprocessors = {'raw', 'smooth', 'snv', 'derivative', 'smooth_snv', 'snv_derivative'}
    if isinstance(pre, dict):
        if set(pre) - {'nir', 'raman', 'uv'} or any((value not in preprocessors for value in pre.values())):
            raise ValueError('Invalid per-modality preprocessing')
    elif pre not in preprocessors:
        raise ValueError('Invalid preprocessing')
    if not isinstance(spec.get('params', {}), dict):
        raise ValueError('Candidate parameters must be a dictionary')

    def plain(value):
        if value is None or isinstance(value, (str, bool, int, float)):
            if isinstance(value, float) and (not np.isfinite(value)):
                raise ValueError('Non-finite candidate parameter')
            return
        if isinstance(value, list):
            for item in value:
                plain(item)
            return
        if isinstance(value, dict) and all((isinstance(key, str) for key in value)):
            for item in value.values():
                plain(item)
            return
        raise ValueError('Candidate specification must contain only JSON values')
    plain(spec)
    return copy.deepcopy(spec)

class _BlockKernelModel:

    def _blocks(self, X):
        result = {}
        n = None
        for mode in self.modes:
            if mode not in X:
                raise ValueError('Prediction modality missing: ' + mode)
            block = self.blocks[mode]
            values = block.transform(X[mode]) / np.sqrt(block.n_features_out_)
            if n is not None and len(values) != n:
                raise ValueError('Prediction modalities have different row counts')
            n = len(values)
            result[mode] = values
        return result

    def _kernel(self, first, second):
        result = None
        for weight, mode in zip(self.weights, self.modes):
            a, b = (first[mode], second[mode])
            piece = a @ b.T if self.kernel == 'linear' else rbf_kernel(a, b, gamma=self.gamma)
            result = weight * piece if result is None else result + weight * piece
        return result

    def predict(self, X, ethanol=None):
        with threadpool_limits(limits=1):
            values = self._kernel(self._blocks(X), self.train_blocks) @ self.dual
            prediction = self.target_scaler.inverse_transform(values)
            if self.target_transform == 'log1p':
                with np.errstate(over='ignore', invalid='ignore'):
                    prediction = np.expm1(prediction)
        if not np.isfinite(prediction).all():
            raise ValueError('Non-finite weighted block kernel predictions')
        return prediction

def _weighted_kernel(spec, X, y, seed):
    if spec.get('representation', 'direct') != 'direct':
        raise ValueError('Additive block kernels require a direct representation')
    params = spec.get('params', {})
    result = _BlockKernelModel()
    result.modes = list(spec['modes'])
    result.kernel = params.get('kernel', 'rbf')
    result.gamma = float(params.get('gamma', 1.0))
    alpha = float(params.get('alpha', 0.1))
    if result.kernel not in ('linear', 'rbf') or alpha <= 0 or result.gamma <= 0:
        raise ValueError('Invalid fixed additive-kernel parameters')
    weights = np.asarray(params.get('weights', [1.0 / len(result.modes)] * len(result.modes)), float)
    if weights.shape != (len(result.modes),) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError('Kernel weights must be finite nonnegative and match modalities')
    if abs(float(weights.sum()) - 1.0) > 1e-12:
        raise ValueError('Kernel weights must sum to one')
    result.weights = weights
    result.blocks = {}
    for mode in result.modes:
        pre = spec.get('pre', 'smooth')
        pre = pre.get(mode, 'raw') if isinstance(pre, dict) else pre
        result.blocks[mode] = legacy._BlockTransform(mode, pre, None, seed).fit(X[mode])
    result.train_blocks = result._blocks(X)
    result.target_transform = spec.get('target_transform', 'identity')
    if result.target_transform == 'log1p' and np.any(y < 0):
        raise ValueError('log1p candidates require nonnegative training targets')
    yt = np.log1p(y) if result.target_transform == 'log1p' else y
    result.target_scaler = StandardScaler().fit(yt)
    target = result.target_scaler.transform(yt)
    k = result._kernel(result.train_blocks, result.train_blocks)
    result.dual = solve(k + len(y) * alpha * np.eye(len(y)), target, assume_a='pos')
    result.n_targets = y.shape[1]
    result.metadata = {'spec': copy.deepcopy(spec), 'family': spec['family'], 'training_bottles': len(y), 'n_targets': y.shape[1], 'target_transform': result.target_transform, 'target_scaling': 'training-only StandardScaler', 'input_requirements': {'training_modalities': result.modes, 'prediction_modalities': result.modes, 'measured_ethanol_required': False, 'fitted_rows': len(y)}, 'feature_states': {m: b.state() for m, b in result.blocks.items()}, 'kernel_weights': weights.tolist(), 'kernel_alpha_times_train_n': len(y) * alpha, 'historical_provenance': 'V2/V3 weighted-block-kernel concept under common V10 preprocessing', 'identical_to_old_V3_preprocessing': False, 'prediction_clipping': False, 'all_fitted_states_use_supplied_training_rows_only': True, 'warnings': []}
    return result

class _DistilledNIRModel:

    def predict(self, X, ethanol=None):
        if 'nir' not in X:
            raise ValueError('The distilled student requires NIR at prediction')
        return self.student.predict({'nir': X['nir']})

def _distilled_nir(spec, X, y, seed):
    if spec.get('representation', 'direct') != 'direct':
        raise ValueError('Distilled NIR supports a direct representation only')
    if spec.get('target_transform', 'identity') != 'identity' or spec['modes'] != ['nir']:
        raise ValueError('Distilled NIR requires identity targets and the NIR student input')
    params = spec.get('params', {})
    teacher_modes = list(params.get('teacher_modes', ['nir', 'uv']))
    if not teacher_modes or set(teacher_modes) - {'nir', 'raman', 'uv'} or len(teacher_modes) != len(set(teacher_modes)):
        raise ValueError('Invalid privileged teacher modalities')
    if set(teacher_modes) - set(X):
        raise ValueError('Privileged teacher modalities must be present during fitting')
    strength = float(params.get('distillation_strength', 0.25))
    if strength not in (0.25, 0.5):
        raise ValueError('Distillation strength is fixed at 0.25 or 0.5')
    if len(y) < 8:
        raise ValueError('At least eight fitting bottles are required for teacher OOF')
    folds = 3 if len(y) >= 12 else 2
    teacher = {'family': 'pls2', 'modes': teacher_modes, 'pre': spec.get('pre', 'smooth'), 'representation': 'direct', 'params': {'n_components': 4}}
    student_spec = {'family': 'pls2', 'modes': ['nir'], 'pre': spec.get('pre', 'smooth'), 'representation': 'direct', 'params': {'n_components': 4}}
    pseudo = np.full(y.shape, np.nan)
    membership = np.full(len(y), -1, dtype=int)
    partitions = []
    for fold, (tr, va) in enumerate(KFold(folds, shuffle=True, random_state=int(seed) + 3571).split(y)):
        fit_X = {mode: np.asarray(X[mode])[tr] for mode in teacher_modes}
        valid_X = {mode: np.asarray(X[mode])[va] for mode in teacher_modes}
        model = legacy.fit_model(teacher, fit_X, y[tr], seed=int(seed) + fold)
        prediction = np.asarray(model.predict(valid_X), dtype=float)
        if prediction.shape != y[va].shape or not np.isfinite(prediction).all():
            raise ValueError('Invalid teacher OOF pseudo-labels')
        pseudo[va], membership[va] = (prediction, fold)
        partitions.append({'fold': fold, 'fit': tr.tolist(), 'validation': va.tolist(), 'teacher_train_n': len(tr), 'validation_reference_excluded_from_teacher_fit': True})
    if not np.isfinite(pseudo).all() or np.any(membership < 0):
        raise RuntimeError('Teacher OOF partition incomplete')
    blended = (1.0 - strength) * y + strength * pseudo
    result = _DistilledNIRModel()
    result.student = legacy.fit_model(student_spec, {'nir': X['nir']}, blended, seed=seed)
    result.metadata = {'spec': copy.deepcopy(spec), 'family': 'distilled_nir', 'training_bottles': len(y), 'n_targets': y.shape[1], 'target_transform': 'identity', 'input_requirements': {'training_modalities': sorted(set(teacher_modes + ['nir'])), 'prediction_modalities': ['nir'], 'measured_ethanol_required': False, 'fitted_rows': len(y), 'privileged_training_modalities': teacher_modes}, 'feature_states': result.student.metadata.get('feature_states', {}), 'distillation_strength': strength, 'teacher_spec': teacher, 'student_spec': student_spec, 'teacher_inner_folds': folds, 'teacher_fold_partitions': partitions, 'teacher_oof_membership': membership.tolist(), 'teacher_oof_pseudo_targets': pseudo.tolist(), 'teacher_hyperparameters_selected': False, 'pseudo_targets_are_performance_estimates': False, 'teacher_full_training_refit_needed_for_student_inference': False, 'same_NIR_teacher_control_in_main_pool': False, 'shuffled_teacher_control_in_main_pool': False, 'historical_provenance': 'V5/V6 privileged-view distillation concept under common V10 protocol', 'historical_V6_teacher_student_procedure_exactly_reproduced': False, 'prediction_clipping': False, 'all_fitted_states_use_supplied_training_rows_only': True, 'warnings': []}
    return result

class _BandFeatureTransform:

    def __init__(self, spec, seed):
        self.spec, self.seed = (spec, seed)
        self.modes = list(spec['modes'])

    def _complete(self, X):
        pieces = []
        for mode in self.modes:
            if mode not in X:
                raise ValueError('Prediction modality missing: ' + mode)
            block = self.blocks[mode]
            pieces.append(block.transform(X[mode]) / np.sqrt(block.n_features_out_))
        if len({len(piece) for piece in pieces}) != 1:
            raise ValueError('Spectral modalities have different row counts')
        return np.column_stack(pieces)

    def fit(self, X, y):
        params = self.spec.get('params', {})
        if self.spec.get('representation', 'direct') != 'direct':
            raise ValueError('Band SVR supports direct spectral inputs only')
        self.n_train = len(y)
        self.blocks, self.windows = ({}, {})
        for mode in self.modes:
            pre = self.spec.get('pre', 'smooth')
            pre = pre.get(mode, 'raw') if isinstance(pre, dict) else pre
            self.blocks[mode] = legacy._BlockTransform(mode, pre, None, self.seed).fit(X[mode])
        complete = self._complete(X)
        k = max(1, min(int(params.get('n_components', 8)), len(y) - 1, complete.shape[1]))
        selector = PLSRegression(n_components=k, scale=True, max_iter=1000)
        selector.fit(complete, y)
        coefficient = np.asarray(selector.coef_)
        if coefficient.shape[0] != complete.shape[1]:
            coefficient = coefficient.T
        importance = np.mean(coefficient ** 2, axis=1)
        count = int(params.get('windows_per_mode', 20))
        retain = int(params.get('selected_windows', 2))
        if count < 1 or retain < 1:
            raise ValueError('Band windows and retained count must be positive')
        selected, offset = ([], 0)
        for mode in self.modes:
            width = self.blocks[mode].n_features_out_
            windows = np.array_split(np.arange(width), min(count, width))
            scores = [float(importance[offset + w].mean()) for w in windows]
            chosen = sorted(sorted(range(len(windows)), key=lambda i: (-scores[i], i))[:retain])
            coordinates = np.concatenate([windows[i] for i in chosen])
            selected.extend((offset + coordinates).tolist())
            self.windows[mode] = {'predeclared_window_count': len(windows), 'selected_windows': chosen, 'window_scores': scores, 'selected_coordinate_indices': coordinates.tolist()}
            offset += width
        self.selected = np.asarray(sorted(selected), dtype=int)
        self.scaler = StandardScaler().fit(complete[:, self.selected])
        features = self.scaler.transform(complete[:, self.selected])
        self.pca = None
        pca_requested = params.get('pca_components')
        if pca_requested is not None:
            actual = max(1, min(int(pca_requested), len(y) - 1, features.shape[1]))
            self.pca = PCA(n_components=actual, svd_solver='full', random_state=self.seed).fit(features)
        self.pls_components = k
        return self

    def transform(self, X, ethanol=None):
        features = self.scaler.transform(self._complete(X)[:, self.selected])
        return self.pca.transform(features) if self.pca is not None else features

    def state(self):
        return {'training_modalities': self.modes, 'prediction_modalities': self.modes, 'measured_ethanol_required': False, 'fitted_rows': self.n_train, 'blocks': {mode: block.state() for mode, block in self.blocks.items()}, 'band_selection': self.windows, 'pls_selection_components': self.pls_components, 'selected_feature_count': len(self.selected), 'band_selection_uses_supplied_training_targets_only': True, 'band_interpretation': 'numerical contiguous windows; no chemical peak assignment', 'historical_band_selection_exactly_reproduced': False}

def _extra_regressor(family, params, n, seed):
    if family == 'elasticnet':
        return MultiOutputRegressor(ElasticNet(alpha=float(params.get('alpha', 0.01)), l1_ratio=float(params.get('l1_ratio', 0.5)), max_iter=int(params.get('max_iter', 20000)), tol=1e-05, random_state=seed), n_jobs=1)
    if family == 'gpr':
        kernel = RBF(float(params.get('length_scale', 1.0)), length_scale_bounds='fixed') + WhiteKernel(float(params.get('noise_level', 0.05)), noise_level_bounds='fixed')
        return GaussianProcessRegressor(kernel=kernel, optimizer=None, alpha=1e-06, normalize_y=False, random_state=seed)
    if family == 'knn':
        return KNeighborsRegressor(n_neighbors=max(1, min(int(params.get('n_neighbors', 3)), n)), weights=params.get('weights', 'distance'), n_jobs=1)
    if family == 'mlp':
        return MLPRegressor(hidden_layer_sizes=tuple(params.get('hidden_layer_sizes', [32])), alpha=float(params.get('alpha', 0.1)), solver='lbfgs', max_iter=int(params.get('max_iter', 5000)), max_fun=int(params.get('max_fun', 100000)), tol=1e-05, random_state=seed)
    if family == 'band_svr':
        return MultiOutputRegressor(SVR(C=float(params.get('C', 10.0)), gamma=params.get('gamma', 0.1), epsilon=float(params.get('epsilon', 0.05)), kernel='rbf'), n_jobs=1)
    raise ValueError('Unknown additional family: ' + str(family))

def fit_model(spec, X, Y, ethanol=None, seed=0):
    spec = validate_spec(spec, allow_measured_ethanol=ethanol is not None)
    family = spec['family']
    if family in legacy.FAMILIES:
        return legacy.fit_model(spec, X, Y, ethanol=ethanol, seed=seed)
    y = legacy._matrix(Y, 'Y')
    if len(y) < 3:
        raise ValueError('At least three fitting bottles are required')
    for mode in spec['modes']:
        if mode not in X or len(legacy._matrix(X[mode], mode)) != len(y):
            raise ValueError('All required spectral blocks must match training target rows')
    if family == 'weighted_block_kernel':
        with threadpool_limits(limits=1):
            return _weighted_kernel(spec, X, y, seed)
    if family == 'distilled_nir':
        with threadpool_limits(limits=1):
            return _distilled_nir(spec, X, y, seed)
    transform = spec.get('target_transform', 'identity')
    if transform == 'log1p' and np.any(y < 0):
        raise ValueError('log1p candidates require nonnegative training targets')
    target = np.log1p(y) if transform == 'log1p' else y
    result = legacy.FittedModel()
    result.n_targets, result.target_transform = (y.shape[1], transform)
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result.target_scaler = StandardScaler().fit(target)
        z = result.target_scaler.transform(target)
        if family == 'band_svr':
            result.features = _BandFeatureTransform(spec, seed).fit(X, z)
        else:
            result.features = legacy._FeatureTransform(spec, seed).fit(X, ethanol, len(y))
        features = result.features.transform(X, ethanol)
        regressor = _extra_regressor(family, spec.get('params', {}), len(y), seed)
        if np.all(np.std(z, axis=0) < 1e-12):
            regressor = legacy._Mean()
        fitting_targets = z[:, 0] if family == 'mlp' and z.shape[1] == 1 else z
        result.regressor = regressor.fit(features, fitting_targets)
    convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    if convergence:
        raise RuntimeError('Integrated candidate did not converge: ' + ' | '.join(convergence))
    state = result.features.state()
    result.metadata = {'spec': spec, 'family': family, 'training_bottles': len(y), 'n_targets': y.shape[1], 'target_transform': transform, 'target_scaling': 'training-only StandardScaler', 'input_requirements': state, 'feature_states': state.get('blocks', {}), 'warnings': sorted(set((str(w.message) for w in caught))), 'prediction_clipping': False, 'all_fitted_states_use_supplied_training_rows_only': True, 'historical_provenance': 'V4 algorithm concept under common V10 preprocessing'}
    return result
