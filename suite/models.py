from __future__ import annotations
import copy
import inspect
import itertools
import warnings
import numpy as np
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import MultiTaskElasticNet, Ridge
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits
FAMILIES = ('mean', 'pls1', 'pls2', 'ridge', 'pcr', 'svr', 'kernel_ridge', 'extra_trees', 'random_forest', 'hist_gradient_boosting', 'multitask_elasticnet', 'reduced_rank', 'regressor_chain', 'graph_ridge', 'masked_ridge', 'abv_only', 'abv_augmented')

def _matrix(values, name):
    a = np.asarray(values, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2 or not a.shape[0] or (not a.shape[1]):
        raise ValueError(name + ' must be a nonempty two-dimensional array')
    if not np.isfinite(a).all():
        raise ValueError(name + ' contains non-finite values; values are not imputed')
    return a

def _row_preprocess(a, pre, mode):
    if mode not in ('nir', 'raman', 'uv'):
        return (a.copy(), 'raw')
    if pre not in ('raw', 'smooth', 'snv', 'derivative', 'smooth_snv', 'snv_derivative'):
        raise ValueError('Unknown spectral preprocessing: ' + str(pre))
    effective = pre
    if mode == 'uv' and pre != 'raw':
        effective = 'smooth'
    out = a.copy()
    if effective in ('snv', 'snv_derivative'):
        sd = out.std(axis=1, keepdims=True)
        out = (out - out.mean(axis=1, keepdims=True)) / np.where(sd > 1e-12, sd, 1.0)
    if effective in ('smooth', 'derivative', 'smooth_snv', 'snv_derivative'):
        window = min(11, out.shape[1] if out.shape[1] % 2 else out.shape[1] - 1)
        deriv = 1 if effective in ('derivative', 'snv_derivative') else 0
        if window >= 3:
            out = savgol_filter(out, window, min(2, window - 1), deriv=deriv, axis=1, delta=1.0, mode='interp')
        elif deriv and out.shape[1] >= 2:
            out = np.gradient(out, axis=1)
        elif deriv:
            raise ValueError('Derivative preprocessing needs at least two coordinates')
    if effective == 'smooth_snv':
        sd = out.std(axis=1, keepdims=True)
        out = (out - out.mean(axis=1, keepdims=True)) / np.where(sd > 1e-12, sd, 1.0)
    return (out, effective)

class _BlockTransform:

    def __init__(self, mode, pre, pca_components=None, seed=0):
        self.mode, self.pre, self.requested_pca, self.seed = (mode, pre, pca_components, seed)

    def fit(self, a):
        a = _matrix(a, self.mode)
        self.n_features_in_ = a.shape[1]
        a, self.effective_pre = _row_preprocess(a, self.pre, self.mode)
        self.scaler = StandardScaler().fit(a)
        z = self.scaler.transform(a)
        self.pca = None
        if self.requested_pca is not None:
            k = max(1, min(int(self.requested_pca), len(z) - 1, z.shape[1]))
            self.pca = PCA(n_components=k, svd_solver='full', random_state=self.seed).fit(z)
            z = self.pca.transform(z)
        self.n_features_out_ = z.shape[1]
        return self

    def transform(self, a):
        a = _matrix(a, self.mode)
        if a.shape[1] != self.n_features_in_:
            raise ValueError('Coordinate count changed for ' + self.mode)
        a, _ = _row_preprocess(a, self.pre, self.mode)
        z = self.scaler.transform(a)
        return self.pca.transform(z) if self.pca is not None else z

    def state(self):
        return {'input_features': int(self.n_features_in_), 'output_features': int(self.n_features_out_), 'effective_preprocessing': self.effective_pre, 'pca_components': None if self.pca is None else int(self.pca.n_components_), 'scaler_fitted_rows': int(self.scaler.n_samples_seen_)}

class _FeatureTransform:

    def __init__(self, spec, seed):
        self.spec, self.seed = (spec, seed)
        self.representation = spec.get('representation', 'direct')
        self.modes = list(spec.get('modes', ['nir']))
        if len(set(self.modes)) != len(self.modes):
            raise ValueError('Duplicate modality names')
        self.use_ethanol = bool(spec.get('use_ethanol', False) or spec['family'] in ('abv_only', 'abv_augmented'))
        if spec['family'] == 'abv_only':
            self.modes = []

    def fit(self, X, ethanol, n):
        params = self.spec.get('params', {})
        paired = self.representation in ('paired_pls', 'paired_shuffled', 'paired_cca')
        if self.representation not in ('direct', 'paired_pls', 'paired_shuffled', 'paired_cca'):
            raise ValueError('Unknown representation: ' + self.representation)
        if paired and ('nir' not in self.modes or len(self.modes) < 2):
            raise ValueError('Paired representations require NIR and at least one extra training view')
        if not self.modes and (not self.use_ethanol):
            raise ValueError('At least one input modality or measured ethanol is required')
        self.blocks = {}
        pieces = {}
        for mode in self.modes:
            if mode not in X:
                raise KeyError('Training modality missing: ' + mode)
            a = _matrix(X[mode], mode)
            if len(a) != n:
                raise ValueError('Training modalities and targets have different row counts')
            pre = self.spec.get('pre', 'raw')
            if isinstance(pre, dict):
                pre = pre.get(mode, 'raw')
            k = params.get('pca_components')
            if paired:
                k = params.get('paired_pca_components', k or 20)
            elif self.spec['family'] == 'pcr':
                k = params.get('n_components', k or 8)
            block = _BlockTransform(mode, pre, k, self.seed).fit(a)
            self.blocks[mode] = block
            pieces[mode] = block.transform(a) / np.sqrt(block.n_features_out_)
        self.paired_model = None
        self.paired_components = None
        if paired:
            a = pieces['nir']
            b = np.column_stack([pieces[m] for m in self.modes if m != 'nir'])
            if self.representation == 'paired_shuffled':
                self.shuffle_indices_ = np.random.RandomState(self.seed).permutation(n)
                b = b[self.shuffle_indices_]
            k = max(1, min(int(params.get('representation_components', params.get('n_components', 8))), n - 1, a.shape[1], b.shape[1]))
            if self.representation == 'paired_cca':
                self.paired_model = CCA(n_components=k, scale=True, max_iter=int(params.get('cca_max_iter', 1000)), tol=1e-06)
            else:
                self.paired_model = PLSRegression(n_components=k, scale=True, max_iter=1000, tol=1e-06)
            self.paired_model.fit(a, b)
            self.paired_components = k
        self.ethanol_scaler = None
        if self.use_ethanol:
            e = self._ethanol(ethanol, n)
            self.ethanol_scaler = StandardScaler().fit(e)
        f = self._assemble(pieces, ethanol, n)
        self.output_scaler = StandardScaler().fit(f)
        self.n_train = n
        return self

    @staticmethod
    def _ethanol(ethanol, n):
        if ethanol is None:
            raise ValueError('This model requires separately measured ethanol/ABV at prediction')
        a = _matrix(ethanol, 'measured_ethanol')
        if a.shape != (n, 1):
            raise ValueError('Measured ethanol must have n rows and one column')
        return a

    def _assemble(self, pieces, ethanol, n):
        if self.paired_model is not None:
            f = self.paired_model.transform(pieces['nir'])
        elif self.modes:
            f = np.column_stack([pieces[m] for m in self.modes])
        else:
            f = np.empty((n, 0))
        if self.use_ethanol:
            f = np.column_stack([f, self.ethanol_scaler.transform(self._ethanol(ethanol, n))])
        return f

    def transform(self, X, ethanol=None):
        required = ['nir'] if self.paired_model is not None else self.modes
        pieces = {}
        n = None
        for mode in required:
            if mode not in X:
                raise KeyError('Prediction modality missing: ' + mode)
            z = self.blocks[mode].transform(X[mode])
            if n is not None and len(z) != n:
                raise ValueError('Prediction modalities have different row counts')
            n = len(z)
            pieces[mode] = z / np.sqrt(self.blocks[mode].n_features_out_)
        if n is None:
            if ethanol is None:
                raise ValueError('Measured ethanol is required')
            n = len(np.asarray(ethanol))
        return self.output_scaler.transform(self._assemble(pieces, ethanol, n))

    def state(self):
        return {'blocks': {m: block.state() for m, block in self.blocks.items()}, 'representation': self.representation, 'paired_components': self.paired_components, 'training_modalities': self.modes, 'prediction_modalities': ['nir'] if self.paired_model is not None else self.modes, 'measured_ethanol_required': self.use_ethanol, 'fitted_rows': self.n_train}

class _IndependentPLS:

    def __init__(self, k):
        self.k = k

    def fit(self, x, y):
        self.models = []
        for j in range(y.shape[1]):
            if np.std(y[:, j]) < 1e-12:
                self.models.append(float(np.mean(y[:, j])))
            else:
                m = PLSRegression(n_components=self.k, scale=False, max_iter=1000)
                m.fit(x, y[:, j:j + 1])
                self.models.append(m)
        return self

    def predict(self, x):
        return np.column_stack([np.full(len(x), m) if isinstance(m, float) else np.asarray(m.predict(x)).reshape(-1) for m in self.models])

class _ReducedRank:

    def __init__(self, alpha, rank):
        self.alpha, self.rank = (alpha, rank)

    def fit(self, x, y):
        self.ridge = Ridge(alpha=self.alpha).fit(x, y)
        fitted = self.ridge.predict(x)
        self.center = fitted.mean(axis=0)
        _, _, vt = np.linalg.svd(fitted - self.center, full_matrices=False)
        self.actual_rank = max(1, min(int(self.rank), y.shape[1], vt.shape[0]))
        self.projection = vt[:self.actual_rank].T @ vt[:self.actual_rank]
        return self

    def predict(self, x):
        return self.center + (self.ridge.predict(x) - self.center) @ self.projection

class _GraphRidge:

    def __init__(self, alpha, strength):
        self.alpha, self.strength = (alpha, strength)

    def fit(self, x, y):
        self.x_mean, self.y_mean = (x.mean(axis=0), y.mean(axis=0))
        a, b = (x - self.x_mean, y - self.y_mean)
        if y.shape[1] == 1:
            adjacency = np.zeros((1, 1))
        else:
            adjacency = np.nan_to_num(np.abs(np.corrcoef(y, rowvar=False)), nan=0.0)
            np.fill_diagonal(adjacency, 0.0)
        laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
        eig, vectors = np.linalg.eigh(laplacian)
        eig = np.maximum(eig, 0.0)
        rotated = b @ vectors
        u, s, vt = np.linalg.svd(a, full_matrices=False)
        aty = u.T @ rotated
        coef = np.column_stack([vt.T @ (s / (s * s + self.alpha + self.strength * eig[j]) * aty[:, j]) for j in range(y.shape[1])])
        self.coef = coef @ vectors.T
        self.adjacency = adjacency
        return self

    def predict(self, x):
        return (x - self.x_mean) @ self.coef + self.y_mean

class _Mean:

    def fit(self, x, y):
        self.mean = y.mean(axis=0)
        return self

    def predict(self, x):
        return np.tile(self.mean, (len(x), 1))

def _regressor(family, params, x, y, seed):
    k = max(1, min(int(params.get('n_components', 8)), len(x) - 1, x.shape[1]))
    alpha = float(params.get('alpha', 1.0))
    if alpha <= 0:
        raise ValueError('alpha must be positive')
    if family == 'mean':
        return _Mean()
    if family == 'pls1':
        return _IndependentPLS(k)
    if family == 'pls2':
        return PLSRegression(n_components=k, scale=False, max_iter=1000)
    if family in ('ridge', 'pcr', 'abv_only', 'abv_augmented', 'masked_ridge'):
        return Ridge(alpha=alpha)
    if family == 'svr':
        return MultiOutputRegressor(SVR(C=float(params.get('C', 3.0)), gamma=params.get('gamma', 'scale'), epsilon=float(params.get('epsilon', 0.05)), kernel=params.get('kernel', 'rbf')), n_jobs=1)
    if family == 'kernel_ridge':
        gamma = params.get('gamma', None)
        if gamma == 'scale':
            gamma = 1.0 / (x.shape[1] * max(float(np.var(x)), 1e-12))
        elif gamma == 'auto':
            gamma = 1.0 / x.shape[1]
        return KernelRidge(alpha=alpha, kernel=params.get('kernel', 'rbf'), gamma=gamma)
    if family in ('extra_trees', 'random_forest'):
        cls = ExtraTreesRegressor if family == 'extra_trees' else RandomForestRegressor
        return cls(n_estimators=int(params.get('n_estimators', 200)), min_samples_leaf=int(params.get('min_samples_leaf', 2)), max_depth=params.get('max_depth'), max_features=params.get('max_features', 1.0), n_jobs=1, random_state=seed)
    if family == 'hist_gradient_boosting':
        return MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=min(500, int(params.get('boosting_iterations', params.get('n_estimators', 150)))), learning_rate=float(params.get('learning_rate', 0.05)), max_leaf_nodes=int(params.get('max_leaf_nodes', 7)), max_depth=params.get('max_depth'), min_samples_leaf=int(params.get('min_samples_leaf', 10)), l2_regularization=float(params.get('l2_regularization', 1.0)), early_stopping=False, random_state=seed), n_jobs=1)
    if family == 'multitask_elasticnet':
        return MultiTaskElasticNet(alpha=alpha, l1_ratio=float(params.get('l1_ratio', 0.5)), max_iter=int(params.get('max_iter', 20000)), tol=1e-05, random_state=seed)
    if family == 'reduced_rank':
        return _ReducedRank(alpha, params.get('rank', 3))
    if family == 'graph_ridge':
        strength = float(params.get('graph_strength', 1.0))
        if strength < 0:
            raise ValueError('graph_strength must be nonnegative')
        return _GraphRidge(alpha, strength)
    if family == 'regressor_chain':
        order = params.get('order', None)
        if isinstance(order, list):
            order = np.asarray(order, dtype=int)
        argument = 'estimator' if 'estimator' in inspect.signature(RegressorChain).parameters else 'base_estimator'
        return RegressorChain(order=order, cv=KFold(min(3, len(x)), shuffle=True, random_state=seed), random_state=seed, **{argument: Ridge(alpha=alpha)})
    raise ValueError('Unknown regression family: ' + str(family))

class FittedModel:

    def predict(self, X, ethanol=None):
        with threadpool_limits(limits=1):
            f = self.features.transform(X, ethanol)
            y = np.asarray(self.regressor.predict(f), dtype=float).reshape(len(f), self.n_targets)
            y = self.target_scaler.inverse_transform(y)
            if self.target_transform == 'log1p':
                y = np.expm1(y)
            if not np.isfinite(y).all():
                raise ValueError('Prediction contains non-finite values')
            return y

class _MaskedFeatureTransform:

    def __init__(self, spec, seed):
        self.spec, self.seed = (spec, seed)
        self.modes = list(spec.get('modes', ['nir', 'raman', 'uv']))
        if len(self.modes) < 2 or len(self.modes) > 4 or len(set(self.modes)) != len(self.modes):
            raise ValueError('masked_ridge needs two to four unique modality names')
        if spec.get('representation', 'direct') != 'direct' or spec.get('use_ethanol', False):
            raise ValueError('masked_ridge supports direct views without measured ethanol')

    def fit(self, X, ethanol, n):
        self.blocks = {}
        self.n_train = n
        for mode in self.modes:
            if mode not in X or len(X[mode]) != n:
                raise ValueError('All masked training views must have the training row count')
            pre = self.spec.get('pre', 'smooth')
            if isinstance(pre, dict):
                pre = pre.get(mode, 'raw')
            k = min(8, int(self.spec.get('params', {}).get('pca_components', 8)))
            self.blocks[mode] = _BlockTransform(mode, pre, k, self.seed).fit(X[mode])
        self.masks = [mask for mask in itertools.product((0, 1), repeat=len(self.modes)) if any(mask)]
        self.augmented_rows = n * len(self.masks)
        return self

    def transform(self, X, ethanol=None, mask=None):
        present = [m for m in self.modes if m in X]
        if not present:
            raise ValueError('At least one measured modality is required')
        n = len(X[present[0]])
        blocks, indicators = ([], [])
        for j, mode in enumerate(self.modes):
            available = mode in X and (mask is None or bool(mask[j]))
            block = self.blocks[mode]
            if available:
                z = block.transform(X[mode])
                if len(z) != n:
                    raise ValueError('Prediction modalities have different row counts')
                z = z / np.sqrt(block.n_features_out_)
            else:
                z = np.zeros((n, block.n_features_out_))
            blocks.append(z)
            indicators.append(np.full((n, 1), float(available)))
        return np.column_stack(blocks + indicators)

    def state(self):
        return {'blocks': {m: b.state() for m, b in self.blocks.items()}, 'representation': 'masked_views', 'training_modalities': self.modes, 'prediction_modalities': 'any nonempty subset of training modalities', 'measured_ethanol_required': False, 'fitted_rows': self.n_train, 'augmented_rows': self.augmented_rows, 'nonempty_masks': len(self.masks), 'augmented_rows_are_independent_samples': False}

def fit_model(spec, X, Y, ethanol=None, seed=0):
    spec = copy.deepcopy(spec)
    family = spec.get('family')
    if family not in FAMILIES:
        raise ValueError('Unknown regression family: ' + str(family))
    y = _matrix(Y, 'Y')
    if len(y) < 3:
        raise ValueError('At least three training bottles are required')
    transform = spec.get('target_transform', 'identity')
    if transform not in ('identity', 'log1p'):
        raise ValueError('target_transform must be identity or log1p')
    if transform == 'log1p' and np.any(y < 0):
        raise ValueError('log1p candidates require nonnegative training targets')
    yt = np.log1p(y) if transform == 'log1p' else y
    result = FittedModel()
    result.n_targets = y.shape[1]
    result.target_transform = transform
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result.features = (_MaskedFeatureTransform(spec, seed) if family == 'masked_ridge' else _FeatureTransform(spec, seed)).fit(X, ethanol, len(y))
        result.target_scaler = StandardScaler().fit(yt)
        z = result.target_scaler.transform(yt)
        if family == 'masked_ridge':
            f = np.vstack([result.features.transform(X, mask=mask) for mask in result.features.masks])
            zfit = np.tile(z, (len(result.features.masks), 1))
        else:
            f, zfit = (result.features.transform(X, ethanol), z)
        fit_params = copy.deepcopy(spec.get('params', {}))
        if family == 'masked_ridge':
            fit_params['alpha'] = float(fit_params.get('alpha', 1.0)) * len(result.features.masks)
        regressor = _regressor(family, fit_params, f, zfit, seed)
        if np.all(np.std(zfit, axis=0) < 1e-12):
            regressor = _Mean()
        result.regressor = regressor.fit(f, zfit)
    convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    if convergence:
        raise RuntimeError('Regression/representation candidate did not converge: ' + ' | '.join(convergence))
    state = result.features.state()
    result.metadata = {'spec': spec, 'family': family, 'training_bottles': len(y), 'n_targets': y.shape[1], 'target_transform': transform, 'target_scaling': 'training-only StandardScaler', 'input_requirements': state, 'feature_states': state.get('blocks', {}), 'warnings': sorted(set((str(w.message) for w in caught))), 'prediction_clipping': False, 'all_fitted_states_use_supplied_training_rows_only': True}
    if family == 'regressor_chain':
        result.metadata['chain_training'] = '3-fold internal predicted previous targets, never true previous targets'
    if family == 'graph_ridge':
        result.metadata['target_graph'] = 'absolute Pearson correlations of standardized training targets only'
    if family == 'masked_ridge':
        result.metadata['augmented_rows'] = result.features.augmented_rows
        result.metadata['augmented_rows_are_independent_samples'] = False
        result.metadata['mask_objective'] = 'sum over original bottles of mean loss over nonempty masks plus alpha penalty'
        result.metadata['effective_sklearn_ridge_alpha'] = fit_params['alpha']
    return result
