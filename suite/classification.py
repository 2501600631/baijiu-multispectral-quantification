import warnings
import numpy as np
from scipy.signal import savgol_filter
from scipy.special import softmax
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from .common import subset

class Features:

    def __init__(self, spec, seed):
        self.spec = spec
        self.seed = seed

    def block(self, a, mode):
        a = np.asarray(a, float).copy()
        pre = self.spec.get('pre', 'smooth')
        if pre != 'raw' and a.shape[1] >= 5:
            win = min(11, a.shape[1] if a.shape[1] % 2 else a.shape[1] - 1)
            deriv = int(pre == 'derivative' and mode != 'uv')
            a = savgol_filter(a, win, min(2, win - 1), deriv=deriv, delta=1, axis=1)
        if pre == 'snv' and mode != 'uv':
            sd = a.std(axis=1, ddof=1, keepdims=True)
            sd = np.where(sd > 1e-12, sd, 1)
            a = (a - a.mean(axis=1, keepdims=True)) / sd
        return a

    def fit_transform(self, X, ethanol=None):
        self.scalers = {}
        self.norm = {}
        blocks = []
        for mode in self.spec.get('modes', []):
            a = self.block(X[mode], mode)
            sc = StandardScaler(with_std=False).fit(a)
            z = sc.transform(a)
            norm = float(np.sqrt(np.mean(z * z))) * np.sqrt(z.shape[1])
            self.scalers[mode] = sc
            self.norm[mode] = max(norm, 1e-12)
            blocks.append(z / self.norm[mode])
        if self.spec.get('use_ethanol', False):
            if ethanol is None:
                raise ValueError('Measured ABV is required by this model')
            a = np.column_stack([ethanol, np.asarray(ethanol) ** 2])
            self.abv_scaler = StandardScaler().fit(a)
            blocks.append(self.abv_scaler.transform(a))
        if not blocks:
            raise ValueError('No declared classification input')
        a = np.hstack(blocks)
        self.pca = None
        nc = self.spec.get('pca_components', 16)
        if nc and a.shape[1] > nc:
            self.pca = PCA(n_components=min(int(nc), len(a) - 1, a.shape[1]), svd_solver='full')
            a = self.pca.fit_transform(a)
        self.scale = StandardScaler().fit(a)
        return self.scale.transform(a)

    def transform(self, X, ethanol=None):
        blocks = [self.scalers[m].transform(self.block(X[m], m)) / self.norm[m] for m in self.spec.get('modes', [])]
        if self.spec.get('use_ethanol', False):
            if ethanol is None:
                raise ValueError('Measured ABV is required')
            blocks.append(self.abv_scaler.transform(np.column_stack([ethanol, np.asarray(ethanol) ** 2])))
        a = np.hstack(blocks)
        if self.pca is not None:
            a = self.pca.transform(a)
        return self.scale.transform(a)

class FittedClassifier:

    def __init__(self, spec, X, y, classes, ethanol, seed):
        self.spec = dict(spec)
        self.classes_ = np.asarray(classes, str)
        self.local = np.unique(np.asarray(y, str))
        self.feature = Features(spec, seed)
        z = self.feature.fit_transform(X, ethanol)
        family = spec['family']
        p = spec.get('params', {})
        self.family = family
        self.estimator = None
        if family == 'majority':
            counts = np.array([np.sum(np.asarray(y, str) == c) for c in self.classes_], float)
            self.prior = counts / counts.sum()
        elif len(self.local) == 1:
            self.prior = np.array([float(c == self.local[0]) for c in self.classes_])
        else:
            if family == 'logistic':
                est = LogisticRegression(C=p.get('C', 1), class_weight='balanced', max_iter=10000, solver='lbfgs', random_state=seed)
            elif family == 'svm':
                est = SVC(C=p.get('C', 3), gamma=p.get('gamma', 'scale'), kernel=p.get('kernel', 'rbf'), class_weight='balanced', probability=False, decision_function_shape='ovr', random_state=seed)
            elif family in ('extra_trees', 'random_forest'):
                cls = ExtraTreesClassifier if family == 'extra_trees' else RandomForestClassifier
                est = cls(n_estimators=p.get('n_estimators', 160), min_samples_leaf=p.get('min_samples_leaf', 2), max_features=p.get('max_features', 1.0), class_weight='balanced', n_jobs=1, random_state=seed)
            elif family == 'hist_gradient_boosting':
                est = HistGradientBoostingClassifier(max_iter=p.get('max_iter', 100), max_leaf_nodes=p.get('max_leaf_nodes', 7), l2_regularization=p.get('l2_regularization', 1), random_state=seed)
            elif family == 'plsda':
                est = PLSRegression(n_components=min(p.get('n_components', 4), len(z) - 1, z.shape[1]), scale=False, max_iter=5000)
            else:
                raise ValueError('Unknown classifier ' + family)
            with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
                warnings.simplefilter('always')
                target = (np.asarray(y, str)[:, None] == self.classes_[None, :]).astype(float) if family == 'plsda' else y
                est.fit(z, target)
            if any((issubclass(w.category, ConvergenceWarning) for w in caught)):
                raise RuntimeError('Classification candidate failed to converge')
            self.estimator = est
        self.metadata = {'family': family, 'classes': self.classes_.tolist(), 'training_n': len(y), 'class_order_assumed': False, 'input_modes': spec.get('modes', []), 'measured_ethanol_required': bool(spec.get('use_ethanol', False)), 'svm_scores_are_uncalibrated': family == 'svm'}

    def predict_proba(self, X, ethanol=None):
        n = len(next(iter(X.values())))
        if self.estimator is None:
            return np.tile(self.prior, (n, 1))
        with threadpool_limits(limits=1):
            z = self.feature.transform(X, ethanol)
            if self.family == 'plsda':
                p = softmax(self.estimator.predict(z), axis=1)
                local = self.classes_
            elif hasattr(self.estimator, 'predict_proba'):
                p = self.estimator.predict_proba(z)
                local = self.estimator.classes_
            else:
                d = self.estimator.decision_function(z)
                if d.ndim == 1:
                    d = np.column_stack([-d, d])
                p = softmax(d, axis=1)
                local = self.estimator.classes_
        answer = np.zeros((n, len(self.classes_)))
        for j, c in enumerate(local):
            answer[:, np.flatnonzero(self.classes_ == c)[0]] = p[:, j]
        return answer

    def predict(self, X, ethanol=None):
        return self.classes_[np.argmax(self.predict_proba(X, ethanol), axis=1)]

def metrics(y, p, classes):
    classes = np.asarray(classes, str)
    pred = classes[np.argmax(p, axis=1)]
    return {'macro_f1': float(f1_score(y, pred, labels=classes, average='macro', zero_division=0)), 'balanced_accuracy': float(balanced_accuracy_score(y, pred)), 'accuracy': float(accuracy_score(y, pred)), 'log_loss': float(log_loss(y, np.maximum(p, 1e-15), labels=classes))}

def catalog(modes, profile, ethanol=False):
    count = profile['candidates_per_family']
    trees = profile['trees']
    fams = ['logistic', 'svm', 'plsda', 'extra_trees', 'random_forest', 'hist_gradient_boosting']
    if profile.get('is_smoke'):
        fams = ['logistic', 'plsda']
    result = []
    for family in fams:
        for i in range(count):
            p = {'C': [0.1, 1, 10, 100][i % 4], 'n_components': [2, 4, 8, 12][i % 4], 'n_estimators': trees, 'min_samples_leaf': [2, 3, 5, 1][i % 4], 'max_leaf_nodes': [3, 7, 15, 7][i % 4], 'l2_regularization': [1, 10, 0.1, 1][i % 4]}
            result.append({'family': family, 'modes': list(modes), 'pre': ['smooth', 'raw', 'snv', 'derivative'][i % 4], 'params': p, 'pca_components': [12, 20, 30, 16][i % 4], 'use_ethanol': ethanol})
    return result

def tune(X, y, classes, candidates, profile, seed, ethanol=None):
    y = np.asarray(y, str)
    counts = [np.sum(y == c) for c in np.unique(y)]
    k = min(profile['inner_folds'], min(counts))
    failures = []
    audit = []
    valid = []
    if k < 2:
        spec = {'family': 'logistic', 'modes': candidates[0].get('modes', []), 'pre': 'smooth', 'pca_components': 12, 'params': {'C': 1}, 'use_ethanol': candidates[0].get('use_ethanol', False)}
        model = FittedClassifier(spec, X, y, classes, ethanol, seed)
        return (model, {'selected': spec, 'selection': 'fixed fallback: inner class count below 2', 'failures': [], 'selected_metrics': None, 'metadata': model.metadata})
    splits = list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))
    for spec in candidates:
        pp = np.zeros((len(y), len(classes)))
        errors = []
        for fold, (tr, va) in enumerate(splits):
            try:
                m = FittedClassifier(spec, subset(X, tr), y[tr], classes, None if ethanol is None else ethanol[tr], seed + fold)
                pp[va] = m.predict_proba(subset(X, va), None if ethanol is None else ethanol[va])
            except Exception as e:
                errors.append({'fold': fold, 'error': repr(e)})
        if errors:
            failures.append({'spec': spec, 'errors': errors})
            continue
        score = metrics(y, pp, classes)
        audit.append({'spec': spec, 'metrics': score})
        valid.append((-score['macro_f1'], -score['balanced_accuracy'], repr(spec), spec))
    if not valid:
        raise RuntimeError('All classification candidates failed: ' + repr(failures))
    for _, __, ___, spec in sorted(valid, key=lambda t: t[:3]):
        try:
            m = FittedClassifier(spec, X, y, classes, ethanol, seed)
            selected_metrics = next((a['metrics'] for a in audit if a['spec'] == spec))
            return (m, {'selected': spec, 'selection': 'training-only stratified macro-F1', 'inner_folds': k, 'candidates': audit, 'failures': failures, 'selected_metrics': selected_metrics, 'metadata': m.metadata})
        except Exception as e:
            failures.append({'spec': spec, 'error': repr(e), 'stage': 'final_fit'})
    raise RuntimeError('All classification refits failed: ' + repr(failures))

def strict_crossfit(X, y, classes, candidates, profile, seed, ethanol=None):
    y = np.asarray(y, str)
    k = min(profile['inner_folds'], min(np.unique(y, return_counts=True)[1]))
    if k < 2:
        raise ValueError('Strict crossfit requires >=2 examples in each class')
    out = np.zeros((len(y), len(classes)))
    audits = []
    for fold, (tr, va) in enumerate(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)):
        model, audit = tune(subset(X, tr), y[tr], classes, candidates, profile, seed + 101 + fold, None if ethanol is None else ethanol[tr])
        out[va] = model.predict_proba(subset(X, va), None if ethanol is None else ethanol[va])
        audits.append(audit)
    return (out, audits)
