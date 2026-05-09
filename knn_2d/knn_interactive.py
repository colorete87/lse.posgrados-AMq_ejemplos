"""
Demo interactivo de k-Nearest Neighbors en 2D.

Funcionalidades:
  - Sliders para # clases (2-6), # train (10-500), # test (5-200), k (1-25).
  - 4 métricas: Euclidiana, Manhattan, Minkowski (p configurable),
    Mahalanobis (Σ global del train).
  - 3 pesos: uniforme, 1/d, gaussiano (h configurable).
  - 3 botones de evaluación: click (single point con vecinos visibles),
    test completo (accuracy en X_test), 5-fold CV.
  - Lista de scores acumulados, regenerar datos, reset todo.
  - Toggle de la región de decisión sobre todo el plano.

Ejecutar:
    uv run python knn_interactive.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, RadioButtons, Button, CheckButtons


RNG = np.random.default_rng()


# ===========================================================
# Distancias (X1: (n1, 2), X2: (n2, 2) -> matriz (n1, n2))
# ===========================================================
def _dist_euclidean(X1, X2):
    diff = X1[:, None, :] - X2[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _dist_manhattan(X1, X2):
    diff = X1[:, None, :] - X2[None, :, :]
    return np.sum(np.abs(diff), axis=-1)


def _dist_minkowski(X1, X2, p=3.0):
    diff = X1[:, None, :] - X2[None, :, :]
    return np.sum(np.abs(diff) ** p, axis=-1) ** (1.0 / p)


def _dist_mahalanobis(X1, X2, VI):
    diff = X1[:, None, :] - X2[None, :, :]   # (n1, n2, 2)
    # cuadrática de Mahalanobis: diff @ VI @ diffᵀ por par
    inner = np.einsum('ijk,kl,ijl->ij', diff, VI, diff)
    return np.sqrt(np.maximum(inner, 0.0))


# ===========================================================
# Pesos (d: array, devuelve mismos shape con weights >= 0)
# ===========================================================
def _w_uniform(d):
    return np.ones_like(d)


def _w_inv_dist(d, eps=1e-16):
    return 1.0 / (d + eps)


def _w_gaussian(d, h=1.0):
    return np.exp(-0.5 * (d / h) ** 2)


# ===========================================================
# Predicción kNN
# ===========================================================
def predict_knn(X_query, X_train, y_train, n_classes, k,
                metric_fn, weight_fn):
    """
    X_query:    (n_query, 2)
    X_train:    (n_train, 2)
    y_train:    (n_train,) enteros 0..n_classes-1
    metric_fn:  f(X1, X2) -> (n1, n2) matriz de distancias
    weight_fn:  f(d) -> array de pesos del mismo shape

    Devuelve:
      pred:       (n_query,) clase predicha
      confidence: (n_query,) peso del voto ganador / suma de pesos
      neighbors:  (n_query, k) índices en X_train de los k vecinos
                  más cercanos (ordenados por distancia)
    """
    X_query = np.atleast_2d(X_query).astype(float)
    n_train = len(X_train)
    k_eff = min(k, n_train)

    D = metric_fn(X_query, X_train)             # (n_query, n_train)
    # k vecinos: índices de los k menores por fila
    idx_part = np.argpartition(D, kth=k_eff - 1, axis=1)[:, :k_eff]
    # ordenarlos por distancia para que neighbors[0] sea el más cercano
    rows = np.arange(len(X_query))[:, None]
    sort_within = np.argsort(D[rows, idx_part], axis=1)
    neighbors = idx_part[rows, sort_within]     # (n_query, k_eff)
    d_nbr = D[rows, neighbors]
    w_nbr = weight_fn(d_nbr)                    # (n_query, k_eff)
    cls_nbr = y_train[neighbors]                # (n_query, k_eff)

    # voto ponderado: sumar pesos por clase
    votes = np.zeros((len(X_query), n_classes))
    for c in range(n_classes):
        mask = (cls_nbr == c)
        votes[:, c] = np.sum(w_nbr * mask, axis=1)

    pred = np.argmax(votes, axis=1)             # empate => menor índice (np.argmax)
    total = np.sum(votes, axis=1)
    confidence = np.where(total > 0,
                          votes[np.arange(len(X_query)), pred] / np.maximum(total, 1e-12),
                          0.0)
    return pred, confidence, neighbors


# ===========================================================
# Generación de datos (n_classes clusters gaussianos 2D)
# ===========================================================
def _random_means(n_classes, rng, x_min=-3.5, x_max=3.5, min_sep=2.0,
                  max_tries=200):
    """Sortea medias para n_classes clusters separadas al menos min_sep."""
    means = []
    tries = 0
    while len(means) < n_classes and tries < max_tries:
        m = rng.uniform(x_min, x_max, size=2)
        if all(np.linalg.norm(m - mm) >= min_sep for mm in means):
            means.append(m)
        tries += 1
    # fallback: si no logra separación, completa con grid uniforme
    while len(means) < n_classes:
        means.append(rng.uniform(x_min, x_max, size=2))
    return np.array(means)


def _random_cov(rng, sigma_lo=0.4, sigma_hi=1.0, rho_max=0.6):
    sx = rng.uniform(sigma_lo, sigma_hi)
    sy = rng.uniform(sigma_lo, sigma_hi)
    rho = rng.uniform(-rho_max, rho_max)
    return np.array([[sx ** 2,        rho * sx * sy],
                     [rho * sx * sy,  sy ** 2]])


def make_random_dataset(n_classes, n_train, n_test, rng=None):
    rng = rng or RNG
    means = _random_means(n_classes, rng)
    covs = [_random_cov(rng) for _ in range(n_classes)]
    Ls = [np.linalg.cholesky(c) for c in covs]

    def sample_n(per_class):
        Xs, ys = [], []
        for c in range(n_classes):
            n = per_class[c]
            if n == 0:
                continue
            z = rng.standard_normal((n, 2))
            Xs.append(z @ Ls[c].T + means[c])
            ys.append(np.full(n, c, dtype=int))
        if not Xs:
            return np.empty((0, 2)), np.empty(0, dtype=int)
        return np.concatenate(Xs), np.concatenate(ys)

    def split(total):
        base, rem = divmod(total, n_classes)
        return [base + (1 if i < rem else 0) for i in range(n_classes)]

    X_train, y_train = sample_n(split(n_train))
    X_test, y_test = sample_n(split(n_test))

    # mezclar para no quedar con clases ordenadas
    perm_tr = rng.permutation(len(X_train))
    perm_te = rng.permutation(len(X_test))
    return (X_train[perm_tr], y_train[perm_tr],
            X_test[perm_te], y_test[perm_te])


# ===========================================================
# K-fold CV (5-fold fijo) sobre el train. Devuelve accuracy promedio.
# ===========================================================
def k_fold_score(X_train, y_train, n_classes, k, metric_fn, weight_fn,
                 n_folds=5, rng=None):
    rng = rng or RNG
    n = len(X_train)
    if n < n_folds:
        return float("nan")
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)
    accs = []
    for i in range(n_folds):
        val_idx = folds[i]
        tr_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        if len(tr_idx) == 0:
            continue
        pred, _, _ = predict_knn(X_train[val_idx], X_train[tr_idx],
                                 y_train[tr_idx], n_classes, k,
                                 metric_fn, weight_fn)
        accs.append(np.mean(pred == y_train[val_idx]))
    return float(np.mean(accs))
