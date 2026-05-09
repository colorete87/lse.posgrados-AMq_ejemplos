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


# ===========================================================
# Registries y defaults
# ===========================================================
METRICS = ["Euclidiana", "Manhattan", "Minkowski", "Mahalanobis"]
WEIGHTS = ["uniform", "1/d", "gaussiano"]


def _build_metric_fn(name, X_train, p, mahal_VI):
    if name == "Euclidiana":
        return _dist_euclidean
    if name == "Manhattan":
        return _dist_manhattan
    if name == "Minkowski":
        return lambda X1, X2: _dist_minkowski(X1, X2, p=p)
    if name == "Mahalanobis":
        return lambda X1, X2: _dist_mahalanobis(X1, X2, VI=mahal_VI)
    raise ValueError(name)


def _build_weight_fn(name, h):
    if name == "uniform":
        return _w_uniform
    if name == "1/d":
        return _w_inv_dist
    if name == "gaussiano":
        return lambda d: _w_gaussian(d, h=h)
    raise ValueError(name)


def _compute_mahal_VI(X_train):
    if len(X_train) < 3:
        return np.eye(2)
    cov = np.cov(X_train.T) + 1e-6 * np.eye(2)
    try:
        return np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return np.eye(2)


X_MIN, X_MAX = -5.0, 5.0
GRID = 60
gxs = np.linspace(X_MIN, X_MAX, GRID)
GX1, GX2 = np.meshgrid(gxs, gxs)
GRID_PTS = np.column_stack([GX1.ravel(), GX2.ravel()])

DEFAULTS = {
    "n_classes": 3,
    "n_train": 150,
    "n_test": 50,
    "k": 5,
    "metric": "Euclidiana",
    "minkowski_p": 3.0,
    "weights": "uniform",
    "gaussian_h": 1.0,
    "show_decision": True,
    "show_test": True,
}

state = dict(DEFAULTS)
state.update({
    "X_train": np.empty((0, 2)),
    "y_train": np.empty(0, dtype=int),
    "X_test": np.empty((0, 2)),
    "y_test": np.empty(0, dtype=int),
    "y_test_pred": None,
    "mahal_VI": np.eye(2),
    "scores": [],          # list[dict]
    "last_query": None,    # dict | None
})


def _regenerate_data():
    """Resortea X_train, y_train, X_test, y_test y recalcula mahal_VI.
    No toca scores ni last_query (eso lo hace el caller)."""
    Xtr, ytr, Xte, yte = make_random_dataset(
        state["n_classes"], state["n_train"], state["n_test"], RNG)
    state["X_train"] = Xtr
    state["y_train"] = ytr
    state["X_test"] = Xte
    state["y_test"] = yte
    state["y_test_pred"] = None
    state["mahal_VI"] = _compute_mahal_VI(Xtr)


# ===========================================================
# Figura
# ===========================================================
fig = plt.figure(figsize=(13, 7.5))
ax = fig.add_axes([0.27, 0.36, 0.50, 0.58])
ax_scores = fig.add_axes([0.80, 0.36, 0.18, 0.58])
ax_scores.set_xticks([])
ax_scores.set_yticks([])
ax_scores.set_facecolor("#fafafa")
ax_scores.set_title("Scores", fontsize=10, weight="bold")

ax.set_aspect("equal")
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(X_MIN, X_MAX)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.grid(alpha=0.3)
ax.set_title("k-NN interactivo (2D)", loc="left")

CLASS_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def _add_group_box(x, y, w, h, label):
    rect = Rectangle((x, y), w, h, transform=fig.transFigure,
                     linewidth=0.9, edgecolor="#666666",
                     facecolor="none", zorder=0)
    fig.add_artist(rect)
    return fig.text(x + w / 2, y + h + 0.006, label,
                    ha="center", va="bottom", fontsize=9, weight="bold",
                    color="#333333")


# Artistas que reusamos en cada redraw
train_scatter = ax.scatter([], [], s=40, edgecolors="black", linewidths=0.5)
test_scatter = ax.scatter([], [], s=40, facecolors="none", linewidths=1.2)
query_marker, = ax.plot([], [], "x", color="black", ms=14, mew=2.5, zorder=12)
neighbor_lines = []   # lista de artistas Line2D que conectan query con vecinos
decision_image = [None]  # AxesImage opcional


def _safe_remove(artist):
    try:
        artist.remove()
    except Exception:
        pass


def _clear_neighbor_lines():
    for ln in neighbor_lines:
        _safe_remove(ln)
    neighbor_lines.clear()


def redraw():
    # 1. Región de decisión
    _safe_remove(decision_image[0])
    decision_image[0] = None
    if state["show_decision"] and len(state["X_train"]) > 0:
        metric_fn = _build_metric_fn(state["metric"], state["X_train"],
                                      state["minkowski_p"], state["mahal_VI"])
        weight_fn = _build_weight_fn(state["weights"], state["gaussian_h"])
        pred_grid, _, _ = predict_knn(GRID_PTS, state["X_train"], state["y_train"],
                                       state["n_classes"], state["k"],
                                       metric_fn, weight_fn)
        # imshow con colormap personalizado por clase
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(CLASS_COLORS[:state["n_classes"]])
        decision_image[0] = ax.imshow(
            pred_grid.reshape(GRID, GRID),
            extent=(X_MIN, X_MAX, X_MIN, X_MAX),
            origin="lower", cmap=cmap, alpha=0.18,
            vmin=0, vmax=state["n_classes"] - 1, interpolation="nearest", zorder=0)

    # 2. Train + test
    if len(state["X_train"]) > 0:
        colors_tr = [CLASS_COLORS[c] for c in state["y_train"]]
        train_scatter.set_offsets(state["X_train"])
        train_scatter.set_facecolors(colors_tr)
    else:
        train_scatter.set_offsets(np.empty((0, 2)))

    if state["show_test"] and len(state["X_test"]) > 0:
        edge_colors = [CLASS_COLORS[c] for c in state["y_test"]]
        test_scatter.set_offsets(state["X_test"])
        test_scatter.set_edgecolors(edge_colors)
    else:
        test_scatter.set_offsets(np.empty((0, 2)))

    # 3. Query y vecinos del último click
    _clear_neighbor_lines()
    if state["last_query"] is not None and len(state["X_train"]) > 0:
        q = state["last_query"]
        query_marker.set_data([q["x"]], [q["y"]])
        for idx in q["neighbors"]:
            xs = [q["x"], state["X_train"][idx, 0]]
            ys = [q["y"], state["X_train"][idx, 1]]
            ln, = ax.plot(xs, ys, color="black", lw=0.7, alpha=0.6, zorder=11)
            neighbor_lines.append(ln)
        ax.set_title(
            f"k-NN interactivo (2D)  —  query=({q['x']:+.2f}, {q['y']:+.2f})  "
            f"pred=clase {q['pred']}  conf={q['conf']:.2f}",
            loc="left", fontsize=10)
    else:
        query_marker.set_data([], [])
        ax.set_title("k-NN interactivo (2D)", loc="left", fontsize=10)

    fig.canvas.draw_idle()


_regenerate_data()
redraw()

# ===========================================================
# Sliders básicos (# clases, # train, # test, k)
# ===========================================================
ax_n_classes = plt.axes([0.06, 0.26, 0.18, 0.020])
sl_n_classes = Slider(ax_n_classes, "# clases", 2, 6, valinit=DEFAULTS["n_classes"],
                       valstep=1)

ax_n_train = plt.axes([0.06, 0.21, 0.18, 0.020])
sl_n_train = Slider(ax_n_train, "# train", 10, 500, valinit=DEFAULTS["n_train"],
                     valstep=10)

ax_n_test = plt.axes([0.06, 0.16, 0.18, 0.020])
sl_n_test = Slider(ax_n_test, "# test", 5, 200, valinit=DEFAULTS["n_test"],
                    valstep=5)

ax_k = plt.axes([0.06, 0.11, 0.18, 0.020])
sl_k = Slider(ax_k, "k", 1, 25, valinit=DEFAULTS["k"], valstep=1)


def _on_data_slider(_v):
    state["n_classes"] = int(sl_n_classes.val)
    state["n_train"] = int(sl_n_train.val)
    state["n_test"] = int(sl_n_test.val)
    _regenerate_data()
    state["scores"].clear()
    state["last_query"] = None
    redraw()


def _on_k(_v):
    state["k"] = int(sl_k.val)
    redraw()


sl_n_classes.on_changed(_on_data_slider)
sl_n_train.on_changed(_on_data_slider)
sl_n_test.on_changed(_on_data_slider)
sl_k.on_changed(_on_k)

# ===========================================================
# Recuadro Vista
# ===========================================================
_add_group_box(0.025, 0.83, 0.21, 0.10, "Vista")
ax_check = plt.axes([0.04, 0.835, 0.18, 0.085])
ax_check.set_facecolor("none")
chk = CheckButtons(ax_check,
                    ["Mostrar región de decisión", "Mostrar test points"],
                    actives=[DEFAULTS["show_decision"], DEFAULTS["show_test"]])


def _on_check(label):
    if label == "Mostrar región de decisión":
        state["show_decision"] = not state["show_decision"]
    elif label == "Mostrar test points":
        state["show_test"] = not state["show_test"]
    redraw()


chk.on_clicked(_on_check)

# ===========================================================
# Click handler: evalúa un punto, dibuja vecinos y caption
# ===========================================================
def on_click(event):
    if event.inaxes != ax:
        return
    if event.xdata is None or event.ydata is None:
        return
    x = float(event.xdata); y = float(event.ydata)
    metric_fn = _build_metric_fn(state["metric"], state["X_train"],
                                  state["minkowski_p"], state["mahal_VI"])
    weight_fn = _build_weight_fn(state["weights"], state["gaussian_h"])
    pred, conf, nbr = predict_knn(np.array([[x, y]]), state["X_train"],
                                   state["y_train"], state["n_classes"],
                                   state["k"], metric_fn, weight_fn)
    state["last_query"] = {
        "x": x, "y": y,
        "pred": int(pred[0]),
        "conf": float(conf[0]),
        "neighbors": nbr[0].tolist(),
    }
    redraw()


fig.canvas.mpl_connect("button_press_event", on_click)

# ===========================================================
# Recuadro Métrica
# ===========================================================
_add_group_box(0.30, 0.04, 0.32, 0.255, "Métrica")
ax_metric = plt.axes([0.31, 0.07, 0.10, 0.20])
radio_metric = RadioButtons(ax_metric, METRICS,
                             active=METRICS.index(DEFAULTS["metric"]))

ax_p = plt.axes([0.45, 0.16, 0.15, 0.020])
sl_p = Slider(ax_p, "p (Mink.)", 1.0, 5.0, valinit=DEFAULTS["minkowski_p"])
sl_p.set_active(DEFAULTS["metric"] == "Minkowski")


def _on_metric(label):
    state["metric"] = label
    sl_p.set_active(label == "Minkowski")
    if label == "Mahalanobis":
        state["mahal_VI"] = _compute_mahal_VI(state["X_train"])
    redraw()


def _on_p(v):
    state["minkowski_p"] = float(v)
    if state["metric"] == "Minkowski":
        redraw()


radio_metric.on_clicked(_on_metric)
sl_p.on_changed(_on_p)


# ===========================================================
# Recuadro Pesos
# ===========================================================
_add_group_box(0.65, 0.04, 0.32, 0.255, "Pesos")
ax_weights = plt.axes([0.66, 0.07, 0.10, 0.20])
radio_weights = RadioButtons(ax_weights, WEIGHTS,
                              active=WEIGHTS.index(DEFAULTS["weights"]))

ax_h = plt.axes([0.80, 0.16, 0.15, 0.020])
sl_h = Slider(ax_h, "h (gauss)", 0.1, 3.0, valinit=DEFAULTS["gaussian_h"])
sl_h.set_active(DEFAULTS["weights"] == "gaussiano")


def _on_weights(label):
    state["weights"] = label
    sl_h.set_active(label == "gaussiano")
    redraw()


def _on_h(v):
    state["gaussian_h"] = float(v)
    if state["weights"] == "gaussiano":
        redraw()


radio_weights.on_clicked(_on_weights)
sl_h.on_changed(_on_h)

if __name__ == "__main__":
    plt.show()
