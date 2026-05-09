"""
Demo interactivo de Proceso Gaussiano en 2D (x ∈ R²).

Mismas funcionalidades que la versión 1D pero la entrada es bidimensional:
  - g(x) = mezcla de dos gaussianas BIVARIADAS (cada una con posible
    correlación entre x1 y x2). Una moda es claramente mayor.
  - g(x) y la posterior μ(x) se grafican como **líneas de nivel**
    (contornos), un solo color cada una. El máximo de cada superficie se
    marca con una "X".
  - Click sobre el plano (x1, x2) agrega una observación y = g(x) + ruido.
  - Estrategias: paso de exploración, argmax(μ), UCB y EI.
  - Misma selección de kernel, contadores, historia con desvanecimiento,
    regenerar g(x), reset, mostrar/ocultar g(x).

Dependencias: numpy, matplotlib, scipy.

Ejecutar:
    uv run python gp_interactive_2d.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, RadioButtons, Button, CheckButtons
from scipy.stats import norm


# ===========================================================
# Función real (oculta para el GP) — mezcla de 2 gaussianas bivariadas
# Cada componente tiene Σ con posible correlación entre x1 y x2.
# Una moda es siempre claramente mayor que la otra.
# ===========================================================
RNG = np.random.default_rng()


def _random_cov_2d(rng):
    sx = rng.uniform(0.5, 1.2)
    sy = rng.uniform(0.5, 1.2)
    rho = rng.uniform(-0.7, 0.7)
    return np.array([[sx ** 2,        rho * sx * sy],
                     [rho * sx * sy,  sy ** 2]])


def make_random_g_2d(rng=None):
    rng = rng or RNG
    while True:
        m1 = rng.uniform(-3.5, 3.5, size=2)
        m2 = rng.uniform(-3.5, 3.5, size=2)
        if np.linalg.norm(m1 - m2) >= 2.5:
            break
    h_big = rng.uniform(1.8, 2.5)
    h_small = rng.uniform(0.7, 1.3)
    if rng.random() < 0.5:
        m_big, m_small = m1, m2
    else:
        m_big, m_small = m2, m1
    inv_big = np.linalg.inv(_random_cov_2d(rng))
    inv_small = np.linalg.inv(_random_cov_2d(rng))

    def g(X):
        X = np.asarray(X, dtype=float)
        d_big = X - m_big
        d_small = X - m_small
        e_big = np.einsum('...i,ij,...j->...', d_big, inv_big, d_big)
        e_small = np.einsum('...i,ij,...j->...', d_small, inv_small, d_small)
        return h_big * np.exp(-0.5 * e_big) + h_small * np.exp(-0.5 * e_small)

    return g


# ===========================================================
# Kernels (radiales: dependen sólo de d=||x-x'||, valen en R^d)
# ===========================================================
def _pairwise_sqdist(X1, X2):
    X1 = np.atleast_2d(X1).astype(float)
    X2 = np.atleast_2d(X2).astype(float)
    A2 = np.sum(X1 * X1, axis=1)
    B2 = np.sum(X2 * X2, axis=1)
    return np.maximum(A2[:, None] + B2[None, :] - 2.0 * X1 @ X2.T, 0.0)


def k_rbf(X1, X2, ls=1.0, sf=1.0):
    return sf ** 2 * np.exp(-0.5 * _pairwise_sqdist(X1, X2) / ls ** 2)


def k_matern32(X1, X2, ls=1.0, sf=1.0):
    d = np.sqrt(_pairwise_sqdist(X1, X2))
    s = np.sqrt(3.0) * d / ls
    return sf ** 2 * (1.0 + s) * np.exp(-s)


def k_matern52(X1, X2, ls=1.0, sf=1.0):
    d2 = _pairwise_sqdist(X1, X2)
    d = np.sqrt(d2)
    s = np.sqrt(5.0) * d / ls
    return sf ** 2 * (1.0 + s + 5.0 * d2 / (3.0 * ls ** 2)) * np.exp(-s)


def k_rq(X1, X2, ls=1.0, sf=1.0, alpha=1.0):
    d2 = _pairwise_sqdist(X1, X2)
    return sf ** 2 * (1.0 + d2 / (2.0 * alpha * ls ** 2)) ** (-alpha)


def k_periodic(X1, X2, ls=1.0, sf=1.0, period=2.0):
    d = np.sqrt(_pairwise_sqdist(X1, X2))
    return sf ** 2 * np.exp(-2.0 * np.sin(np.pi * d / period) ** 2 / ls ** 2)


KERNELS = {
    "RBF": k_rbf,
    "Matern 3/2": k_matern32,
    "Matern 5/2": k_matern52,
    "Rational Q.": k_rq,
    "Periodic": k_periodic,
}


KERNEL_FORMULAS = {
    "RBF":
        r"$k(x,x') = \sigma_f^2\,\exp\!\left(-\dfrac{d^2}{2\ell^2}\right)"
        r",\ \ d=\|x-x'\|$",
    "Matern 3/2":
        r"$k(x,x') = \sigma_f^2\left(1+\dfrac{\sqrt{3}\,d}{\ell}\right)"
        r"e^{-\sqrt{3}\,d/\ell},\ \ d=\|x-x'\|$",
    "Matern 5/2":
        r"$k(x,x') = \sigma_f^2\left(1+\dfrac{\sqrt{5}\,d}{\ell}+\dfrac{5d^2}{3\ell^2}\right)"
        r"e^{-\sqrt{5}\,d/\ell},\ \ d=\|x-x'\|$",
    "Rational Q.":
        r"$k(x,x') = \sigma_f^2\left(1+\dfrac{d^2}{2\alpha\ell^2}\right)^{-\alpha}"
        r",\ \ \alpha=1$",
    "Periodic":
        r"$k(x,x') = \sigma_f^2\,\exp\!\left(-\dfrac{2\sin^2(\pi d/p)}{\ell^2}\right)"
        r",\ \ p=2$",
}


# ===========================================================
# Posterior del GP (con ruido heterocedástico por punto)
# ===========================================================
def gp_posterior(X, y, sn, Xs, kfn, ls, sf):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    sn = np.asarray(sn, dtype=float)
    K = kfn(X, X, ls=ls, sf=sf) + np.diag(sn ** 2) + 1e-8 * np.eye(len(X))
    Ks = kfn(X, Xs, ls=ls, sf=sf)
    Kss_diag = sf ** 2 * np.ones(len(Xs))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    mu = Ks.T @ alpha
    v = np.linalg.solve(L, Ks)
    var = np.clip(Kss_diag - np.sum(v ** 2, axis=0), 1e-10, None)
    return mu, var


# ===========================================================
# Estado
# ===========================================================
state = {
    "X": [],              # lista de np.array shape (2,)
    "y": [],
    "sn": [],
    "kernel": "RBF",
    "ls": 1.0,
    "sf": 1.0,
    "sigma_n_current": 0.2,
    "ucb_beta": 2.0,
    "history_means": [],  # lista de arrays shape (GRID, GRID)
    "g": make_random_g_2d(),
    "n_explore": 0,
    "n_exploit": 0,
}

X1_MIN, X1_MAX = -5.0, 5.0
X2_MIN, X2_MAX = -5.0, 5.0
GRID = 50
xs1 = np.linspace(X1_MIN, X1_MAX, GRID)
xs2 = np.linspace(X2_MIN, X2_MAX, GRID)
XX1, XX2 = np.meshgrid(xs1, xs2)
Xs_grid = np.column_stack([XX1.ravel(), XX2.ravel()])  # (GRID*GRID, 2)

TRUE_LEVELS = [0.2, 0.6, 1.0, 1.4, 1.8, 2.2]
POST_LEVELS = np.linspace(-0.5, 2.5, 6)


# ===========================================================
# Figura y layout
# ===========================================================
fig, ax = plt.subplots(figsize=(11, 7))
plt.subplots_adjust(left=0.28, bottom=0.34, right=0.97, top=0.93)
ax.set_aspect("equal")
ax.set_xlim(X1_MIN, X1_MAX)
ax.set_ylim(X2_MIN, X2_MAX)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.grid(alpha=0.3)
ax.set_title("Proceso Gaussiano interactivo (2D)", loc="left")

# Marcadores
g_max_marker, = ax.plot([], [], "x", color="green", ms=14, mew=2.5)
post_max_marker, = ax.plot([], [], "x", color="blue", ms=14, mew=2.5)
points_marker, = ax.plot([], [], "o", color="red", ms=7, mec="black",
                          mew=0.8, ls="none")

# Proxies para la leyenda (las líneas de nivel no aparecen ahí solas)
g_proxy = Line2D([], [], color="green", ls="--", lw=1.5)
post_proxy = Line2D([], [], color="blue", ls="-", lw=1.5)


def _refresh_legend():
    handles, labels = [], []
    if g_max_marker.get_visible():
        handles += [g_proxy, g_max_marker]
        labels += ["g(x) real", "argmax g(x)"]
    handles += [post_proxy, post_max_marker, points_marker]
    labels += [r"$\mu(x)$ posterior", r"argmax $\mu(x)$", "observaciones"]
    ax.legend(handles, labels, loc="upper right", fontsize=9)


# Contornos: contenedor mutable para poder reemplazar en cada redraw
g_contour = [None]
post_contour = [None]
hist_contours = []


def _safe_remove(artist):
    try:
        artist.remove()
    except Exception:
        pass


def _draw_true_contour():
    """Redibuja el contorno de g(x), respetando la visibilidad actual."""
    _safe_remove(g_contour[0])
    g_contour[0] = None
    G = state["g"](Xs_grid).reshape(GRID, GRID)
    qcs = ax.contour(XX1, XX2, G, levels=TRUE_LEVELS, colors="green",
                     linestyles="--", linewidths=1.2, alpha=0.9)
    qcs.set_visible(g_max_marker.get_visible())
    g_contour[0] = qcs
    # Actualizar posición del marcador de máximo de g
    idx = int(np.argmax(state["g"](Xs_grid)))
    g_max_marker.set_data([Xs_grid[idx, 0]], [Xs_grid[idx, 1]])


# Inicialización: g oculta por defecto
g_max_marker.set_visible(False)
_draw_true_contour()
_refresh_legend()


def _add_group_box(x, y, w, h, label):
    rect = Rectangle((x, y), w, h, transform=fig.transFigure,
                     linewidth=0.9, edgecolor="#666666",
                     facecolor="none", zorder=0)
    fig.add_artist(rect)
    return fig.text(x + w / 2, y + h + 0.006, label,
                    ha="center", va="bottom", fontsize=9, weight="bold",
                    color="#333333")


def redraw():
    explore_label.set_text(f"Exploración:  {state['n_explore']}")
    exploit_label.set_text(f"Explotación:  {state['n_exploit']}")

    _safe_remove(post_contour[0])
    post_contour[0] = None
    for h in hist_contours:
        _safe_remove(h)
    hist_contours.clear()

    # Historia (más viejo => más transparente)
    n = len(state["history_means"])
    for i, mu_grid in enumerate(state["history_means"]):
        age = n - 1 - i
        alpha = 0.45 * (0.88 ** age)
        if alpha < 0.04:
            continue
        qcs = ax.contour(XX1, XX2, mu_grid, levels=POST_LEVELS,
                         colors="blue", linewidths=0.5, alpha=alpha)
        hist_contours.append(qcs)

    # Posterior actual + máximo + observaciones
    if len(state["X"]) > 0:
        kfn = KERNELS[state["kernel"]]
        X_train = np.asarray(state["X"])
        mu, _ = gp_posterior(X_train, state["y"], state["sn"],
                             Xs_grid, kfn, state["ls"], state["sf"])
        mu_grid = mu.reshape(GRID, GRID)
        post_contour[0] = ax.contour(XX1, XX2, mu_grid, levels=POST_LEVELS,
                                     colors="blue", linewidths=1.4, alpha=0.95)
        idx = int(np.argmax(mu))
        post_max_marker.set_data([Xs_grid[idx, 0]], [Xs_grid[idx, 1]])
        post_max_marker.set_visible(True)
        points_marker.set_data(X_train[:, 0], X_train[:, 1])
    else:
        post_max_marker.set_visible(False)
        points_marker.set_data([], [])

    fig.canvas.draw_idle()


def _current_posterior():
    if len(state["X"]) == 0:
        return None, None
    kfn = KERNELS[state["kernel"]]
    X_train = np.asarray(state["X"])
    return gp_posterior(X_train, state["y"], state["sn"], Xs_grid,
                        kfn, state["ls"], state["sf"])


def _add_observation(x_new, y_new, sn):
    mu_now, _ = _current_posterior()
    if mu_now is not None:
        state["history_means"].append(mu_now.reshape(GRID, GRID))
    state["X"].append(np.asarray(x_new, dtype=float))
    state["y"].append(float(y_new))
    state["sn"].append(float(sn))
    redraw()


def _sample_g_at(x):
    sn = state["sigma_n_current"]
    g_val = float(state["g"](np.atleast_2d(x))[0])
    return g_val + RNG.normal(0.0, sn), sn


def _random_x():
    return np.array([RNG.uniform(X1_MIN, X1_MAX),
                     RNG.uniform(X2_MIN, X2_MAX)])


def _argmax_random_ties(values, rel_tol=1e-3):
    """argmax con empate aleatorio.
    En 2D la varianza es prácticamente plana (= σ_f²) en toda zona alejada
    de las observaciones, así que el argmax determinista siempre cae en el
    primer índice de la grilla (borde inferior-izquierdo). Con esto, entre
    todos los puntos que empatan dentro de `rel_tol·rango` con el máximo,
    se elige uno al azar.
    """
    values = np.asarray(values)
    span = max(values.max() - values.min(), 1e-12)
    threshold = values.max() - rel_tol * span
    candidates = np.flatnonzero(values >= threshold)
    return int(RNG.choice(candidates))


# ===========================================================
# Click handler: agrega observación en (x1, x2) con y = g(x) + ruido
# ===========================================================
def on_click(event):
    if event.inaxes != ax:
        return
    if event.xdata is None or event.ydata is None:
        return
    x_new = np.array([event.xdata, event.ydata])
    y_new, sn = _sample_g_at(x_new)
    _add_observation(x_new, y_new, sn)


fig.canvas.mpl_connect("button_press_event", on_click)


# ===========================================================
# Sliders y selección de kernel (zona inferior)
# ===========================================================
ax_sn = plt.axes([0.32, 0.290, 0.55, 0.020])
sl_sn = Slider(ax_sn, r"$\sigma_n$  (incerteza obs.)", 0.01, 1.0, valinit=0.2)


def _on_sn(v):
    state["sigma_n_current"] = float(v)


sl_sn.on_changed(_on_sn)


_add_group_box(0.025, 0.040, 0.945, 0.225, "Kernel")
ax_radio = plt.axes([0.045, 0.060, 0.13, 0.180])
radio = RadioButtons(ax_radio, list(KERNELS.keys()), active=0)

ax_ls = plt.axes([0.32, 0.215, 0.55, 0.018])
sl_ls = Slider(ax_ls, r"$\ell$  (escala)", 0.05, 5.0, valinit=1.0)

ax_sf = plt.axes([0.32, 0.175, 0.55, 0.018])
sl_sf = Slider(ax_sf, r"$\sigma_f$  (amplitud)", 0.1, 3.0, valinit=1.0)

formula_text = fig.text(0.32, 0.090, KERNEL_FORMULAS["RBF"],
                         fontsize=12, va="center", ha="left")


def _on_ls(v):
    state["ls"] = float(v); redraw()


def _on_sf(v):
    state["sf"] = float(v); redraw()


def _on_kernel(label):
    state["kernel"] = label
    formula_text.set_text(KERNEL_FORMULAS[label])
    fig.canvas.draw_idle()
    redraw()


sl_ls.on_changed(_on_ls)
sl_sf.on_changed(_on_sf)
radio.on_clicked(_on_kernel)


# ===========================================================
# Botones (4 grupos visuales)
# ===========================================================

# --- Grupo: Vista ---
_add_group_box(0.025, 0.880, 0.21, 0.045, "Vista")
ax_check_g = plt.axes([0.04, 0.882, 0.18, 0.040])
ax_check_g.set_facecolor("none")
chk_g = CheckButtons(ax_check_g, ["Mostrar g(x)"], actives=[False])


def _on_toggle_g(_label):
    visible = not g_max_marker.get_visible()
    g_max_marker.set_visible(visible)
    if g_contour[0] is not None:
        g_contour[0].set_visible(visible)
    _refresh_legend()
    fig.canvas.draw_idle()


chk_g.on_clicked(_on_toggle_g)


# --- Grupo: Exploración ---
explore_label = _add_group_box(0.025, 0.790, 0.21, 0.05, "Exploración")
ax_explore = plt.axes([0.04, 0.795, 0.18, 0.04])
btn_explore = Button(ax_explore, "Paso de exploración",
                      color="#cde7ff", hovercolor="#a8d2f8")


def _on_explore(_event):
    if len(state["X"]) == 0:
        x_new = _random_x()
    else:
        _, var = _current_posterior()
        x_new = Xs_grid[_argmax_random_ties(var)]
    y_new, sn = _sample_g_at(x_new)
    state["n_explore"] += 1
    _add_observation(x_new, y_new, sn)


btn_explore.on_clicked(_on_explore)


# --- Grupo: Explotación ---
exploit_label = _add_group_box(0.025, 0.560, 0.21, 0.185, "Explotación")
ax_exploit = plt.axes([0.04, 0.690, 0.18, 0.04])
btn_exploit = Button(ax_exploit, "Paso argmax(μ)",
                      color="#ffd8c2", hovercolor="#f8b690")


def _on_exploit(_event):
    if len(state["X"]) == 0:
        x_new = _random_x()
    else:
        mu, _ = _current_posterior()
        x_new = Xs_grid[_argmax_random_ties(mu)]
    y_new, sn = _sample_g_at(x_new)
    state["n_exploit"] += 1
    _add_observation(x_new, y_new, sn)


btn_exploit.on_clicked(_on_exploit)


ax_ucb = plt.axes([0.04, 0.640, 0.18, 0.04])
btn_ucb = Button(ax_ucb, "Paso UCB", color="#e8d6ff", hovercolor="#c9adf2")


def _on_ucb(_event):
    beta = state["ucb_beta"]
    if len(state["X"]) == 0:
        x_new = _random_x()
    else:
        mu, var = _current_posterior()
        x_new = Xs_grid[_argmax_random_ties(mu + beta * np.sqrt(var))]
    y_new, sn = _sample_g_at(x_new)
    state["n_exploit"] += 1
    _add_observation(x_new, y_new, sn)


btn_ucb.on_clicked(_on_ucb)


ax_beta = plt.axes([0.06, 0.615, 0.14, 0.020])
sl_beta = Slider(ax_beta, r"$\beta$", 0.0, 5.0, valinit=2.0)


def _on_beta(v):
    state["ucb_beta"] = float(v)


sl_beta.on_changed(_on_beta)


ax_ei = plt.axes([0.04, 0.570, 0.18, 0.04])
btn_ei = Button(ax_ei, "Paso EI", color="#fff0b0", hovercolor="#f5dc73")


def _on_ei(_event):
    if len(state["X"]) == 0:
        x_new = _random_x()
    else:
        mu, var = _current_posterior()
        sd = np.sqrt(var)
        f_star = max(state["y"])
        eps = 1e-9
        z = (mu - f_star) / np.maximum(sd, eps)
        ei = (mu - f_star) * norm.cdf(z) + sd * norm.pdf(z)
        ei = np.where(sd < eps, 0.0, ei)
        x_new = Xs_grid[_argmax_random_ties(ei)]
    y_new, sn = _sample_g_at(x_new)
    state["n_exploit"] += 1
    _add_observation(x_new, y_new, sn)


btn_ei.on_clicked(_on_ei)


# --- Grupo: Control ---
_add_group_box(0.025, 0.340, 0.21, 0.185, "Control")
ax_regen = plt.axes([0.04, 0.465, 0.18, 0.04])
btn_regen = Button(ax_regen, "Regenerar g(x)",
                    color="#d9f5d2", hovercolor="#b6e6a8")


def _on_regen(_event):
    state["g"] = make_random_g_2d()
    state["X"].clear()
    state["y"].clear()
    state["sn"].clear()
    state["history_means"].clear()
    state["n_explore"] = 0
    state["n_exploit"] = 0
    _draw_true_contour()
    redraw()


btn_regen.on_clicked(_on_regen)


ax_clr_hist = plt.axes([0.04, 0.415, 0.18, 0.04])
btn_clr_hist = Button(ax_clr_hist, "Limpiar historia")


def _on_clr_hist(_event):
    state["history_means"].clear()
    redraw()


btn_clr_hist.on_clicked(_on_clr_hist)


ax_reset = plt.axes([0.04, 0.360, 0.18, 0.04])
btn_reset = Button(ax_reset, "Reset todo")


def _on_reset(_event):
    state["X"].clear()
    state["y"].clear()
    state["sn"].clear()
    state["history_means"].clear()
    state["n_explore"] = 0
    state["n_exploit"] = 0
    redraw()


btn_reset.on_clicked(_on_reset)


redraw()
plt.show()
