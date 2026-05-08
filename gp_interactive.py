"""
Demo interactivo de Proceso Gaussiano (GP) en 1D.

Dependencias gestionadas con uv (ver pyproject.toml: numpy, matplotlib).

Ejecutar:
    uv run python gp_interactive.py

Si el backend por defecto no es interactivo:
    MPLBACKEND=TkAgg uv run python gp_interactive.py

Uso:
  - Click izquierdo en el gráfico:
        agrega una observación en x = posición del clic.
        y = g(x) + ruido gaussiano con sigma = slider σ_n.
  - Click derecho (o Shift+click izquierdo):
        agrega una observación arbitraria en (x, y) = posición del clic
        (también con incerteza σ_n en y).
  - Sliders:
        σ_n  -> incerteza de la próxima observación (afecta solo nuevos puntos).
        ℓ    -> escala de longitud del kernel.
        σ_f  -> amplitud (varianza prior) del kernel.
  - Radio: kernel a usar.
  - Botones: Reset (todo) / Limpiar historia (deja solo posterior actual).

Cada vez que se agrega un punto, la posterior previa queda dibujada
con menor opacidad (las más viejas se desvanecen).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button


# ===========================================================
# Función real (oculta para el GP, mostrada como referencia)
# Bimodal: dos gaussianas con amplitudes distintas (moda izq. > moda der.)
# ===========================================================
def g(x):
    peak_high = 2.2 * np.exp(-0.5 * ((x - (-1.5)) / 0.7) ** 2)  # moda mayor
    peak_low = 1.0 * np.exp(-0.5 * ((x - 2.0) / 0.9) ** 2)       # moda menor
    return peak_high + peak_low


# ===========================================================
# Kernels
# ===========================================================
def k_rbf(x1, x2, ls=1.0, sf=1.0):
    d2 = np.subtract.outer(np.ravel(x1), np.ravel(x2)) ** 2
    return sf ** 2 * np.exp(-0.5 * d2 / ls ** 2)


def k_matern32(x1, x2, ls=1.0, sf=1.0):
    d = np.abs(np.subtract.outer(np.ravel(x1), np.ravel(x2)))
    s = np.sqrt(3.0) * d / ls
    return sf ** 2 * (1.0 + s) * np.exp(-s)


def k_matern52(x1, x2, ls=1.0, sf=1.0):
    d = np.abs(np.subtract.outer(np.ravel(x1), np.ravel(x2)))
    s = np.sqrt(5.0) * d / ls
    return sf ** 2 * (1.0 + s + 5.0 * d ** 2 / (3.0 * ls ** 2)) * np.exp(-s)


def k_rq(x1, x2, ls=1.0, sf=1.0, alpha=1.0):
    d2 = np.subtract.outer(np.ravel(x1), np.ravel(x2)) ** 2
    return sf ** 2 * (1.0 + d2 / (2.0 * alpha * ls ** 2)) ** (-alpha)


def k_periodic(x1, x2, ls=1.0, sf=1.0, period=2.0):
    d = np.abs(np.subtract.outer(np.ravel(x1), np.ravel(x2)))
    return sf ** 2 * np.exp(-2.0 * np.sin(np.pi * d / period) ** 2 / ls ** 2)


KERNELS = {
    "RBF": k_rbf,
    "Matern 3/2": k_matern32,
    "Matern 5/2": k_matern52,
    "Rational Q.": k_rq,
    "Periodic": k_periodic,
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
    Kss_diag = sf ** 2 * np.ones(len(Xs))  # k(x*, x*) para los kernels usados

    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    mu = Ks.T @ alpha

    v = np.linalg.solve(L, Ks)
    var = Kss_diag - np.sum(v ** 2, axis=0)
    var = np.clip(var, 1e-10, None)
    return mu, var


# ===========================================================
# Estado
# ===========================================================
state = {
    "X": [],
    "y": [],
    "sn": [],            # incerteza por punto (heterocedástico)
    "kernel": "RBF",
    "ls": 1.0,
    "sf": 1.0,
    "sigma_n_current": 0.2,
    "history_means": [], # lista de arrays mu(Xs) anteriores
}

X_MIN, X_MAX = -5.0, 5.0
Y_MIN, Y_MAX = -3.0, 3.0
Xs = np.linspace(X_MIN, X_MAX, 400)


# ===========================================================
# Figura y layout
# ===========================================================
fig, ax = plt.subplots(figsize=(11, 7))
plt.subplots_adjust(left=0.28, bottom=0.28, right=0.97, top=0.93)

ax.plot(Xs, g(Xs), color="green", ls="--", lw=2, label="g(x) real (oculta)")
ax.axhline(0.0, color="gray", ls=":", lw=1, label="prior  f(x)=0")

post_line, = ax.plot([], [], color="blue", lw=2, label="f(x) posterior (actual)")
post_band = ax.fill_between(Xs, np.zeros_like(Xs), np.zeros_like(Xs),
                            color="blue", alpha=0.0)

ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Y_MIN, Y_MAX)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(alpha=0.3)
ax.set_title("GP interactivo  —  click izq.: muestra g(x)+ruido   |   click der.: punto arbitrario")
ax.legend(loc="upper right")

history_artists = []
errbar_holder = [None]


def _safe_remove(artist):
    try:
        artist.remove()
    except Exception:
        pass


def _remove_errorbar(eb):
    if eb is None:
        return
    line, caps, bars = eb.lines
    _safe_remove(line)
    for c in caps:
        _safe_remove(c)
    for b in bars:
        _safe_remove(b)


def redraw():
    global post_band

    # Borrar artistas previos de historia y puntos
    for a in history_artists:
        _safe_remove(a)
    history_artists.clear()

    _remove_errorbar(errbar_holder[0])
    errbar_holder[0] = None

    # Dibujar historia (más viejo => más transparente)
    n = len(state["history_means"])
    for i, mu_old in enumerate(state["history_means"]):
        age = n - 1 - i  # 0 = el más reciente de los viejos
        alpha = 0.40 * (0.55 ** age)
        if alpha < 0.03:
            continue
        ln, = ax.plot(Xs, mu_old, color="blue", alpha=alpha, lw=1.0)
        history_artists.append(ln)

    # Posterior actual + banda de 2σ
    _safe_remove(post_band)
    if len(state["X"]) > 0:
        kfn = KERNELS[state["kernel"]]
        mu, var = gp_posterior(state["X"], state["y"], state["sn"],
                               Xs, kfn, state["ls"], state["sf"])
        sd = np.sqrt(var)
        post_line.set_data(Xs, mu)
        post_band = ax.fill_between(Xs, mu - 2 * sd, mu + 2 * sd,
                                    color="blue", alpha=0.20)

        eb = ax.errorbar(state["X"], state["y"], yerr=state["sn"],
                         fmt="o", color="red", ecolor="red",
                         capsize=4, ms=6, zorder=10)
        errbar_holder[0] = eb
    else:
        post_line.set_data([], [])
        post_band = ax.fill_between(Xs, np.zeros_like(Xs), np.zeros_like(Xs),
                                    color="blue", alpha=0.0)

    fig.canvas.draw_idle()


# ===========================================================
# Click handler
# ===========================================================
def on_click(event):
    if event.inaxes != ax:
        return
    if event.xdata is None or event.ydata is None:
        return

    sn = state["sigma_n_current"]
    x_new = float(event.xdata)

    # Click izquierdo => observación de g(x) + ruido
    # Click derecho o Shift+izq => usa la y del clic
    arbitrary = (event.button == 3) or (
        event.button == 1 and getattr(event, "key", None) == "shift"
    )
    if arbitrary:
        y_new = float(event.ydata)
    else:
        y_new = float(g(x_new) + np.random.normal(0.0, sn))

    # Guardar la posterior actual como historia (antes de incorporar el nuevo dato)
    if len(state["X"]) > 0:
        kfn = KERNELS[state["kernel"]]
        mu_now, _ = gp_posterior(state["X"], state["y"], state["sn"],
                                 Xs, kfn, state["ls"], state["sf"])
        state["history_means"].append(mu_now)

    state["X"].append(x_new)
    state["y"].append(y_new)
    state["sn"].append(sn)

    redraw()


fig.canvas.mpl_connect("button_press_event", on_click)


# ===========================================================
# Sliders
# ===========================================================
ax_sn = plt.axes([0.30, 0.16, 0.60, 0.025])
sl_sn = Slider(ax_sn, r"$\sigma_n$ (incerteza nuevo punto)", 0.01, 1.0, valinit=0.2)

ax_ls = plt.axes([0.30, 0.10, 0.60, 0.025])
sl_ls = Slider(ax_ls, r"$\ell$ (escala kernel)", 0.05, 5.0, valinit=1.0)

ax_sf = plt.axes([0.30, 0.04, 0.60, 0.025])
sl_sf = Slider(ax_sf, r"$\sigma_f$ (amplitud kernel)", 0.1, 3.0, valinit=1.0)


def _on_sn(v):
    state["sigma_n_current"] = float(v)


def _on_ls(v):
    state["ls"] = float(v)
    redraw()


def _on_sf(v):
    state["sf"] = float(v)
    redraw()


sl_sn.on_changed(_on_sn)
sl_ls.on_changed(_on_ls)
sl_sf.on_changed(_on_sf)


# ===========================================================
# Radio buttons (kernel)
# ===========================================================
ax_radio = plt.axes([0.02, 0.50, 0.22, 0.30])
ax_radio.set_title("Kernel  k(x, x')", fontsize=10)
radio = RadioButtons(ax_radio, list(KERNELS.keys()), active=0)


def _on_kernel(label):
    state["kernel"] = label
    redraw()


radio.on_clicked(_on_kernel)


# ===========================================================
# Botones
# ===========================================================
ax_clr_hist = plt.axes([0.04, 0.40, 0.18, 0.05])
btn_clr_hist = Button(ax_clr_hist, "Limpiar historia")


def _on_clr_hist(event):
    state["history_means"].clear()
    redraw()


btn_clr_hist.on_clicked(_on_clr_hist)


ax_reset = plt.axes([0.04, 0.33, 0.18, 0.05])
btn_reset = Button(ax_reset, "Reset todo")


def _on_reset(event):
    state["X"].clear()
    state["y"].clear()
    state["sn"].clear()
    state["history_means"].clear()
    redraw()


btn_reset.on_clicked(_on_reset)


# ===========================================================
# Inicio
# ===========================================================
redraw()
plt.show()
