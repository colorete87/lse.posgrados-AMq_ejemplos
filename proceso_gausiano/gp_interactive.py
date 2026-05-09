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
  - Botón "Exploración":
        agrega un punto en el x donde la varianza posterior es máxima
        (estrategia de exploración pura).
  - Botón "Explotación":
        agrega un punto en el x donde la media posterior es máxima
        (estrategia greedy: candidato a maximizar g(x)).
  - Botón "UCB":
        agrega un punto en argmax(μ(x) + β·σ(x)). El slider β controla
        el balance entre explotación (β bajo) y exploración (β alto).
  - Botón "EI" (Expected Improvement):
        agrega un punto en argmax E[max(0, f(x) - f*)], donde f* es el
        mejor y observado. Balancea explotación/exploración sin hiperparámetro
        y naturalmente evita re-muestrear donde σ(x) ya es chica.
  - Botón "Regenerar g(x)":
        sortea una nueva función real bimodal y limpia las observaciones.
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
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, RadioButtons, Button, CheckButtons
from scipy.stats import norm


# ===========================================================
# Función real (oculta para el GP, mostrada como referencia)
# Bimodal aleatoria: dos gaussianas con amplitudes distintas
# (una moda claramente mayor que la otra).
# ===========================================================
RNG = np.random.default_rng()


def make_random_g(rng=None):
    rng = rng or RNG
    # Posiciones de las modas, separadas para que se distinga la bimodalidad.
    while True:
        x1 = rng.uniform(-3.5, 3.5)
        x2 = rng.uniform(-3.5, 3.5)
        if abs(x1 - x2) >= 2.0:
            break
    # Alturas: una mayor (moda dominante), otra menor.
    h_big = rng.uniform(1.8, 2.5)
    h_small = rng.uniform(0.7, 1.3)
    # Cuál de las dos es la dominante (izq o der) se decide al azar.
    if rng.random() < 0.5:
        x_big, x_small = x1, x2
    else:
        x_big, x_small = x2, x1
    # Anchos.
    w_big = rng.uniform(0.5, 1.0)
    w_small = rng.uniform(0.5, 1.0)

    def g(x):
        return (h_big * np.exp(-0.5 * ((x - x_big) / w_big) ** 2)
                + h_small * np.exp(-0.5 * ((x - x_small) / w_small) ** 2))

    return g


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


KERNEL_FORMULAS = {
    "RBF":
        r"$k(x,x') = \sigma_f^2\,\exp\!\left(-\dfrac{(x-x')^2}{2\ell^2}\right)$",
    "Matern 3/2":
        r"$k(x,x') = \sigma_f^2\left(1+\dfrac{\sqrt{3}\,d}{\ell}\right)e^{-\sqrt{3}\,d/\ell}"
        r",\ \ d=|x-x'|$",
    "Matern 5/2":
        r"$k(x,x') = \sigma_f^2\left(1+\dfrac{\sqrt{5}\,d}{\ell}+\dfrac{5d^2}{3\ell^2}\right)"
        r"e^{-\sqrt{5}\,d/\ell},\ \ d=|x-x'|$",
    "Rational Q.":
        r"$k(x,x') = \sigma_f^2\left(1+\dfrac{(x-x')^2}{2\alpha\ell^2}\right)^{-\alpha}"
        r",\ \ \alpha=1$",
    "Periodic":
        r"$k(x,x') = \sigma_f^2\,\exp\!\left(-\dfrac{2\sin^2(\pi|x-x'|/p)}{\ell^2}\right)"
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
    "ucb_beta": 2.0,
    "history_means": [], # lista de arrays mu(Xs) anteriores
    "g": make_random_g(),
    "n_explore": 0,
    "n_exploit": 0,
}

X_MIN, X_MAX = -5.0, 5.0
Y_MIN, Y_MAX = -3.0, 3.0
Xs = np.linspace(X_MIN, X_MAX, 400)


# ===========================================================
# Figura y layout
# ===========================================================
fig, ax = plt.subplots(figsize=(11, 7))
plt.subplots_adjust(left=0.28, bottom=0.34, right=0.97, top=0.93)

TRUE_LINE_LABEL = "g(x) real"
true_line, = ax.plot(Xs, state["g"](Xs), color="green", ls="--", lw=2,
                     label="_nolegend_")
true_line.set_visible(False)  # por defecto oculta; se muestra desde el checkbox
ax.axhline(0.0, color="gray", ls=":", lw=1, label="prior  f(x)=0")

post_line, = ax.plot([], [], color="blue", lw=2, label="f(x) posterior (actual)")
post_band = ax.fill_between(Xs, np.zeros_like(Xs), np.zeros_like(Xs),
                            color="blue", alpha=0.0)

ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Y_MIN, Y_MAX)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(alpha=0.3)
ax.set_title("Proceso Gaussiano interactivo", loc="left")
ax.legend(loc="upper right")

history_artists = []
errbar_holder = [None]

def _add_group_box(x, y, w, h, label):
    """Dibuja un recuadro con etiqueta encima en coordenadas de figura.
    Devuelve el artista de texto para poder actualizar la etiqueta luego.
    """
    rect = Rectangle(
        (x, y), w, h,
        transform=fig.transFigure,
        linewidth=0.9, edgecolor="#666666", facecolor="none",
        zorder=0,
    )
    fig.add_artist(rect)
    label_artist = fig.text(
        x + w / 2, y + h + 0.006, label,
        ha="center", va="bottom",
        fontsize=9, weight="bold", color="#333333",
    )
    return label_artist


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

    explore_label.set_text(f"Exploración:  {state['n_explore']}")
    exploit_label.set_text(f"Explotación:  {state['n_exploit']}")

    # Borrar artistas previos de historia y puntos
    for a in history_artists:
        _safe_remove(a)
    history_artists.clear()

    _remove_errorbar(errbar_holder[0])
    errbar_holder[0] = None

    # Dibujar historia (más viejo => más transparente, decaimiento suave)
    n = len(state["history_means"])
    for i, mu_old in enumerate(state["history_means"]):
        age = n - 1 - i  # 0 = el más reciente de los viejos
        alpha = 0.45 * (0.88 ** age)
        if alpha < 0.02:
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
# Helpers para agregar observaciones (usado por click y botones)
# ===========================================================
def _current_posterior():
    """Devuelve (mu, var) sobre Xs con el estado actual, o (None, None) si no hay datos."""
    if len(state["X"]) == 0:
        return None, None
    kfn = KERNELS[state["kernel"]]
    return gp_posterior(state["X"], state["y"], state["sn"],
                        Xs, kfn, state["ls"], state["sf"])


def _add_observation(x_new, y_new, sn):
    # Snapshot de la posterior actual antes de incorporar el dato.
    mu_now, _ = _current_posterior()
    if mu_now is not None:
        state["history_means"].append(mu_now)
    state["X"].append(float(x_new))
    state["y"].append(float(y_new))
    state["sn"].append(float(sn))
    redraw()


def _sample_g_at(x):
    """Simula una observación ruidosa de g(x) con la incerteza configurada."""
    sn = state["sigma_n_current"]
    return float(state["g"](x) + RNG.normal(0.0, sn)), sn


def _argmax_random_ties(values, rel_tol=1e-3):
    """argmax con empate aleatorio.
    Cuando varios puntos del grid tienen valor casi idéntico (típico en zonas
    donde la varianza posterior es ≈ σ_f²), `np.argmax` devuelve siempre el
    primer índice y sesga la selección hacia un extremo del dominio. Acá se
    elige al azar entre todos los puntos cuyo valor está dentro de
    `rel_tol·rango` del máximo.
    """
    values = np.asarray(values)
    span = max(values.max() - values.min(), 1e-12)
    threshold = values.max() - rel_tol * span
    candidates = np.flatnonzero(values >= threshold)
    return int(RNG.choice(candidates))


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
        y_new = float(state["g"](x_new) + RNG.normal(0.0, sn))

    _add_observation(x_new, y_new, sn)


fig.canvas.mpl_connect("button_press_event", on_click)


# ===========================================================
# Sliders y selección de kernel (zona inferior)
# ===========================================================

# Slider de σ_n: ruido de observación, NO es parámetro del kernel.
ax_sn = plt.axes([0.32, 0.290, 0.55, 0.020])
sl_sn = Slider(ax_sn, r"$\sigma_n$  (incerteza obs.)", 0.01, 1.0, valinit=0.2)


def _on_sn(v):
    state["sigma_n_current"] = float(v)


sl_sn.on_changed(_on_sn)


# Recuadro "Kernel": radio + sliders de hiperparámetros + fórmula.
_add_group_box(0.025, 0.040, 0.945, 0.225, "Kernel")

ax_radio = plt.axes([0.045, 0.060, 0.13, 0.180])
radio = RadioButtons(ax_radio, list(KERNELS.keys()), active=0)

ax_ls = plt.axes([0.32, 0.215, 0.55, 0.018])
sl_ls = Slider(ax_ls, r"$\ell$  (escala)", 0.05, 5.0, valinit=1.0)

ax_sf = plt.axes([0.32, 0.175, 0.55, 0.018])
sl_sf = Slider(ax_sf, r"$\sigma_f$  (amplitud)", 0.1, 3.0, valinit=1.0)

formula_text = fig.text(
    0.32, 0.090, KERNEL_FORMULAS["RBF"],
    fontsize=12, va="center", ha="left",
)


def _on_ls(v):
    state["ls"] = float(v)
    redraw()


def _on_sf(v):
    state["sf"] = float(v)
    redraw()


def _on_kernel(label):
    state["kernel"] = label
    formula_text.set_text(KERNEL_FORMULAS[label])
    fig.canvas.draw_idle()
    redraw()


sl_ls.on_changed(_on_ls)
sl_sf.on_changed(_on_sf)
radio.on_clicked(_on_kernel)


# ===========================================================
# Botones (agrupados visualmente en 3 recuadros)
# ===========================================================
# --- Grupo: Vista ---
_add_group_box(0.025, 0.880, 0.21, 0.045, "Vista")
ax_check_g = plt.axes([0.04, 0.882, 0.18, 0.040])
ax_check_g.set_facecolor("none")
chk_g = CheckButtons(ax_check_g, ["Mostrar g(x)"], actives=[False])


def _on_toggle_g(label):
    visible = not true_line.get_visible()
    true_line.set_visible(visible)
    true_line.set_label(TRUE_LINE_LABEL if visible else "_nolegend_")
    ax.legend(loc="upper right")
    fig.canvas.draw_idle()


chk_g.on_clicked(_on_toggle_g)


# --- Grupo: Exploración ---
explore_label = _add_group_box(0.025, 0.790, 0.21, 0.05, "Exploración")
ax_explore = plt.axes([0.04, 0.795, 0.18, 0.04])
btn_explore = Button(ax_explore, "Paso de exploración", color="#cde7ff", hovercolor="#a8d2f8")


def _on_explore(event):
    """Agrega un punto donde la varianza posterior es máxima."""
    if len(state["X"]) == 0:
        # Sin datos, la varianza prior es uniforme: muestreo aleatorio en el dominio.
        x_new = float(RNG.uniform(X_MIN, X_MAX))
    else:
        _, var = _current_posterior()
        x_new = float(Xs[_argmax_random_ties(var)])
    y_new, sn = _sample_g_at(x_new)
    state["n_explore"] += 1
    _add_observation(x_new, y_new, sn)


btn_explore.on_clicked(_on_explore)


# --- Grupo: Explotación ---
exploit_label = _add_group_box(0.025, 0.560, 0.21, 0.185, "Explotación")
ax_exploit = plt.axes([0.04, 0.690, 0.18, 0.04])
btn_exploit = Button(ax_exploit, "Paso argmax(μ)", color="#ffd8c2", hovercolor="#f8b690")


def _on_exploit(event):
    """Agrega un punto donde la media posterior es máxima (greedy)."""
    if len(state["X"]) == 0:
        # Sin datos, la media prior es 0 en todos lados: muestreo aleatorio.
        x_new = float(RNG.uniform(X_MIN, X_MAX))
    else:
        mu, _ = _current_posterior()
        x_new = float(Xs[_argmax_random_ties(mu)])
    y_new, sn = _sample_g_at(x_new)
    state["n_exploit"] += 1
    _add_observation(x_new, y_new, sn)


btn_exploit.on_clicked(_on_exploit)


ax_ucb = plt.axes([0.04, 0.640, 0.18, 0.04])
btn_ucb = Button(ax_ucb, "Paso UCB", color="#e8d6ff", hovercolor="#c9adf2")


def _on_ucb(event):
    """Agrega un punto en argmax(μ + β·σ): balance entre explotación y exploración."""
    beta = state["ucb_beta"]
    if len(state["X"]) == 0:
        # Sin datos: μ=0 y σ=σ_f en todo Xs => UCB plano. Muestreo aleatorio.
        x_new = float(RNG.uniform(X_MIN, X_MAX))
    else:
        mu, var = _current_posterior()
        acq = mu + beta * np.sqrt(var)
        x_new = float(Xs[_argmax_random_ties(acq)])
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


def _on_ei(event):
    """Agrega un punto en argmax de Expected Improvement sobre el mejor y observado."""
    if len(state["X"]) == 0:
        # Sin datos: f* indefinido, EI plana => muestreo aleatorio.
        x_new = float(RNG.uniform(X_MIN, X_MAX))
    else:
        mu, var = _current_posterior()
        sd = np.sqrt(var)
        f_star = max(state["y"])
        eps = 1e-9
        z = (mu - f_star) / np.maximum(sd, eps)
        ei = (mu - f_star) * norm.cdf(z) + sd * norm.pdf(z)
        ei = np.where(sd < eps, 0.0, ei)
        x_new = float(Xs[_argmax_random_ties(ei)])
    y_new, sn = _sample_g_at(x_new)
    state["n_exploit"] += 1
    _add_observation(x_new, y_new, sn)


btn_ei.on_clicked(_on_ei)


# --- Grupo: Control ---
_add_group_box(0.025, 0.340, 0.21, 0.185, "Control")
ax_regen = plt.axes([0.04, 0.465, 0.18, 0.04])
btn_regen = Button(ax_regen, "Regenerar g(x)", color="#d9f5d2", hovercolor="#b6e6a8")


def _on_regen(event):
    """Sortea una nueva g(x) bimodal y limpia las observaciones."""
    state["g"] = make_random_g()
    state["X"].clear()
    state["y"].clear()
    state["sn"].clear()
    state["history_means"].clear()
    state["n_explore"] = 0
    state["n_exploit"] = 0
    true_line.set_data(Xs, state["g"](Xs))
    redraw()


btn_regen.on_clicked(_on_regen)


ax_clr_hist = plt.axes([0.04, 0.415, 0.18, 0.04])
btn_clr_hist = Button(ax_clr_hist, "Limpiar historia")


def _on_clr_hist(event):
    state["history_means"].clear()
    redraw()


btn_clr_hist.on_clicked(_on_clr_hist)


ax_reset = plt.axes([0.04, 0.360, 0.18, 0.04])
btn_reset = Button(ax_reset, "Reset todo")


def _on_reset(event):
    state["X"].clear()
    state["y"].clear()
    state["sn"].clear()
    state["history_means"].clear()
    state["n_explore"] = 0
    state["n_exploit"] = 0
    redraw()


btn_reset.on_clicked(_on_reset)


# ===========================================================
# Inicio
# ===========================================================
redraw()
plt.show()
