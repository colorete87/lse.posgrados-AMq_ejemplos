# Proceso Gaussiano interactivo (2D)

Versión 2D del demo: la entrada es ahora `x = (x1, x2) ∈ R²`. La función real
`g(x)` es una **mezcla de dos gaussianas bivariadas** (cada una con posible
correlación entre `x1` y `x2`), con una moda claramente mayor que la otra.

`g(x)` y la posterior `μ(x)` se grafican como **líneas de nivel** (un color
por superficie: verde para g, azul para μ). Los máximos se marcan con una
"X". No se grafica mapa de calor.

## Instalación

Requiere [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Ejecución

```bash
uv run python gp_interactive_2d.py
```

Si tu backend de matplotlib no es interactivo:

```bash
MPLBACKEND=TkAgg uv run python gp_interactive_2d.py
```

## Uso

**Click sobre el plano (x1, x2)** agrega una observación en la posición del
cursor con `y = g(x) + ruido`.

**Recuadros del panel izquierdo (de arriba abajo):**

- **Vista** — `Mostrar g(x)`: alterna visibilidad de los contornos verdes
  y de la X verde del máximo real (default oculto).
- **Exploración** — `Paso de exploración`: agrega un punto en `argmax σ²(x)`.
- **Explotación** — 3 mecanismos:
  - `Paso argmax(μ)`: greedy puro sobre la media.
  - `Paso UCB`: `argmax(μ + β·σ)` con `β` ajustable por slider.
  - `Paso EI`: Expected Improvement sobre el mejor `y` observado.
- **Control** — `Regenerar g(x)` (sortea nueva mezcla bimodal y reinicia el
  estado), `Limpiar historia`, `Reset todo`.

**Zona inferior:**

- Slider `σ_n`: incerteza con que se incorpora la próxima observación.
- Recuadro **Kernel**: radio de selección del kernel + sliders `ℓ` (escala)
  y `σ_f` (amplitud) + fórmula `k(x, x')` que se actualiza al cambiar.

Los contornos azules tenues son **posteriors anteriores** que se desvanecen
gradualmente al sumar nuevas observaciones (más viejas → más transparentes).
Los contadores en los títulos de Exploración / Explotación cuentan los pasos
de cada estrategia.
