# Proceso Gaussiano interactivo (1D)

Demo educativo de un Proceso Gaussiano. Muestra una función real bimodal
`g(x)` (oculta para el GP) y permite entrenar el modelo agregando
observaciones ruidosas con el mouse o mediante distintas estrategias de
adquisición (exploración, explotación, UCB, EI).

## Instalación

Requiere [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Ejecución

```bash
uv run python gp_interactive.py
```

Si tu backend de matplotlib no es interactivo:

```bash
MPLBACKEND=TkAgg uv run python gp_interactive.py
```

## Uso

**Sliders**
- `σ_n` — incerteza con la que se incorpora la próxima observación.
- `ℓ` — escala de longitud del kernel.
- `σ_f` — amplitud (varianza prior) del kernel.
- `β` — peso del término de varianza para UCB.

**Click sobre el gráfico**
- Click izquierdo → observación de `g(x) + ruido` en el `x` del cursor.
- Click derecho (o `Shift` + click) → punto arbitrario en `(x, y)`.

**Botones de adquisición** (cada click agrega un punto)
- `Exploración` — argmax `σ²(x)` (cubre las zonas más inciertas).
- `Explotación` — argmax `μ(x)` (greedy puro, busca el máximo aparente).
- `UCB` — argmax `μ(x) + β·σ(x)` (balance ajustable con el slider de β).
- `EI` — Expected Improvement sobre `f* = max(y_obs)` (auto-balanceado, sin
  hiperparámetro; tiende a evitar re-muestrear donde `σ` ya es chica).

**Otros botones**
- `Regenerar g(x)` — sortea una nueva función bimodal y limpia el estado.
- `Limpiar historia` — borra las posteriors anteriores (deja solo la actual).
- `Reset todo` — reinicia observaciones y contadores.

**Visualización**
- Verde a rayas: `g(x)` real.
- Gris punteado: prior `f(x) = 0`.
- Azul sólido + banda celeste: posterior actual con `μ ± 2σ`.
- Trazos azules tenues: posteriors anteriores (más viejas → más transparentes).
- Puntos rojos con barras: observaciones y su incerteza `σ_n` por punto.
- Esquina superior izquierda: contador de iteraciones de Exploración / Explotación.

**Kernel** (radio buttons): RBF, Matérn 3/2, Matérn 5/2, Rational Quadratic, Periodic.
