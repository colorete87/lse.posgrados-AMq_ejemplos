# k-Nearest Neighbors interactivo (2D)

Demo interactivo del clasificador k-NN sobre el plano `(x1, x2) ∈ R²`.
Permite jugar con `k`, la métrica de distancia, la función de peso y el
tamaño del dataset, y ver en vivo cómo cambian la región de decisión, la
predicción de un punto cualquiera y la accuracy sobre un test set.

## Instalación

Requiere [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Ejecución

```bash
uv run python knn_interactive.py
```

Si tu backend de matplotlib no es interactivo:

```bash
MPLBACKEND=TkAgg uv run python knn_interactive.py
```

## Uso

**Click sobre el plano** evalúa un punto cualquiera: marca la query con
`x` y dibuja líneas a sus `k` vecinos más cercanos. El título del plot
reporta clase predicha y confianza.

**Recuadros del panel izquierdo (de arriba abajo):**

- **Vista** — checkboxes para mostrar/ocultar la región de decisión y los
  test points.
- **Evaluación** — `Eval. punto (click)` (recordatorio visual; el click
  siempre evalúa), `Test completo` (clasifica todo `X_test` y reporta
  accuracy), `K-fold CV (5)` (5-fold sobre el train).
- **Control** — `Regenerar datos`, `Reset todo`.

**Panel derecho:** lista las últimas evaluaciones (test y k-fold) con
`k`, métrica, pesos y accuracy.

**Zona inferior (sliders + recuadros):**

- Sliders sueltos: `# clases` (2-6), `# train` (10-500), `# test` (5-200), `k` (1-25).
- Recuadro **Métrica**: Euclidiana / Manhattan / Minkowski / Mahalanobis.
  El slider `p` se activa cuando elegís Minkowski.
- Recuadro **Pesos**: uniform / 1/d / gaussiano. El slider `h` se activa
  cuando elegís gaussiano.

Mahalanobis usa la covarianza global del train (un solo `Σ` para todas
las clases). Empates de votos se resuelven por menor índice de clase.
