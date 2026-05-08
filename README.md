# Ejemplos — Aprendizaje de Máquinas (Posgrado LSE)

Colección de ejemplos didácticos. Cada subcarpeta es un proyecto
autocontenido con su propio `pyproject.toml`, gestionado con
[uv](https://docs.astral.sh/uv/).

## Ejemplos

| Carpeta | Tema |
|---|---|
| [`proceso_gausiano/`](proceso_gausiano/) | Demo interactivo de Procesos Gaussianos en 1D (regresión y funciones de adquisición). |

## Cómo correr un ejemplo

```bash
cd <carpeta-del-ejemplo>
uv sync
uv run python <script>.py
```

Por ejemplo:

```bash
cd proceso_gausiano
uv sync
uv run python gp_interactive.py
```
