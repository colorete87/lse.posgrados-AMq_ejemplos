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
