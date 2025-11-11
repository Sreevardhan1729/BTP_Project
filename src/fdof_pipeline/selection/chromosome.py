from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from typing import Tuple

@dataclass
class GAParams:
    pop_size: int = 40
    generations: int = 30
    tournament_size: int = 3
    crossover_prob: float = 0.9
    mutation_prob: float = 0.01
    min_features: int = 10
    early_stop_patience: int = 6

@dataclass
class Population:
    bits: np.ndarray  # shape: (pop_size, n_features) boolean
    fitness: np.ndarray  # shape: (pop_size,) float
    eval_flags: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))  # optional

def init_population(n_features: int, params: GAParams, rng: np.random.Generator) -> Population:
    """Initialize population with a bias toward ~50% features on, but enforce min_features."""
    p = np.clip(0.5, 0.05, 0.95)
    bits = rng.random((params.pop_size, n_features)) < p
    # Enforce min_features
    for i in range(params.pop_size):
        on = bits[i].sum()
        if on < params.min_features:
            # randomly turn on additional features
            idx = rng.choice(n_features, size=params.min_features - on, replace=False)
            bits[i, idx] = True
    fitness = np.full(params.pop_size, -np.inf, dtype=float)
    return Population(bits=bits, fitness=fitness)

def tournament_select(pop: Population, k: int, rng: np.random.Generator) -> int:
    """Return index of selected individual via tournament selection."""
    idxs = rng.integers(0, pop.bits.shape[0], size=k)
    best = idxs[0]
    best_fit = pop.fitness[best]
    for j in idxs[1:]:
        if pop.fitness[j] > best_fit:
            best = j
            best_fit = pop.fitness[j]
    return best

def uniform_crossover(a: np.ndarray, b: np.ndarray, prob: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform crossover with probability prob; otherwise children are copies."""
    n = a.shape[0]
    if rng.random() >= prob:
        return a.copy(), b.copy()
    mask = rng.random(n) < 0.5
    c1 = np.where(mask, a, b)
    c2 = np.where(mask, b, a)
    return c1, c2

def mutate(bits: np.ndarray, prob: float, rng: np.random.Generator) -> None:
    """In-place bit-flip mutation with per-bit probability."""
    if prob <= 0.0:
        return
    flips = rng.random(bits.shape) < prob
    bits ^= flips  # xor to flip booleans in place

def enforce_min_features(indiv: np.ndarray, min_features: int, rng: np.random.Generator) -> None:
    """Ensure at least min_features are ON; if not, randomly turn on more."""
    on = int(indiv.sum())
    if on >= min_features:
        return
    n = indiv.shape[0]
    off_idx = np.where(~indiv)[0]
    if off_idx.size == 0:
        return
    k = min(min_features - on, off_idx.size)
    turn_on = rng.choice(off_idx, size=k, replace=False)
    indiv[turn_on] = True
