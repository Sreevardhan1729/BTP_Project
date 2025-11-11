from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple
import json
import numpy as np
import pandas as pd
from pathlib import Path

from ..utils.io import get_logger
from .chromosome import (
    GAParams, Population, init_population, tournament_select,
    uniform_crossover, mutate, enforce_min_features
)
from .fitness import CVCfg, ClassifierCfg, ensure_xy, cv_metric_for_mask

logger = get_logger("fdof.efsa")

@dataclass
class EFSAConfig:
    ga: GAParams
    cv: CVCfg
    clf: ClassifierCfg
    maximize_metric: Literal["accuracy", "f1"] = "accuracy"
    seed: int = 42

@dataclass
class GAHistoryItem:
    generation: int
    best_score: float
    mean_score: float
    best_on_count: int

class EFSA:
    def __init__(self, df: pd.DataFrame, label_col: str, cfg: EFSAConfig):
        self.df = df
        self.label_col = label_col
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.X, self.y, self.feat_cols = ensure_xy(df, label_col)
        self.n_features = len(self.feat_cols)

        if self.cfg.ga.min_features > self.n_features:
            logger.warning("min_features exceeds total features; reducing to n_features.")
            self.cfg.ga.min_features = self.n_features

        self.cache: Dict[str, float] = {}  # mask_key -> fitness
        self.history: List[GAHistoryItem] = []

    def _mask_key(self, mask: np.ndarray) -> str:
        # Compact key for caching
        return mask.astype(np.uint8).tobytes().hex()

    def _fitness(self, mask: np.ndarray) -> float:
        on = int(mask.sum())
        if on < self.cfg.ga.min_features:
            return 0.0
        key = self._mask_key(mask)
        if key in self.cache:
            return self.cache[key]
        score = cv_metric_for_mask(
            X=self.X, y=self.y, mask=mask, cv=self.cfg.cv,
            clf_cfg=self.cfg.clf, metric=self.cfg.maximize_metric
        )
        self.cache[key] = score
        return score

    def run(self) -> Tuple[np.ndarray, float]:
        pop = init_population(self.n_features, self.cfg.ga, self.rng)

        # Evaluate initial population
        for i in range(self.cfg.ga.pop_size):
            pop.fitness[i] = self._fitness(pop.bits[i])

        best_idx = int(np.argmax(pop.fitness))
        best_mask = pop.bits[best_idx].copy()
        best_score = float(pop.fitness[best_idx])
        stagnation = 0

        self.history.append(GAHistoryItem(
            generation=0,
            best_score=best_score,
            mean_score=float(np.mean(pop.fitness)),
            best_on_count=int(best_mask.sum()),
        ))
        logger.info(f"[gen=0] best={best_score:.4f} on={int(best_mask.sum())} mean={np.mean(pop.fitness):.4f}")

        for gen in range(1, self.cfg.ga.generations + 1):
            new_bits = []
            # Elitism: keep current best
            new_bits.append(best_mask.copy())

            # Fill rest via tournament, crossover, mutation
            while len(new_bits) < self.cfg.ga.pop_size:
                p1 = tournament_select(pop, self.cfg.ga.tournament_size, self.rng)
                p2 = tournament_select(pop, self.cfg.ga.tournament_size, self.rng)
                c1, c2 = uniform_crossover(pop.bits[p1], pop.bits[p2], self.cfg.ga.crossover_prob, self.rng)
                mutate(c1, self.cfg.ga.mutation_prob, self.rng)
                mutate(c2, self.cfg.ga.mutation_prob, self.rng)
                enforce_min_features(c1, self.cfg.ga.min_features, self.rng)
                enforce_min_features(c2, self.cfg.ga.min_features, self.rng)
                new_bits.extend([c1, c2])

            # Truncate in case of odd sizes
            new_bits = np.array(new_bits[: self.cfg.ga.pop_size], dtype=bool)

            # Evaluate
            new_fitness = np.zeros(self.cfg.ga.pop_size, dtype=float)
            for i in range(self.cfg.ga.pop_size):
                new_fitness[i] = self._fitness(new_bits[i])

            pop = Population(bits=new_bits, fitness=new_fitness)

            gen_best_idx = int(np.argmax(pop.fitness))
            gen_best_score = float(pop.fitness[gen_best_idx])
            gen_best_mask = pop.bits[gen_best_idx]

            if gen_best_score > best_score:
                best_score = gen_best_score
                best_mask = gen_best_mask.copy()
                stagnation = 0
            else:
                stagnation += 1

            self.history.append(GAHistoryItem(
                generation=gen,
                best_score=gen_best_score,
                mean_score=float(np.mean(pop.fitness)),
                best_on_count=int(gen_best_mask.sum()),
            ))
            logger.info(f"[gen={gen}] best={gen_best_score:.4f} on={int(gen_best_mask.sum())} mean={np.mean(pop.fitness):.4f}")

            if stagnation >= self.cfg.ga.early_stop_patience:
                logger.info(f"Early stopping at generation {gen} (no improvement for {stagnation} gens).")
                break

        return best_mask, best_score

    def save_outputs(
        self,
        best_mask: np.ndarray,
        out_selected_json: str,
        out_mask_npy: str,
        out_history_csv: str,
        out_report_json: str,
    ) -> None:
        sel_cols = [c for c, m in zip(self.feat_cols, best_mask) if m]
        p_json = Path(out_selected_json); p_json.parent.mkdir(parents=True, exist_ok=True)
        with open(p_json, "w", encoding="utf-8") as f:
            json.dump({"selected_features": sel_cols}, f, indent=2)

        p_npy = Path(out_mask_npy); p_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(p_npy, best_mask.astype(bool))

        # History CSV
        p_hist = Path(out_history_csv); p_hist.parent.mkdir(parents=True, exist_ok=True)
        import pandas as pd
        hist_df = pd.DataFrame([{
            "generation": h.generation,
            "best_score": h.best_score,
            "mean_score": h.mean_score,
            "best_on_count": h.best_on_count,
        } for h in self.history])
        hist_df.to_csv(p_hist, index=False)

        # Report JSON
        p_rep = Path(out_report_json); p_rep.parent.mkdir(parents=True, exist_ok=True)
        with open(p_rep, "w", encoding="utf-8") as f:
            json.dump({
                "metric": self.cfg.maximize_metric,
                "best_score": float(max(h.best_score for h in self.history)),
                "best_on_count": int(sum(best_mask)),
                "total_features": int(self.n_features),
            }, f, indent=2)

def filter_and_save_by_mask(
    mask: np.ndarray,
    label_col: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_train: str,
    out_val: str,
    out_test: str,
) -> None:
    feat_cols = [c for c in train_df.columns if c != label_col]
    sel_cols = [c for c, m in zip(feat_cols, mask) if m]
    # Preserve label
    tr = pd.concat([train_df[sel_cols], train_df[[label_col]]], axis=1)
    va = pd.concat([val_df[sel_cols],   val_df[[label_col]]], axis=1)
    te = pd.concat([test_df[sel_cols],  test_df[[label_col]]], axis=1)

    pt = Path(out_train); pt.parent.mkdir(parents=True, exist_ok=True)
    pv = Path(out_val);   pv.parent.mkdir(parents=True, exist_ok=True)
    pte = Path(out_test); pte.parent.mkdir(parents=True, exist_ok=True)

    tr.to_csv(pt, index=False)
    va.to_csv(pv, index=False)
    te.to_csv(pte, index=False)

    logger.info(f"Saved selected feature CSVs -> {pt} | {pv} | {pte}")
