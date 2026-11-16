"""
evolution.py
============
Hardware-aware EVOLUTIONARY architecture search on CIFAR-10.

Algorithm
---------
1. Initial population: sample random architectures until POPULATION_SIZE
   feasible ones are found.
2. For every generation:
     a. pick a parent with a tournament (best fitness of TOURNAMENT_SIZE random individuals),
     b. mutate it to create a child (see search_space.mutate),
     c. evaluate the child; keep it only if it is feasible,
     d. repeat until POPULATION_SIZE children exist,
     e. survivors = the best POPULATION_SIZE of (parents + children).
3. The best architecture ever evaluated is saved to output/best_architecture.json.

Evaluating one candidate (evaluate_candidate)
---------------------------------------------
  build network -> MEASURE latency / size / peak memory on the CPU
  -> violates a constraint?  YES: reject, never train it.
                             NO : train for SEARCH_EPOCHS on the train split,
                                  score on the validation split.

Fitness = val_accuracy/100 - LATENCY_PENALTY * latency/MAX_LATENCY
                           - SIZE_PENALTY    * size/MAX_SIZE
so among similarly accurate networks the faster and smaller one wins.

Every evaluated candidate (rejected ones included) is appended to
output/search_history.csv immediately, so partial results survive a crash.

Usage:
    python evolution.py [--download] [--population 10] [--generations 6] [--search_epochs 3]
"""

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch

import config
from data import CifarData
from hardware import check_constraints, profile_model
from model_builder import build_model
from search_space import Architecture, mutate, random_architecture
from trainer import get_device, train_model

# Column order of search_history.csv
CSV_FIELDS = ["candidate_id", "generation", "resolution", "num_blocks", "params_million",
              "latency_ms", "model_size_mb", "peak_memory_mb", "feasible", "reject_reason",
              "val_accuracy", "fitness", "train_seconds", "architecture"]


@dataclass
class Individual:
    """One feasible, evaluated member of the population."""
    arch: Architecture
    record: dict

    @property
    def fitness(self) -> float:
        return self.record["fitness"]


def _clean(value):
    """Replace NaN by None so CSV/JSON store an empty value / null."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def compute_fitness(val_accuracy: float, latency_ms: float, size_mb: float) -> float:
    """Accuracy (0-1) minus small penalties for latency and size (see module docstring)."""
    return (val_accuracy / 100.0
            - config.LATENCY_PENALTY * latency_ms / config.MAX_LATENCY_MS
            - config.SIZE_PENALTY * size_mb / config.MAX_MODEL_SIZE_MB)


class SearchHistory:
    """Keeps every candidate record, appending each one to the CSV file immediately."""

    def __init__(self):
        config.ensure_output_dirs()
        self.records: List[dict] = []
        with open(config.SEARCH_HISTORY_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    def add(self, record: dict) -> None:
        """Store a record in memory and append it to the CSV."""
        self.records.append(record)
        row = dict(record)
        row["architecture"] = json.dumps(record["architecture"])
        with open(config.SEARCH_HISTORY_CSV, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow({k: row[k] for k in CSV_FIELDS})

    def save_json(self) -> None:
        """Write the complete history as JSON."""
        with open(config.SEARCH_HISTORY_JSON, "w") as f:
            json.dump(self.records, f, indent=2)


def evaluate_candidate(arch: Architecture, candidate_id: int, generation: int,
                       data: CifarData, device: torch.device) -> dict:
    """Measure, check constraints and (if feasible) train + validate one architecture."""
    model = build_model(arch)
    measures = profile_model(model)                       # real CPU measurements
    feasible, reason = check_constraints(measures["latency_ms"], measures["model_size_mb"])

    record = {
        "candidate_id": candidate_id, "generation": generation,
        "resolution": arch.resolution, "num_blocks": len(arch.blocks),
        "params_million": round(measures["params_million"], 4),
        "latency_ms": round(measures["latency_ms"], 3),
        "model_size_mb": round(measures["model_size_mb"], 4),
        "peak_memory_mb": _clean(round(measures["peak_memory_mb"], 3)),
        "feasible": feasible, "reject_reason": reason,
        "val_accuracy": None, "fitness": None, "train_seconds": None,
        "architecture": arch.to_dict(),
    }
    if feasible:                                           # rejected models are never trained
        start = time.time()
        val_acc = train_model(model, data.train_x, data.train_y, data.val_x, data.val_y,
                              config.SEARCH_EPOCHS, device)
        record["val_accuracy"] = round(val_acc, 2)
        record["fitness"] = compute_fitness(val_acc, measures["latency_ms"],
                                            measures["model_size_mb"])
        record["train_seconds"] = round(time.time() - start, 1)
    return record


def tournament_select(population: List[Individual], rng: random.Random) -> Individual:
    """Parent selection: the fittest of TOURNAMENT_SIZE randomly drawn individuals."""
    contenders = rng.sample(population, min(config.TOURNAMENT_SIZE, len(population)))
    return max(contenders, key=lambda ind: ind.fitness)


def run_search(data: CifarData, device: torch.device, population_size: int,
               generations: int) -> Optional[Individual]:
    """Run the evolutionary search and return the best individual found."""
    rng = random.Random(config.SEED)
    history = SearchHistory()
    seen = set()          # keys of already evaluated architectures (no duplicates)
    best: Optional[Individual] = None
    next_id = 0

    def try_candidate(arch: Architecture, generation: int) -> Optional[Individual]:
        """Evaluate a new architecture, log it, and return it if feasible."""
        nonlocal next_id
        seen.add(arch.key())
        record = evaluate_candidate(arch, next_id, generation, data, device)
        next_id += 1
        history.add(record)
        status = (f"val {record['val_accuracy']:.2f}% fitness {record['fitness']:.4f}"
                  if record["feasible"] else f"REJECTED ({record['reject_reason']})")
        print(f"[gen {generation}] cand {record['candidate_id']:3d} | res {arch.resolution} "
              f"blocks {len(arch.blocks)} | {record['latency_ms']:.2f} ms "
              f"{record['model_size_mb']:.2f} MB | {status}")
        return Individual(arch, record) if record["feasible"] else None

    max_attempts = config.MAX_ATTEMPTS_FACTOR * population_size

    # ---- Generation 0: random feasible population -------------------------------
    population: List[Individual] = []
    attempts = 0
    while len(population) < population_size and attempts < max_attempts:
        attempts += 1
        arch = random_architecture(rng)
        if arch.key() in seen:
            continue
        individual = try_candidate(arch, generation=0)
        if individual:
            population.append(individual)
    if not population:
        print("No feasible architecture found. Relax the constraints in config.py.")
        history.save_json()
        return None

    # ---- Evolution --------------------------------------------------------------
    for generation in range(1, generations + 1):
        children: List[Individual] = []
        attempts = 0
        while len(children) < population_size and attempts < max_attempts:
            attempts += 1
            parent = tournament_select(population, rng)
            arch = mutate(parent.arch, rng)
            if arch.key() in seen:
                continue
            individual = try_candidate(arch, generation)
            if individual:
                children.append(individual)

        # Survivor selection: keep the best of parents + children.
        population = sorted(population + children, key=lambda ind: ind.fitness,
                            reverse=True)[:population_size]
        print(f"--- generation {generation} done | best fitness {population[0].fitness:.4f} "
              f"| best val acc {population[0].record['val_accuracy']:.2f}% ---")
        history.save_json()

    # ---- Pick the overall best feasible candidate ever evaluated ----------------
    feasible_records = [r for r in history.records if r["feasible"]]
    best_record = max(feasible_records, key=lambda r: r["fitness"])
    best = Individual(Architecture.from_dict(best_record["architecture"]), best_record)
    history.save_json()
    return best


def save_best(best: Individual) -> None:
    """Write the winning architecture and its measurements to output/best_architecture.json."""
    metrics = {k: v for k, v in best.record.items() if k != "architecture"}
    with open(config.BEST_ARCH_JSON, "w") as f:
        json.dump({"architecture": best.arch.to_dict(), "search_metrics": metrics}, f, indent=2)
    print(f"Best architecture saved -> {config.BEST_ARCH_JSON}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evolutionary hardware-aware NAS")
    parser.add_argument("--data_dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--download", action="store_true", help="download CIFAR-10 if missing")
    parser.add_argument("--population", type=int, default=config.POPULATION_SIZE)
    parser.add_argument("--generations", type=int, default=config.GENERATIONS)
    parser.add_argument("--search_epochs", type=int, default=config.SEARCH_EPOCHS)
    args = parser.parse_args()
    config.SEARCH_EPOCHS = args.search_epochs   # picked up by evaluate_candidate

    torch.manual_seed(config.SEED)
    device = get_device()
    print(f"Device: {device}")
    data = CifarData(args.data_dir, args.download).to(device)

    best = run_search(data, device, args.population, args.generations)
    if best:
        save_best(best)


if __name__ == "__main__":
    main()
