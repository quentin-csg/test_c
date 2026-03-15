"""
Système de logs pour le trading bot.
  - Résumé hebdomadaire JSON + CSV cumulatif
  - Résumé mensuel CSV cumulatif
  - Résultats de backtest JSON
  - Séparation logs/train/ et logs/live/
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from config.settings import LOGS_TRAIN_DIR, LOGS_LIVE_DIR

logger = logging.getLogger(__name__)

Mode = Literal["train", "live"]


def _get_dirs(mode: Mode = "train") -> dict[str, Path]:
    """Retourne les chemins de logs pour un mode donné."""
    base = LOGS_TRAIN_DIR if mode == "train" else LOGS_LIVE_DIR
    return {
        "weekly": base / "weekly",
        "backtests": base / "backtests",
        "weekly_csv": base / "weekly_summary.csv",
        "monthly_csv": base / "monthly_summary.csv",
        "tensorboard": base / "tensorboard",
    }


WEEKLY_CSV_COLUMNS = [
    "week",
    "timestamp",
    "model_name",
    "pnl_usdt",
    "pnl_cumul_usdt",
    "net_worth",
    "total_return_pct",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown_pct",
    "total_trades",
    "total_fees",
    "mode",
]

MONTHLY_CSV_COLUMNS = [
    "month",
    "timestamp",
    "model_name",
    "pnl_usdt",
    "pnl_cumul_usdt",
    "net_worth",
    "total_return_pct",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown_pct",
    "total_trades",
    "total_fees",
    "mode",
]


def _ensure_dirs(mode: Mode = "train"):
    """Crée les dossiers de logs si nécessaire."""
    dirs = _get_dirs(mode)
    dirs["weekly"].mkdir(parents=True, exist_ok=True)
    dirs["backtests"].mkdir(parents=True, exist_ok=True)
    dirs["tensorboard"].mkdir(parents=True, exist_ok=True)


def log_weekly_summary(
    stats: dict,
    week_label: Optional[str] = None,
    mode: Mode = "train",
) -> Path:
    """
    Sauvegarde un résumé hebdomadaire au format JSON.

    Args:
        stats: dict de métriques (PnL, Sharpe, trades, etc.)
        week_label: label de la semaine (auto-généré si None)
        mode: "train" ou "live"

    Returns:
        Chemin du fichier sauvegardé
    """
    _ensure_dirs(mode)
    dirs = _get_dirs(mode)

    if week_label is None:
        now = datetime.now()
        week_label = f"{now.year}_W{now.isocalendar()[1]:02d}"

    filepath = dirs["weekly"] / f"week_{week_label}.json"

    entry = {
        "week": week_label,
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        **{k: _serialize(v) for k, v in stats.items()},
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)

    logger.info(f"Résumé hebdomadaire sauvegardé: {filepath}")
    return filepath


def append_weekly_csv(
    stats: dict,
    week_label: Optional[str] = None,
    mode: Mode = "train",
    csv_path: Optional[Path] = None,
) -> Path:
    """
    Ajoute une ligne au CSV cumulatif hebdomadaire.

    Args:
        stats: dict de métriques
        week_label: label de la semaine (auto-généré si None)
        mode: "train" ou "live"
        csv_path: chemin du CSV (auto si None)

    Returns:
        Chemin du fichier CSV
    """
    if csv_path is None:
        csv_path = _get_dirs(mode)["weekly_csv"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if week_label is None:
        now = datetime.now()
        week_label = f"{now.year}_W{now.isocalendar()[1]:02d}"

    row = {
        "week": week_label,
        "timestamp": datetime.now().isoformat(),
    }
    for col in WEEKLY_CSV_COLUMNS:
        if col not in row:
            val = stats.get(col, "")
            if col == "mode" and val == "":
                val = mode
            row[col] = _serialize(val)

    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=WEEKLY_CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    logger.info(f"Ligne ajoutée au CSV hebdomadaire: {csv_path}")
    return csv_path


def append_monthly_csv(
    stats: dict,
    month_label: Optional[str] = None,
    mode: Mode = "train",
    csv_path: Optional[Path] = None,
) -> Path:
    """
    Ajoute une ligne au CSV cumulatif mensuel.

    Args:
        stats: dict de métriques
        month_label: label du mois ex: "2026_03" (auto-généré si None)
        mode: "train" ou "live"
        csv_path: chemin du CSV (auto si None)

    Returns:
        Chemin du fichier CSV
    """
    if csv_path is None:
        csv_path = _get_dirs(mode)["monthly_csv"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if month_label is None:
        now = datetime.now()
        month_label = f"{now.year}_{now.month:02d}"

    row = {
        "month": month_label,
        "timestamp": datetime.now().isoformat(),
    }
    for col in MONTHLY_CSV_COLUMNS:
        if col not in row:
            val = stats.get(col, "")
            if col == "mode" and val == "":
                val = mode
            row[col] = _serialize(val)

    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MONTHLY_CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    logger.info(f"Ligne ajoutée au CSV mensuel: {csv_path}")
    return csv_path


def log_backtest_result(
    stats: dict,
    model_name: str = "ppo_trading",
    run_name: Optional[str] = None,
    mode: Mode = "train",
) -> Path:
    """
    Sauvegarde les résultats d'un backtest au format JSON.

    Args:
        stats: dict de métriques du backtest
        model_name: nom du modèle utilisé
        run_name: nom du run (auto-généré si None)
        mode: "train" ou "live"

    Returns:
        Chemin du fichier sauvegardé
    """
    _ensure_dirs(mode)
    dirs = _get_dirs(mode)

    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    filepath = dirs["backtests"] / f"backtest_{model_name}_{run_name}.json"

    entry = {
        "run_name": run_name,
        "model_name": model_name,
        "mode": mode,
        "timestamp": datetime.now().isoformat(),
        **{k: _serialize(v) for k, v in stats.items()},
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)

    logger.info(f"Résultat backtest sauvegardé: {filepath}")
    return filepath


def load_backtest_results(
    model_name: Optional[str] = None,
    mode: Mode = "train",
) -> list[dict]:
    """
    Charge tous les résultats de backtest pour comparaison.

    Args:
        model_name: filtrer par nom de modèle (None = tous)
        mode: "train" ou "live"

    Returns:
        Liste de dicts de résultats, triés par date
    """
    _ensure_dirs(mode)
    dirs = _get_dirs(mode)
    results = []

    for filepath in sorted(dirs["backtests"].glob("backtest_*.json")):
        if model_name and model_name not in filepath.name:
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            results.append(json.load(f))

    return results


def print_stats(stats: dict, title: str = "Résultats") -> None:
    """Affiche les statistiques de manière formatée."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key:.<35s} {value:>10.4f}")
        else:
            print(f"  {key:.<35s} {str(value):>10s}")
    print(f"{'='*50}\n")


def _serialize(value):
    """Convertit les types numpy pour la sérialisation JSON."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
