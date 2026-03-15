"""
Système de logs pour le trading bot.
  - Résumé hebdomadaire (PnL, nb trades, Sharpe) dans logs/
  - Intégration TensorBoard
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from config.settings import LOGS_DIR

logger = logging.getLogger(__name__)

# Sous-dossiers de logs
WEEKLY_LOG_DIR = LOGS_DIR / "weekly"
BACKTEST_LOG_DIR = LOGS_DIR / "backtests"
WEEKLY_CSV_PATH = LOGS_DIR / "weekly_summary.csv"
MONTHLY_CSV_PATH = LOGS_DIR / "monthly_summary.csv"

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


def _ensure_dirs():
    """Crée les dossiers de logs si nécessaire."""
    WEEKLY_LOG_DIR.mkdir(parents=True, exist_ok=True)
    BACKTEST_LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_weekly_summary(
    stats: dict,
    week_label: Optional[str] = None,
) -> Path:
    """
    Sauvegarde un résumé hebdomadaire au format JSON.

    Args:
        stats: dict de métriques (PnL, Sharpe, trades, etc.)
        week_label: label de la semaine (auto-généré si None)

    Returns:
        Chemin du fichier sauvegardé
    """
    _ensure_dirs()

    if week_label is None:
        now = datetime.now()
        week_label = f"{now.year}_W{now.isocalendar()[1]:02d}"

    filepath = WEEKLY_LOG_DIR / f"week_{week_label}.json"

    entry = {
        "week": week_label,
        "timestamp": datetime.now().isoformat(),
        **{k: _serialize(v) for k, v in stats.items()},
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)

    logger.info(f"Résumé hebdomadaire sauvegardé: {filepath}")
    return filepath


def append_weekly_csv(
    stats: dict,
    week_label: Optional[str] = None,
    csv_path: Optional[Path] = None,
) -> Path:
    """
    Ajoute une ligne au CSV cumulatif hebdomadaire.

    Args:
        stats: dict de métriques (doit contenir les clés de WEEKLY_CSV_COLUMNS)
        week_label: label de la semaine (auto-généré si None)
        csv_path: chemin du CSV (défaut: logs/weekly_summary.csv)

    Returns:
        Chemin du fichier CSV
    """
    if csv_path is None:
        csv_path = WEEKLY_CSV_PATH
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if week_label is None:
        now = datetime.now()
        week_label = f"{now.year}_W{now.isocalendar()[1]:02d}"

    # Construire la ligne avec les colonnes attendues
    row = {
        "week": week_label,
        "timestamp": datetime.now().isoformat(),
    }
    for col in WEEKLY_CSV_COLUMNS:
        if col not in row:
            row[col] = _serialize(stats.get(col, ""))

    # Créer le fichier avec header si nécessaire, sinon append
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
    csv_path: Optional[Path] = None,
) -> Path:
    """
    Ajoute une ligne au CSV cumulatif mensuel.

    Args:
        stats: dict de métriques (doit contenir les clés de MONTHLY_CSV_COLUMNS)
        month_label: label du mois ex: "2026_03" (auto-généré si None)
        csv_path: chemin du CSV (défaut: logs/monthly_summary.csv)

    Returns:
        Chemin du fichier CSV
    """
    if csv_path is None:
        csv_path = MONTHLY_CSV_PATH
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
            row[col] = _serialize(stats.get(col, ""))

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
) -> Path:
    """
    Sauvegarde les résultats d'un backtest au format JSON.

    Args:
        stats: dict de métriques du backtest
        model_name: nom du modèle utilisé
        run_name: nom du run (auto-généré si None)

    Returns:
        Chemin du fichier sauvegardé
    """
    _ensure_dirs()

    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    filepath = BACKTEST_LOG_DIR / f"backtest_{model_name}_{run_name}.json"

    entry = {
        "run_name": run_name,
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        **{k: _serialize(v) for k, v in stats.items()},
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)

    logger.info(f"Résultat backtest sauvegardé: {filepath}")
    return filepath


def load_backtest_results(model_name: Optional[str] = None) -> list[dict]:
    """
    Charge tous les résultats de backtest pour comparaison.

    Args:
        model_name: filtrer par nom de modèle (None = tous)

    Returns:
        Liste de dicts de résultats, triés par date
    """
    _ensure_dirs()
    results = []

    for filepath in sorted(BACKTEST_LOG_DIR.glob("backtest_*.json")):
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
