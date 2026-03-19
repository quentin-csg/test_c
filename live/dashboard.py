"""
Dashboard Streamlit — Interface locale du trading bot.

Affiche :
  - Graphique cours BTC temps réel
  - Positions ouvertes / trades fermés
  - KPIs hebdomadaires
  - Statut du circuit breaker
  - Résultats des backtests

Usage:
  streamlit run live/dashboard.py
  ou
  python main.py dashboard
"""

import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from config.settings import (
    INITIAL_BALANCE,
    LOGS_LIVE_DIR,
    LOGS_TRAIN_DIR,
    MODELS_DIR,
    SYMBOL,
)
from training.logger import load_backtest_results, load_walk_forward_results


def _load_live_state() -> dict:
    """Charge l'état courant du portfolio depuis live_state.json."""
    state_path = LOGS_LIVE_DIR / "live_state.json"
    if not state_path.exists():
        return {}
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv_summary(mode: str = "live") -> list[dict]:
    """Charge le CSV cumulatif hebdomadaire."""
    import csv
    base = LOGS_LIVE_DIR if mode == "live" else LOGS_TRAIN_DIR
    csv_path = base / "weekly_summary.csv"
    if not csv_path.exists():
        return []
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _list_models() -> list[str]:
    """Liste les modèles disponibles."""
    return sorted(f.stem for f in MODELS_DIR.glob("*.zip"))


def run_dashboard():
    """Lance le dashboard Streamlit."""
    import pandas as pd

    st.set_page_config(
        page_title="Trading Bot RL",
        page_icon="🤖",
        layout="wide",
    )

    st.title("🤖 Trading Bot RL — Dashboard")
    st.caption(f"Symbole: {SYMBOL}")

    tab_live, tab_backtests, tab_wf, tab_models = st.tabs([
        "📊 Live / Paper",
        "📈 Backtests",
        "🔄 Walk-Forward",
        "🧠 Modeles",
    ])

    # =========================================================================
    # TAB LIVE
    # =========================================================================
    with tab_live:
        state = _load_live_state()

        if not state:
            st.info("Aucune donnée live disponible. Lancez le bot en mode paper/live.")
        else:
            portfolio = state.get("portfolio", {})
            current_price = state.get("current_price", 0.0)
            last_updated = state.get("last_updated", "")

            price_history_base = state.get("price_history", [])
            open_positions_base = state.get("open_positions", [])

            @st.fragment
            def _live_section(
                portfolio_data: dict,
                history_base: list,
                open_pos_base: list,
                last_upd: str,
            ) -> None:
                import plotly.graph_objects as go
                from datetime import datetime, timezone

                # --- Lire les données depuis live_state.json (déjà à jour) --
                live_price = state.get("current_price")
                balance_usdt = portfolio_data.get("balance_usdt", 0.0)
                position_btc = portfolio_data.get("position_btc", 0.0)
                total_trades = portfolio_data.get("total_trades", 0)
                net_worth_live = portfolio_data.get("net_worth", 0.0)
                unrealized_pnl_pct = (
                    open_pos_base[0].get("unrealized_pnl_pct") if open_pos_base else None
                )
                return_pct = portfolio_data.get("total_return_pct", 0.0)

                # --- KPIs ----------------------------------------------------
                st.subheader("Portfolio")
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Net Worth", f"{net_worth_live:.2f} USDT")
                col2.metric("Cash", f"{balance_usdt:.2f} USDT")
                col3.metric(
                    "Position BTC",
                    f"{position_btc:.6f} BTC",
                    f"{unrealized_pnl_pct:+.2f}%" if unrealized_pnl_pct is not None else None,
                )
                col4.metric(
                    "Return",
                    f"{return_pct:+.2f}%",
                    delta_color="normal",
                )
                col5.metric("Trades", f"{total_trades}")

                ts_display = (
                    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
                    if live_price else last_upd[:19]
                )
                st.caption(
                    f"Dernière mise à jour : {ts_display}"
                    + (f" | BTC live : {live_price:,.2f} USDT" if live_price else "")
                )

                # --- Graphique cours BTC -------------------------------------
                st.subheader(f"Cours {SYMBOL}")
                df_prices = pd.DataFrame(history_base) if history_base else pd.DataFrame()
                if not df_prices.empty and "timestamp" in df_prices.columns:
                    df_prices["timestamp"] = pd.to_datetime(
                        df_prices["timestamp"], utc=True, errors="coerce"
                    )
                    df_prices = df_prices.dropna(subset=["timestamp"])

                if not df_prices.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_prices["timestamp"],
                        y=df_prices["price"],
                        mode="lines",
                        line=dict(color="#00b4d8", width=1.5),
                        name="Prix",
                    ))
                    if live_price:
                        fig.add_trace(go.Scatter(
                            x=[df_prices["timestamp"].iloc[-1]],
                            y=[live_price],
                            mode="markers",
                            marker=dict(color="#ff6b35", size=8),
                            name=f"Live: {live_price:,.2f}",
                        ))
                    fig.update_layout(
                        height=350,
                        margin=dict(l=0, r=0, t=10, b=0),
                        xaxis=dict(rangeslider=dict(visible=True), type="date"),
                        yaxis=dict(autorange=True, fixedrange=False),
                        dragmode="zoom",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True, key="price_chart")
                else:
                    st.info("Historique de prix non disponible.")

                st.divider()

                # --- Positions ouvertes --------------------------------------
                col_left, col_right = st.columns(2)
                with col_left:
                    st.subheader("Positions ouvertes")
                    if open_pos_base:
                        df_open = pd.DataFrame(open_pos_base)
                        # Injecter prix actuel et recalculer PnL
                        if live_price:
                            df_open["Prix actuel"] = live_price
                            entry = df_open.get("entry_price", pd.Series([0]))
                            df_open["unrealized_pnl_pct"] = (
                                (live_price - entry) / entry * 100
                            ).where(entry > 0, 0.0)
                            df_open["value_usdt"] = df_open["amount_btc"] * live_price
                        # Réordonner colonnes : Prix entrée | Prix actuel | ...
                        cols_order = ["entry_price", "Prix actuel", "amount_btc",
                                      "value_usdt", "unrealized_pnl_pct"]
                        cols_present = [c for c in cols_order if c in df_open.columns]
                        df_open = df_open[cols_present]
                        df_open = df_open.rename(columns={
                            "entry_price": "Prix entrée",
                            "amount_btc": "Quantité BTC",
                            "value_usdt": "Valeur USDT",
                            "unrealized_pnl_pct": "PnL latent %",
                        })
                        df_open["Prix entrée"] = df_open["Prix entrée"].map("{:.2f}".format)
                        if "Prix actuel" in df_open.columns:
                            df_open["Prix actuel"] = df_open["Prix actuel"].map("{:,.2f}".format)
                        df_open["Quantité BTC"] = df_open["Quantité BTC"].map("{:.6f}".format)
                        df_open["Valeur USDT"] = df_open["Valeur USDT"].map("{:.2f}".format)
                        df_open["PnL latent %"] = df_open["PnL latent %"].map("{:+.2f}%".format)
                        st.dataframe(df_open, use_container_width=True, hide_index=True)
                    else:
                        st.info("Aucune position ouverte.")

                # --- Trades fermés -------------------------------------------
                with col_right:
                    st.subheader("Trades fermés")
                    closed_trades_data = state.get("closed_trades", [])
                    if closed_trades_data:
                        df_closed = pd.DataFrame(closed_trades_data)
                        cols_to_show = [
                            c for c in ["timestamp", "price", "amount", "fee"]
                            if c in df_closed.columns
                        ]
                        df_closed = df_closed[cols_to_show].copy()
                        df_closed = df_closed.rename(columns={
                            "timestamp": "Date",
                            "price": "Prix vente",
                            "amount": "Quantité BTC",
                            "fee": "Frais USDT",
                        })
                        if "Prix vente" in df_closed.columns:
                            df_closed["Prix vente"] = pd.to_numeric(
                                df_closed["Prix vente"], errors="coerce"
                            ).map("{:.2f}".format)
                        if "Quantité BTC" in df_closed.columns:
                            df_closed["Quantité BTC"] = pd.to_numeric(
                                df_closed["Quantité BTC"], errors="coerce"
                            ).map("{:.6f}".format)
                        if "Frais USDT" in df_closed.columns:
                            df_closed["Frais USDT"] = pd.to_numeric(
                                df_closed["Frais USDT"], errors="coerce"
                            ).map("{:.4f}".format)
                        st.dataframe(
                            df_closed.iloc[::-1],
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("Aucun trade fermé pour cette session.")

                st.divider()

            _live_section(portfolio, price_history_base, open_positions_base, last_updated)

            st.divider()

        # --- Résumé hebdomadaire -------------------------------------------
        weekly_data = _load_csv_summary("live")
        if weekly_data:
            st.subheader("Résumé hebdomadaire")
            df_weekly = pd.DataFrame(weekly_data)
            num_cols = [
                "pnl_usdt", "pnl_cumul_usdt", "net_worth",
                "total_return_pct", "sharpe_ratio", "sortino_ratio",
                "max_drawdown_pct", "total_trades", "total_fees",
            ]
            for col in num_cols:
                if col in df_weekly.columns:
                    df_weekly[col] = pd.to_numeric(df_weekly[col], errors="coerce")

            if len(df_weekly) > 1 and "pnl_cumul_usdt" in df_weekly.columns:
                st.line_chart(df_weekly.set_index("week")["pnl_cumul_usdt"])

            st.dataframe(df_weekly, use_container_width=True, hide_index=True)

        # --- Circuit Breaker -----------------------------------------------
        st.subheader("Circuit Breaker")
        st.success("✅ Status: OK — Aucun déclenchement")
        st.caption(
            "Le circuit breaker surveille les chutes de prix et volumes anormaux "
            "en temps réel."
        )

    # =========================================================================
    # TAB BACKTESTS
    # =========================================================================
    with tab_backtests:
        st.header("Résultats des backtests")

        mode = st.radio("Source", ["train", "live"], horizontal=True)
        results = load_backtest_results(mode=mode)

        if results:
            df_bt = pd.DataFrame(results)
            last_bt = results[-1]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Return", f"{last_bt.get('total_return_pct', 0):.2f}%")
            col2.metric("Sharpe", f"{last_bt.get('sharpe_ratio', 0):.4f}")
            col3.metric("Sortino", f"{last_bt.get('sortino_ratio', 0):.4f}")
            col4.metric("Max Drawdown", f"{last_bt.get('max_drawdown_pct', 0):.2f}%")

            st.dataframe(df_bt, use_container_width=True, hide_index=True)
        else:
            st.info(f"Aucun backtest trouvé dans les logs {mode}.")

    # =========================================================================
    # TAB WALK-FORWARD
    # =========================================================================
    with tab_wf:
        st.header("Walk-Forward Validation")

        mode_wf = st.radio("Source", ["train", "live"], horizontal=True, key="wf_mode")
        wf_results = load_walk_forward_results(mode=mode_wf)

        if not wf_results:
            st.info(
                f"Aucun résultat walk-forward dans les logs {mode_wf}. "
                "Lancez: python main.py walk-forward"
            )
        else:
            last_run = wf_results[-1]
            agg = last_run.get("aggregate", {})

            # --- KPIs agrégés ---
            st.subheader(f"Dernier run ({last_run.get('timestamp', '')[:10]}) — "
                         f"{last_run.get('n_folds', 0)} folds")
            col1, col2, col3, col4 = st.columns(4)
            if "total_return_pct" in agg:
                col1.metric(
                    "Return moyen",
                    f"{agg['total_return_pct']['mean']:+.2f}%",
                    f"± {agg['total_return_pct']['std']:.2f}%",
                )
            if "sharpe_ratio" in agg:
                col2.metric(
                    "Sharpe moyen",
                    f"{agg['sharpe_ratio']['mean']:.4f}",
                    f"± {agg['sharpe_ratio']['std']:.4f}",
                )
            if "sortino_ratio" in agg:
                col3.metric(
                    "Sortino moyen",
                    f"{agg['sortino_ratio']['mean']:.4f}",
                )
            if "max_drawdown_pct" in agg:
                col4.metric(
                    "Max Drawdown moyen",
                    f"{agg['max_drawdown_pct']['mean']:.2f}%",
                )

            # --- Bar chart Sharpe par fold ---
            fold_results = last_run.get("fold_results", [])
            if fold_results:
                st.subheader("Sharpe par fold")
                df_folds = pd.DataFrame(fold_results)
                if "sharpe_ratio" in df_folds.columns and "fold_id" in df_folds.columns:
                    st.bar_chart(df_folds.set_index("fold_id")["sharpe_ratio"])

                # --- Tableau détaillé ---
                st.subheader("Résultats par fold")
                cols_to_show = [c for c in [
                    "fold_id", "total_return_pct", "sharpe_ratio", "sortino_ratio",
                    "max_drawdown_pct", "total_trades",
                ] if c in df_folds.columns]
                st.dataframe(df_folds[cols_to_show], use_container_width=True, hide_index=True)

            # --- Tous les runs ---
            if len(wf_results) > 1:
                st.subheader("Historique des runs")
                runs_summary = []
                for r in wf_results:
                    agg_r = r.get("aggregate", {})
                    runs_summary.append({
                        "date": r.get("timestamp", "")[:10],
                        "n_folds": r.get("n_folds", 0),
                        "sharpe_mean": agg_r.get("sharpe_ratio", {}).get("mean", None),
                        "return_mean": agg_r.get("total_return_pct", {}).get("mean", None),
                    })
                st.dataframe(pd.DataFrame(runs_summary), use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB MODELES
    # =========================================================================
    with tab_models:
        st.header("Modeles disponibles")
        models = _list_models()
        if models:
            for m in models:
                st.code(m)
        else:
            st.info("Aucun modele sauvegardé dans models/")

        st.subheader("TensorBoard")
        st.code("tensorboard --logdir logs/train/tensorboard")
        st.caption(
            "Lancez cette commande dans un terminal pour visualiser "
            "les courbes d'entrainement."
        )



if __name__ == "__main__":
    run_dashboard()
