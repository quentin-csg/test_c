# Monitoring — mn-bot

Stack : **Loki** (log storage) + **Promtail** (log shipper) + **Grafana** (dashboard + alertes).

## Prérequis

- Docker + Docker Compose v2
- Le bot configuré avec `LOG_FILE=logs/mn-bot.jsonl` dans `.env`

## Démarrage

```bash
cd monitoring
docker compose up -d
```

Grafana accessible sur **http://localhost:3000** (admin / admin).

Le dashboard "mn-bot" est auto-provisionné. Il affiche :
- Compteur RiskError (5 min glissantes)
- Compteur WS reconnects (1 h glissante)
- Compteur entrées / sorties de position
- Log des signaux de trading
- Log des événements warning/error

## Configuration bot

Dans `.env` :

```env
LOG_FILE=logs/mn-bot.jsonl
```

Le bot créé `logs/mn-bot.jsonl` au lancement. Promtail scrape ce fichier et envoie vers Loki.

## Alertes

Les règles sont dans `loki/rules.yml` :
- `HighRiskErrorRate` — 1 RiskError ou plus sur 5 min → warning
- `ExcessiveWsReconnects` — > 3 reconnexions WS sur 1 h → warning

Le routing des alertes (Slack, PagerDuty, email) nécessite un Alertmanager. Remplacer
`alertmanager_url` dans `loki-config.yml` par l'URL réelle quand disponible.

## Arrêt

```bash
docker compose down          # conserve les données
docker compose down -v       # supprime aussi les volumes (repart de zéro)
```
