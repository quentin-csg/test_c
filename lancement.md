0. Installation

pip install -r requirements.txt
1. Entraînement

python main.py train --model first_try
Entraîne PPO + CNN 1D sur BTC/USDT 1H (2020-2023), 23 features, 1M steps. Sauvegarde models/first_try.zip.

Surveiller en parallèle :


tensorboard --logdir logs/train/tensorboard/
2. Backtest rapide

python main.py backtest --model first_try
Test rapide sur 2024+. Donne une première idée, mais ne suffit pas — un seul backtest peut être chanceux.

3. Walk-forward validation ← la plus importante

python main.py walk-forward
Ré-entraîne et backteste sur plusieurs fenêtres temporelles successives :

Fold 1 : train 2020→2022 | test 3 mois (+ purge gap 200h)
Fold 2 : train 2020→mi-2022 | test 3 mois suivants
... jusqu'à aujourd'hui
C'est ici qu'on sait vraiment si le modèle est robuste ou overfitté. Résultats dans logs/walk_forward/. Si le Sharpe moyen est positif et stable sur tous les folds → bon signe.
Résultat : 10 Sharpe ratios sur des périodes différentes
Si la moyenne est > 0.5 et stable → la méthode est vraiment valide

4. Paper trading

python main.py live --model first_try
Tick toutes les heures en simulation. Arrêt : Ctrl+C.

5. Dashboard

python main.py dashboard


Backtest = quelques minutes. Il charge ton modèle déjà entraîné et le teste sur 2024+. C'est un filtre rapide.
Walk-forward = plusieurs heures. Il ré-entraîne N modèles (un par fold × 1M steps chacun).
Si le backtest montre un Sharpe négatif et un return de -30%, inutile de lancer un walk-forward de 6h pour confirmer que le modèle est mauvais. Tu ajustes d'abord, puis tu valides avec le walk-forward.


Métriques critiques (dans l'ordre de priorité)
Métrique	Bon	Médiocre	Mauvais — ne pas continuer
Total return %	> 0%	-10% à 0%	< -10%
Sharpe ratio	> 1.0	0 à 1.0	< 0
Max drawdown	< 15%	15-25%	> 25%
Sortino ratio	> 1.5	0 à 1.5	< 0
Nombre de trades	50-300	10-50 ou 300-500	0 (inactif) ou >500 (overtrading)
Comment interpréter
Return positif mais Sharpe < 0.5 → le bot a eu de la chance, les gains sont dus à la volatilité et pas à une vraie stratégie.

Sharpe > 1 mais drawdown > 25% → le bot gagne en moyenne mais prend des risques énormes, il peut tout perdre en un crash.

0 trades → le modèle reste dans la dead zone [-0.05, +0.05] et ne fait rien. L'entraînement n'a pas convergé.

> 500 trades sur quelques mois → overtrading, les frais (0.1% par trade) mangent tous les gains.

La comparaison clé : bot vs buy & hold
Compare le return du bot au simple fait d'acheter du BTC en janvier 2024 et de le garder. Si le bot fait +5% et le buy & hold fait +60%, le bot est inutile — il sous-performe une stratégie qui ne demande aucun effort.

Signal pour continuer au walk-forward
Feu vert : Sharpe > 0.5, drawdown < 20%, nombre de trades raisonnable → lance le walk-forward.

Feu rouge : Sharpe négatif, return < -10%, ou 0 trades → ajuste les hyperparamètres ou le reward avant de perdre du temps sur le walk-forward.