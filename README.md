# Detecção de Fraude em Cartão de Crédito

Projeto de machine learning ponta a ponta: da análise exploratória até uma API servindo o modelo em produção local. O objetivo foi enfrentar um problema clássico de **dados extremamente desbalanceados** apenas 0,17% das transações são fraude e documentar as decisões tomadas no caminho, incluindo o que **não** funcionou.

## O problema

O dataset ([Credit Card Fraud Detection — ULB/Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)) contém 284.807 transações de cartão de crédito, das quais somente 492 são fraudulentas. As features V1–V28 são componentes de PCA (anonimizadas por privacidade); apenas `Time` e `Amount` são originais.

Com esse desbalanceamento, **acurácia é uma métrica inútil**: um modelo que responde "não é fraude" para tudo acerta 99,83%. Por isso o projeto usa **PR-AUC** (área sob a curva precision-recall) como métrica oficial, além de analisar precision e recall separadamente.

## Resultados

| Modelo | PR-AUC |
|---|---|
| Regressão logística (scaled + class_weight) | 0,719 |
| Regressão logística + SMOTE | 0,724 |
| Random Forest | 0,873 |
| **XGBoost (scale_pos_weight)** | **0,881** |

O que os números contam:

- **SMOTE quase não ajudou.** Criar 227 mil fraudes sintéticas rendeu praticamente o mesmo que o parâmetro `class_weight` de uma linha. O gargalo não era o balanceamento — era a fronteira linear da regressão logística. Testei, documentei e descartei pela complexidade extra.
- **Modelos de árvore quebraram o teto.** Random Forest e XGBoost capturam interações não-lineares entre as features (as mais importantes: V17, V14 e V12) que a logística não alcança.
- **Threshold é decisão de negócio, não de estatística.** No threshold de 0,9, o modelo final opera com **recall 0,84 e precision 0,92** — captura 84% das fraudes com pouquíssimos alarmes falsos. A escolha veio de análise de custo: 0,9 domina 0,5 (mesmo recall, mais precision) e perde menos fraudes que 0,99.

## Como rodar

```bash
# 1. clonar e preparar ambiente
git clone git@github.com:rafa-vargas-code/deteccao-fraude.git
cd deteccao-fraude
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. baixar o dataset do Kaggle (link acima) e salvar em data/creditcard.csv

# 3. subir a API (o modelo treinado já está em models/)
uvicorn src.api:app --reload
```

Interface de teste: http://127.0.0.1:8000/docs — endpoint `POST /predict` recebe o JSON da transação e responde:

```json
{
  "probabilidade": 0.9999908208847046,
  "fraude": true
}
```

## Estrutura

```
├── notebooks/          # análises (exploração → baseline → melhorias)
│   ├── 01-eda.ipynb
│   ├── 02-baseline.ipynb
│   └── 03-melhorias.ipynb
├── src/api.py          # API FastAPI servindo o modelo
├── models/             # modelo XGBoost treinado (joblib)
└── data/               # dataset (não versionado — baixar do Kaggle)
```

## Melhorias futuras

- Tuning de hiperparâmetros do XGBoost (GridSearch/Optuna)
- Validação cruzada estratificada em vez de holdout único
- Containerização com Docker
- Monitoramento de drift do modelo em produção
