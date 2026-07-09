# Detecção de Fraude em Cartão de Crédito

Projeto de machine learning ponta a ponta: da análise exploratória até uma API servindo o modelo em produção local. O objetivo foi enfrentar um problema clássico de **dados extremamente desbalanceados** apenas 0,17% das transações são fraude e documentar as decisões tomadas no caminho, incluindo o que **não** funcionou.

## O problema

O dataset ([Credit Card Fraud Detection — ULB/Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)) contém 284.807 transações de cartão de crédito, das quais somente 492 são fraudulentas. As features V1–V28 são componentes de PCA (anonimizadas por privacidade); apenas `Time` e `Amount` são originais.

Com esse desbalanceamento, **acurácia é uma métrica inútil**: um modelo que responde "não é fraude" para tudo acerta 99,83%. Por isso o projeto usa **PR-AUC** (área sob a curva precision-recall) como métrica oficial, além de analisar precision e recall separadamente.

## Metodologia

A comparação inicial de modelos foi feita em split 80/20 estratificado. Depois percebi (com ajuda de uma revisão externa) um erro sutil: eu tinha escolhido o threshold de decisão olhando as métricas do próprio conjunto de teste — o que contamina a avaliação, porque o teste deixa de ser "dado nunca visto" no momento em que orienta uma decisão.

A correção foi adotar **três conjuntos (60/20/20)**: o modelo treina no conjunto de treino, o threshold é escolhido no de **validação**, e o de **teste** é tocado uma única vez para o número final. Os resultados abaixo seguem esse protocolo.

## Resultados

Comparação de modelos (split 80/20, fase exploratória):

| Modelo | PR-AUC |
|---|---|
| Regressão logística (scaled + class_weight) | 0,719 |
| Regressão logística + SMOTE | 0,724 |
| Random Forest | 0,873 |
| **XGBoost (scale_pos_weight)** | **0,881** |

**Modelo final (XGBoost, protocolo 60/20/20, threshold 0,7 escolhido na validação):**

| Métrica (conjunto de teste) | Valor |
|---|---|
| PR-AUC | **0,864** |
| Precision (fraude) | 0,91 |
| Recall (fraude) | 0,82 |

Os números finais são um pouco menores que os da fase exploratória — e é assim que deve ser: são medidos sem vazamento de decisão, com menos dados de treino. Prefiro números menores e defensáveis a números bonitos e contaminados.

O que os resultados contam:

- **SMOTE quase não ajudou.** Criar 227 mil fraudes sintéticas rendeu praticamente o mesmo que o parâmetro `class_weight` de uma linha. O gargalo não era o balanceamento — era a fronteira linear da regressão logística. Testei, documentei e descartei pela complexidade extra.
- **Modelos de árvore quebraram o teto.** Random Forest e XGBoost capturam interações não-lineares entre as features (as mais importantes: V17, V14 e V12) que a logística não alcança.
- **Threshold é decisão de negócio, não de estatística.** A escolha veio de análise de custo na validação: entre capturar mais fraudes (recall) e bloquear menos clientes honestos (precision), o threshold 0,7 manteve o melhor equilíbrio para o cenário de banco — fraude perdida custa o valor da transação, alarme falso custa um SMS de confirmação.

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
├── notebooks/          # análises (exploração → baseline → melhorias → validação)
│   ├── 01-eda.ipynb
│   ├── 02-baseline.ipynb
│   ├── 03-melhorias.ipynb
│   └── 04-validacao.ipynb
├── src/api.py          # API FastAPI servindo o modelo
├── models/             # modelo XGBoost treinado (joblib)
└── data/               # dataset (não versionado — baixar do Kaggle)
```

## Melhorias futuras

- Remover a feature `Time` do modelo (artefato do dataset: segundos desde a primeira transação do CSV — nenhum sistema real fornece isso de forma coerente com o treino)
- Split temporal (treinar no passado, testar no futuro) — mais honesto para fraude, que tem dinâmica no tempo
- Testes automatizados da API (TestClient do FastAPI)
- Tuning de hiperparâmetros do XGBoost (GridSearch/Optuna)
- Validação cruzada estratificada
- Containerização com Docker
- Monitoramento de drift do modelo em produção
