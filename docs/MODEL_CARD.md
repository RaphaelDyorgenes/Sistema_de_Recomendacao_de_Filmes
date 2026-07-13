# Model Card — Recomendador NCF (`ncf-recommender`)

## Visão geral

Modelo de recomendação de filmes baseado em **Neural Collaborative
Filtering (NCF)**: embeddings de usuário e filme concatenados e passados
por um MLP que estima a probabilidade de o usuário gostar do título.
Treinado com PyTorch e versionado no MLflow Model Registry.

- **Versão em produção:** `ncf-recommender` v2 (estágio *Production*)
- **Run de origem:** `ncf-emb16` (menor loss de validação entre os
  experimentos rastreados)
- **Tarefa:** ranking top-K de itens por usuário (feedback implícito)

## Arquitetura e hiperparâmetros

| Hiperparâmetro | Valor |
|---|---|
| Dimensão dos embeddings | 16 |
| Camadas ocultas do MLP | 32 → 16 → 8 |
| Dropout | 0.2 |
| Otimizador / learning rate | Adam / 1e-3 |
| Batch size | 512 |
| Épocas máximas / early stopping | 20 / patience 5 |
| Loss | Binary Cross-Entropy |
| Semente | 42 (fixada em PyTorch e NumPy) |

Três configurações foram comparadas no MLflow (embeddings 16, 32 e 64);
a menor venceu por loss de validação (0.4227 vs 0.4256 e 0.4267),
indicando que capacidade extra só acelerou o overfitting neste volume de
dados.

## Dados

- **Fonte:** MovieLens-20M (20M de avaliações de filmes): avaliação
  ≥ 3.5 é tratada como interação positiva (o usuário gostou do filme).
- **Amostragem:** 99.905 interações de 1.174 usuários ativos (mínimo de
  5 interações por usuário, histórico completo preservado), 7.263 itens.
- **Split temporal:** 64% treino / 16% validação / 20% teste, ordenado
  por timestamp — o modelo nunca vê o futuro no treino.
- **Negativos:** 4 negativos amostrados por positivo, em treino e
  validação, excluindo itens que o usuário já consumiu.

## Performance

Métricas de ranking no conjunto de teste (K = 10), comparadas com dois
baselines em Scikit-Learn:

| Métrica | NCF | Popularidade | KNN de usuários |
|---|---|---|---|
| Precision@10 | **0.2852** | 0.2780 | 0.2796 |
| Recall@10 | 0.0443 | **0.0452** | 0.0451 |
| MAP@10 | 0.1682 | 0.1893 | **0.1911** |
| NDCG@10 | 0.2694 | 0.2927 | **0.2939** |

Leitura honesta dos números: o NCF tem a melhor precisão do top-10, mas
perde em MAP/NDCG — ou seja, acerta uma quantidade parecida de itens,
porém os ordena pior que os baselines. Com ~64k interações de treino, o
sinal colaborativo simples (popularidade e vizinhança) ainda é muito
competitivo; a vantagem esperada da rede neural tende a aparecer com
mais dados e mais sinais de contexto.

## Limitações conhecidas

1. **Cold start:** usuários ou itens fora do treino não têm embedding;
   as recomendações degradam para as dos baselines.
2. **Overfitting rápido:** a loss de validação piora já a partir da 2ª
   época; o early stopping devolve um modelo de pouquíssimas épocas.
   Regularização mais forte ou mais dados são os próximos passos óbvios.
3. **Ranking abaixo dos baselines (MAP/NDCG):** no volume atual de
   dados, o modelo neural ainda não supera KNN/popularidade em ordenação.
4. **Amostra pequena por design:** 100k interações (0.5% do dataset)
   para viabilizar treino local em CPU; as métricas não devem ser
   extrapoladas para o dataset completo.

## Vieses potenciais

- **Viés de popularidade:** o treino é dominado por itens muito
  consumidos; itens de cauda longa são sub-recomendados, o que pode
  reforçar o próprio viés (feedback loop).
- **Viés de seleção:** a amostra exige ≥ 5 interações por usuário, então
  o modelo representa usuários engajados — usuários casuais ficam fora
  da distribuição de treino.
- **Viés temporal:** o split por timestamp concentra o teste no período
  mais recente; mudanças de catálogo/comportamento afetam as métricas.
- **Auditoria demográfica impossível:** o dataset não tem atributos
  demográficos, então não é possível medir disparidade de qualidade de
  recomendação entre grupos (idade, gênero, região).

## Reprodutibilidade

```bash
dvc repro                                  # pipeline completo (dados → avaliação)
python -m recsys.training.experiments      # 3 runs no MLflow
python -m recsys.training.registry         # registra e promove o melhor
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Sementes fixadas (`RANDOM_SEED=42`), dados versionados com DVC e
parâmetros/métricas/artefatos de todos os runs rastreados no MLflow.
