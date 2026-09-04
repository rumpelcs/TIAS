# Análise dos Modelos — Problemas 1 e 2

## 1. Análise para verificar overfitting

> **Observação:** as imagens apresentadas não são matrizes de confusão, mas tabelas com métricas dos modelos. Para analisar **overfitting**, podemos utilizar principalmente a diferença entre **Acurácia de Treino** e **Acurácia de Teste**, chamada de `Gap_Treino_Teste`.

### Problema 1

No primeiro problema, podemos observar:

| Modelo | Acurácia Treino | Acurácia Teste | Gap |
|---|---:|---:|---:|
| Regressão Logística | 0.709 | 0.653 | 0.055 |
| Naive Bayes | 0.720 | 0.653 | 0.067 |
| SVM | 0.777 | 0.680 | 0.097 |
| KNN | 0.766 | 0.600 | 0.166 |
| Random Forest | 0.966 | 0.613 | 0.352 |
| Árvore de Decisão | 0.863 | 0.600 | 0.263 |

**Conclusão:** há indícios de **overfitting principalmente no Random Forest e na Árvore de Decisão**. 

O Random Forest, por exemplo, possui **96,6% de acurácia no treinamento**, mas apenas **61,3% no teste**, gerando um gap de **0,352**. Isso significa que o modelo aprendeu muito bem os dados de treinamento, mas teve dificuldade para generalizar para dados novos.

A **Regressão Logística e o Naive Bayes** apresentam gaps bem menores, indicando um comportamento mais equilibrado entre treino e teste.

---

### Problema 2

| Modelo | Acurácia Treino | Acurácia Teste | Gap |
|---|---:|---:|---:|
| Naive Bayes | 0.766 | 0.813 | -0.048 |
| Regressão Logística | 0.766 | 0.760 | 0.006 |
| SVM | 0.817 | 0.747 | 0.070 |
| Random Forest | 0.989 | 0.747 | 0.242 |
| Árvore de Decisão | 0.891 | 0.720 | 0.171 |
| KNN | 0.800 | 0.693 | 0.107 |

**Conclusão:** novamente, o **Random Forest apresenta o maior indício de overfitting**, pois possui **98,9% de acurácia no treinamento**, mas apenas **74,7% no teste**.

A **Regressão Logística apresenta o melhor equilíbrio**, com 76,6% no treinamento e 76,0% no teste, resultando em um gap de apenas **0,006**.

O Naive Bayes possui acurácia de teste maior que a de treino, portanto não apresenta indício de overfitting nesse caso.

### O que é overfitting?

**Overfitting** acontece quando o modelo "decora" muito os dados de treinamento e não consegue generalizar bem para dados novos.

Uma forma simples de identificar:

> **Acurácia de treino muito maior que a acurácia de teste → possível overfitting.**

---

# 2. Análise das métricas Acurácia e F1-Score

## Acurácia

A **Acurácia** representa a proporção de previsões que o modelo acertou em relação ao total de previsões.

A fórmula é:

**Acurácia = (Previsões corretas) / (Total de previsões)**

Por exemplo, uma acurácia de **0,80 (80%)** significa que o modelo acertou aproximadamente 80% das classificações.

### Problema 1

A maior acurácia de teste foi:

- **SVM: 0,680 (68%)**

Porém, isso não significa necessariamente que ele seja o melhor modelo em todas as situações. É importante observar também outras métricas, como o F1-Score.

### Problema 2

A maior acurácia de teste foi:

- **Naive Bayes: 0,813 (81,3%)**

Portanto, considerando apenas a acurácia, o **Naive Bayes foi o melhor modelo no Problema 2**.

---

## F1-Score

O **F1-Score** combina duas métricas:

- **Precisão (Precision):** entre as previsões positivas feitas pelo modelo, quantas estavam corretas.
- **Recall:** entre os casos que realmente eram positivos, quantos o modelo conseguiu encontrar.

O F1-Score é calculado pela média harmônica entre Precisão e Recall:

**F1 = 2 × (Precisão × Recall) / (Precisão + Recall)**

O valor varia de **0 a 1**:

- **1,0:** modelo excelente;
- **0,5:** desempenho intermediário;
- **0:** modelo muito ruim.

O F1-Score é especialmente útil quando queremos um equilíbrio entre **precisão e capacidade de encontrar os casos positivos**.

### Problema 1

Os melhores F1-Scores foram:

- Regressão Logística: **0,480**
- Naive Bayes: **0,480**
- SVM: **0,478**

Portanto, esses três modelos apresentam desempenho semelhante considerando o equilíbrio entre precisão e recall.

### Problema 2

O melhor F1-Score foi:

- **Naive Bayes: 0,800**

Além de possuir a maior acurácia (**81,3%**), o Naive Bayes também apresentou o maior F1-Score. Portanto, **é o modelo que apresentou o melhor desempenho geral no Problema 2 entre os modelos avaliados**.

---

## Resumo

| Problema | Melhor modelo pela Acurácia de Teste | Melhor F1-Score | Overfitting mais evidente |
|---|---|---|---|
| **1** | SVM — 68,0% | Regressão Logística / Naive Bayes — 0,480 | Random Forest |
| **2** | Naive Bayes — 81,3% | Naive Bayes — 0,800 | Random Forest |

### Em poucas palavras

- **Acurácia:** mostra **quanto o modelo acertou no total**.
- **Precisão:** mostra **quanto das previsões positivas estavam corretas**.
- **Recall:** mostra **quantos dos casos positivos reais o modelo encontrou**.
- **F1-Score:** procura um **equilíbrio entre precisão e recall**.
- **Overfitting:** acontece quando o modelo vai muito bem no treinamento, mas seu desempenho cai bastante nos dados de teste.
