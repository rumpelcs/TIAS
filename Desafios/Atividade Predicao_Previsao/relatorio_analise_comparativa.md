# Análise dos Modelos de Predição

## 1. Análise da Matriz de Confusão e Overfitting

A matriz de confusão permite verificar quantas previsões o modelo acertou e quantos erros foram cometidos, principalmente entre **falsos positivos** e **falsos negativos**.

### Problema 1 — Predição de Compra

No problema de previsão de compra, os modelos apresentaram muitos **falsos negativos**, ou seja, vários clientes que realmente compraram foram classificados como pessoas que não comprariam.

Além disso, alguns modelos apresentaram sinais claros de **overfitting**:

- **Random Forest:** 96,6% de acurácia no treino e 61,3% no teste.
- **Árvore de Decisão:** 86,3% no treino e 60,0% no teste.

Essa grande diferença entre treino e teste mostra que esses modelos aprenderam muito bem os dados de treinamento, mas tiveram dificuldade para generalizar para novos dados.

Já a **Regressão Logística** e o **Naive Bayes** apresentaram diferenças menores entre treino e teste, indicando um comportamento mais estável e sem overfitting significativo.

**Conclusão:** os dados são adequados para análise e aprendizado, mas as variáveis utilizadas possuem pouco poder para prever a compra. O problema principal não é somente overfitting, mas também a existência de pouco sinal preditivo nos dados.

---

### Problema 2 — Predição de Risco de Internação

No problema de risco de internação, a matriz de confusão apresentou resultados mais equilibrados. O modelo conseguiu identificar melhor tanto os casos de baixo risco quanto os de alto risco.

O **Naive Bayes** apresentou o melhor resultado, com:

- 33 acertos em 38 casos de baixo risco;
- 28 acertos em 37 casos de alto risco;
- 5 falsos positivos;
- 9 falsos negativos.

O **Random Forest** novamente apresentou overfitting, com 98,9% de acurácia no treino contra 74,7% no teste.

A **Regressão Logística** apresentou um comportamento mais estável, com 76,6% no treino e 76,0% no teste.

**Conclusão:** neste problema, os dados apresentaram um padrão mais consistente. O Naive Bayes conseguiu generalizar melhor e apresentou os melhores resultados.

---

## 2. Acurácia e F1-Score

### Acurácia

A **acurácia** representa a proporção de previsões que o modelo acertou em relação ao total de previsões.

Por exemplo, uma acurácia de 80% significa que o modelo acertou aproximadamente 80% das classificações.

No entanto, a acurácia sozinha não é suficiente para avaliar um modelo, principalmente quando as classes estão desbalanceadas.

### F1-Score

O **F1-Score** combina duas métricas:

- **Precisão:** entre os casos classificados como positivos, quantos realmente eram positivos.
- **Recall:** entre os casos que realmente eram positivos, quantos foram identificados pelo modelo.

O F1-Score é útil porque considera tanto a precisão quanto o recall. Seu valor varia de 0 a 1, sendo que valores mais próximos de 1 indicam melhor equilíbrio.

### Resultados

#### Problema 1 — Compra

- Melhor acurácia no teste: **SVM — 68,0%**
- Melhor F1-Score: **Regressão Logística e Naive Bayes — 0,48**

Esses resultados são relativamente baixos. A acurácia ficou próxima da proporção da classe majoritária, e o F1-Score baixo mostra dificuldade em identificar corretamente os clientes que realmente compraram.

#### Problema 2 — Risco de Internação

- Melhor acurácia no teste: **Naive Bayes — 81,3%**
- Melhor F1-Score: **Naive Bayes — 0,80**

Nesse problema, os resultados foram significativamente melhores. O F1-Score de 0,80 indica que o modelo conseguiu equilibrar bem precisão e recall.

**Conclusão:** o Problema 2 apresentou dados mais adequados para a classificação. O Naive Bayes foi o modelo que apresentou o melhor equilíbrio entre acurácia, precisão, recall e F1-Score.
