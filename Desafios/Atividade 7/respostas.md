# Análise de Treinamento e Generalização dos Modelos Preditivos

## Metodologia

Para os dois problemas (Predição de Compra e Predição de Risco de Internação), foram treinados três modelos de classificação — **Regressão Logística**, **Árvore de Decisão** e **Random Forest** — utilizando divisão treino/teste de 70%/30% (175 exemplos de treino e 75 de teste), estratificada pela variável alvo, sobre os datasets de 250 linhas fornecidos.

---

## Resultados — Problema 1 (Compro_Produto)

| Modelo | Acc treino | Acc teste | Gap Acc | F1 treino | F1 teste | Gap F1 |
|---|---|---|---|---|---|---|
| Regressão Logística | 0,709 | 0,653 | +0,055 | 0,585 | 0,480 | +0,105 |
| Árvore de Decisão | 1,000 | 0,640 | +0,360 | 1,000 | 0,526 | +0,474 |
| Random Forest | 1,000 | 0,640 | +0,360 | 1,000 | 0,471 | +0,529 |

Matrizes de confusão (teste) — formato `[[VN, FP], [FN, VP]]`:

- Regressão Logística: `[[37, 8], [18, 12]]`
- Árvore de Decisão: `[[33, 12], [15, 15]]`
- Random Forest: `[[36, 9], [18, 12]]`

---

## Resultados — Problema 2 (Risco_Internacao)

| Modelo | Acc treino | Acc teste | Gap Acc | F1 treino | F1 teste | Gap F1 |
|---|---|---|---|---|---|---|
| Regressão Logística | 0,766 | 0,760 | +0,006 | 0,766 | 0,763 | +0,003 |
| Árvore de Decisão | 1,000 | 0,760 | +0,240 | 1,000 | 0,769 | +0,231 |
| Random Forest | 1,000 | 0,760 | +0,240 | 1,000 | 0,750 | +0,250 |

Matrizes de confusão (teste):

- Regressão Logística: `[[28, 10], [8, 29]]`
- Árvore de Decisão: `[[27, 11], [7, 30]]`
- Random Forest: `[[30, 8], [10, 27]]`

---

## Defesa: os modelos foram adequadamente treinados? Há capacidade de generalização?

A resposta é diferente para cada tipo de modelo, e é exatamente essa diferença que sustenta a defesa.

### 1. Árvore de Decisão e Random Forest: **não generalizam bem — overfitting claro**

Nos dois problemas, esses modelos atingem accuracy e F1 de **1,000 no treino** (acertam 100%, matriz de confusão perfeita, zero erros). Isso não é sinal de bom modelo — é sinal de **memorização**: como a árvore não foi podada (sem limite de profundidade) e o dataset é pequeno (250 linhas), o modelo simplesmente decorou cada exemplo de treino.

A prova está no *gap*: a accuracy cai de 1,00 para ~0,64–0,76 no teste, um gap de 24 a 53 pontos percentuais no F1. Isso é overfitting clássico, evidenciado não pela matriz de confusão do teste isolada, mas **pela comparação entre treino e teste** — exatamente o critério que o enunciado pede para não ignorar.

### 2. Regressão Logística: **generaliza de forma consistente, ainda que com desempenho mais modesto**

No Problema 1, o gap é pequeno (+0,055 em accuracy, +0,105 em F1) — o modelo tem desempenho parecido em treino e teste, mostrando que aprendeu um padrão real, não decorou os dados.

No Problema 2, o gap é praticamente zero (+0,006 em accuracy, +0,003 em F1) — desempenho quase idêntico entre treino e teste, o melhor indício possível de generalização entre os modelos avaliados.

### 3. Por que isso acontece

A Regressão Logística é um modelo linear, com poucos parâmetros e restrições implícitas de forma — ela não consegue "decorar" pontos individuais do dataset. Já a Árvore de Decisão e o Random Forest, sem limite de profundidade, têm capacidade praticamente ilimitada de se ajustar a cada ponto do treino. Com apenas 175 exemplos de treino, essa flexibilidade excessiva é fatal para a generalização.

---

## O treinamento "surtiu efeito"?

Sim, mas **somente nos modelos lineares**. Nos modelos baseados em árvore, o "treinamento" na prática produziu memorização, não aprendizado de padrão — o que equivale a dizer que, para esses modelos, o treinamento **não surtiu o efeito desejado** (generalizar para dados novos), apesar de a accuracy de treino perfeita passar a falsa impressão de sucesso.

---

## Algum modelo pode ser utilizado em produção?

- **Problema 2 (saúde) + Regressão Logística** é o caso mais defensável: accuracy e F1 de aproximadamente 0,76 no teste, gap virtualmente nulo em relação ao treino, e recall de 0,784 — métrica relevante nesse contexto, já que minimizar falsos negativos (pacientes de risco não identificados) é prioridade clínica. Ainda assim, 0,76 de accuracy não representa um patamar "pronto para produção" em um contexto de saúde; o mais correto seria classificar este modelo como pronto para uma **fase de piloto validado com especialistas**, e não para decisão autônoma.

- **Problema 1 + Regressão Logística**: accuracy de teste de apenas 0,653 é fraca — mesmo generalizando bem (baixo overfitting), o modelo simplesmente não é muito preciso. Precisaria de mais features ou engenharia de atributos antes de ir para produção.

- **Árvore de Decisão e Random Forest, nos dois problemas**: **não devem ir para produção como estão**, pois o overfitting demonstrado indica que o desempenho em dados novos é não confiável e pode variar de forma imprevisível. Seria necessário aplicar poda (limitar `max_depth`), validação cruzada e ampliar a base de dados antes de qualquer consideração para uso real.

---

## Conclusão

Dos seis modelos testados, apenas a **Regressão Logística** demonstrou capacidade real de generalização nos dois problemas — sustentado pelo gap mínimo entre treino e teste, e não pela accuracy isolada. Isso reforça exatamente o alerta do enunciado: **overfitting se prova pela comparação treino/teste, não pela matriz de confusão isolada do teste**. É justamente essa comparação que desmascara a Árvore de Decisão e o Random Forest como modelos não generalizáveis neste cenário, apesar de suas métricas de teste parecerem "razoáveis" à primeira vista quando observadas isoladamente.
