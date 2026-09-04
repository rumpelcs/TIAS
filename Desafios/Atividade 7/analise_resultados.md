# Exercício de Fixação — Comparativo de Modelos de Predição

**Algoritmos comparados:** Regressão Logística, Árvore de Decisão, Random Forest, KNN (k=5) e SVM (kernel RBF)
**Metodologia:** divisão treino/teste 70/30 (estratificada), padronização (StandardScaler) para os modelos sensíveis a escala (Regressão Logística, KNN, SVM), `random_state=42` para reprodutibilidade.

---

## Problema 1 — Predição de Compra de Produto (`dados_predicao_modelos.csv`)

**Base:** 250 registros | Target `Compro_Produto`: 151 (não comprou) x 99 (comprou) — moderadamente desbalanceada.

| Modelo | Acurácia Treino | Acurácia Teste | Gap (Treino-Teste) | Precisão | Recall | F1-Score |
|---|---|---|---|---|---|---|
| Regressão Logística | 0.7086 | 0.6533 | 0.0552 | 0.6000 | 0.4000 | 0.4800 |
| Árvore de Decisão | 0.8629 | 0.6000 | **0.2629** | 0.5000 | 0.2333 | 0.3182 |
| Random Forest | **1.0000** | 0.6400 | **0.3600** | 0.5714 | 0.4000 | 0.4706 |
| KNN (k=5) | 0.7771 | 0.6533 | 0.1238 | 0.5833 | 0.4667 | **0.5185** |
| SVM (RBF) | 0.7771 | 0.6800 | 0.0971 | 0.6875 | 0.3667 | 0.4783 |

**Matrizes de Confusão (teste, 75 registros: 45 classe 0 / 30 classe 1)**

```
Regressão Logística     Árvore de Decisão        Random Forest
[[37  8]                [[38  7]                 [[36  9]
 [18 12]]                [23  7]]                 [18 12]]

KNN (k=5)                SVM (RBF)
[[35 10]                 [[40  5]
 [16 14]]                 [19 11]]
```
(Linhas = real, colunas = previsto; ordem [VN, FP] / [FN, VP])

### Análise de Overfitting (Problema 1)

Os dados **apresentam sinais claros de overfitting em modelos mais flexíveis**:

- **Random Forest**: acurácia de treino de **100%** contra apenas **64% no teste** — gap de 36 pontos percentuais. Isso é o sintoma clássico de overfitting: o modelo memorizou o ruído do conjunto de treino em vez de aprender o padrão real.
- **Árvore de Decisão**: gap de **26,3 p.p.** (86,3% treino vs 60% teste), também indicando overfitting, mesmo limitando a profundidade (`max_depth=5`).
- **Regressão Logística e SVM**: gaps pequenos (5,5 p.p. e 9,7 p.p.), sugerindo que esses modelos, por serem mais restritivos (fronteiras de decisão mais simples), generalizam melhor e **não overfitam** de forma significativa.
- **KNN**: gap intermediário (12,4 p.p.), esperado dado que KNN tende a se ajustar localmente aos dados de treino.

**Conclusão:** para este problema, os dados em si parecem conter um padrão real, porém fraco (nenhum modelo passa de 68% de acurácia no teste), somado a ruído. Modelos de alta capacidade (Random Forest, Árvore) capturam ruído específico do treino (overfitting), enquanto modelos mais simples (Regressão Logística, SVM) evitam esse problema, mas também não conseguem extrair muito mais sinal — o teto de desempenho é baixo para todos.

### Acurácia e F1-Score (Problema 1)

- **Acurácia** mede a proporção total de acertos (VP+VN)/Total. Aqui ela varia de 60% a 68% — pouco acima do que um classificador aleatório ponderado pela classe majoritária conseguiria (a classe 0 já representa 60% da base), o que é um sinal de alerta: a acurácia sozinha **superestima** a qualidade real do modelo em bases desbalanceadas.
- **F1-Score** é a média harmônica entre Precisão e Recall, e é mais informativo aqui porque penaliza modelos que erram muito a classe minoritária (quem compra). O melhor F1 foi do **KNN (0.5185)**, seguido do Random Forest (0.4706) — mas nenhum modelo atinge um F1 satisfatório (>0.7), evidenciando dificuldade real em prever corretamente quem vai comprar o produto.
- Em todos os modelos, o **Recall da classe 1** é baixo (23% a 47%), ou seja, os modelos erram (Falso Negativo) mais da metade dos clientes que realmente comprariam — um problema sério se o objetivo de negócio for identificar compradores potenciais.

---

## Problema 2 — Predição de Risco de Internação (`dados_saude_predicao.csv`)

**Base:** 250 registros | Target `Risco_Internacao`: 125 x 125 — **perfeitamente balanceada**.

| Modelo | Acurácia Treino | Acurácia Teste | Gap (Treino-Teste) | Precisão | Recall | F1-Score |
|---|---|---|---|---|---|---|
| Regressão Logística | 0.7657 | 0.7600 | **0.0057** | 0.7436 | 0.7838 | **0.7632** |
| Árvore de Decisão | 0.8914 | 0.7200 | 0.1714 | 0.7000 | 0.7568 | 0.7273 |
| Random Forest | **1.0000** | 0.7600 | **0.2400** | 0.7714 | 0.7297 | 0.7500 |
| KNN (k=5) | 0.8000 | 0.6933 | 0.1067 | 0.6667 | 0.7568 | 0.7089 |
| SVM (RBF) | 0.8171 | 0.7467 | 0.0705 | 0.7368 | 0.7568 | 0.7467 |

**Matrizes de Confusão (teste, 75 registros: 38 classe 0 / 37 classe 1)**

```
Regressão Logística     Árvore de Decisão        Random Forest
[[28 10]                [[26 12]                 [[30  8]
 [ 8 29]]                [ 9 28]]                 [10 27]]

KNN (k=5)                SVM (RBF)
[[24 14]                 [[28 10]
 [ 9 28]]                 [ 9 28]]
```

### Análise de Overfitting (Problema 2)

- **Random Forest** volta a apresentar overfitting evidente: **100% no treino contra 76% no teste** (gap de 24 p.p.), mesmo padrão do Problema 1 — floresta sem limitação de profundidade tende a memorizar.
- **Árvore de Decisão** também overfita (gap de 17,1 p.p.), embora um pouco menos que no Problema 1.
- **Regressão Logística** é o destaque: gap de apenas **0,57 p.p.** — praticamente idêntico desempenho em treino e teste, o que é o comportamento ideal, indicando que o modelo aprendeu um padrão real e generalizável, sem overfitting.
- **SVM** também generaliza bem (gap de 7 p.p.).
- **KNN** tem gap moderado (10,7 p.p.).

**Conclusão:** diferente do Problema 1, aqui há evidência de um **padrão real e mais forte** nos dados (acurácias de teste entre 69% e 76%, bem acima do baseline de 50% de uma base balanceada). O overfitting existe apenas nos modelos de alta capacidade (Árvore, Random Forest), mas os modelos lineares/de margem (Regressão Logística, SVM) confirmam que o sinal é genuíno, pois generalizam quase sem perda de desempenho.

### Acurácia e F1-Score (Problema 2)

- Como o dataset é **balanceado (125/125)**, a acurácia aqui é uma métrica confiável e comparável diretamente ao F1-Score.
- A **Regressão Logística** obteve o melhor equilíbrio: acurácia de 76% e o maior F1-Score (**0.7632**), com Precisão de 0.7436 e Recall de 0.7838 — ou seja, ela identifica corretamente ~78% dos pacientes de alto risco (poucos Falsos Negativos: apenas 8), o que é especialmente importante em contexto de saúde, onde deixar de identificar um paciente de risco (Falso Negativo) é mais grave que um Falso Positivo.
- **Random Forest** empata em acurácia (76%) mas tem F1 menor (0.75) e Recall menor (0.7297) — mesmo acurácia igual, o modelo erra mais pacientes de risco (10 Falsos Negativos), o que é pior clinicamente, apesar do overfitting mascarar isso no treino.
- O **SVM** fica muito próximo da Regressão Logística (F1 = 0.7467), reforçando a hipótese de padrão real nos dados.

---

## Apresentação ao Professor — Defesa dos Resultados

### Os dados de treinamento são válidos?

**Problema 1 (Compra de Produto): validade parcial/limitada.**
O treinamento *surtiu algum efeito* — todos os modelos superam ligeiramente um "chute" ingênuo pela classe majoritária (60,4%) — mas o efeito é fraco. O F1-Score baixo (máximo 0,52) e o Recall pobre da classe positiva mostram que as features disponíveis (Idade, Renda, Score de Crédito, Engajamento) explicam apenas parcialmente a decisão de compra, condizente com o enunciado, que descreve os dados como gerados com "ruído estatístico" propositalmente. **Nenhum modelo deste problema está pronto para produção** sem melhorias (mais features, engenharia de atributos, balanceamento de classes, ajuste de hiperparâmetros ou coleta de mais dados).

**Problema 2 (Risco de Internação): validade confirmada, com ressalvas por modelo.**
Aqui o treinamento claramente surtiu efeito real: modelos simples (Regressão Logística, SVM) mostram acurácia e F1 estáveis e muito próximos entre treino e teste, evidenciando que o padrão aprendido é genuíno e generalizável, não decorado. As variáveis clínicas (Pressão Arterial, Colesterol, Frequência Cardíaca Máxima, Idade) parecem ter relação real e mensurável com o risco de internação.

### Quais modelos podem ir para produção?

| Critério | Problema 1 | Problema 2 |
|---|---|---|
| Melhor modelo para produção | **KNN (k=5)** — melhor F1 (0.52), mas ainda fraco | **Regressão Logística** — melhor F1 (0.76), sem overfitting, interpretável |
| Modelos a evitar | Árvore de Decisão e Random Forest (overfitting severo, gap > 25 p.p.) | Random Forest (overfitting, gap 24 p.p., apesar da acurácia competitiva) |
| Recomendação geral | Não recomendado para produção sem melhorias na base/features | **Recomendado para produção**, com monitoramento contínuo e revalidação periódica |

**Justificativa técnica central da defesa:**
1. Um gap grande entre acurácia de treino e teste (como visto no Random Forest em ambos os problemas, e na Árvore de Decisão) é evidência direta de overfitting — o modelo decorou particularidades do treino que não se repetem em dados novos, tornando-o **não confiável em produção**, mesmo que sua acurácia de teste pareça aceitável isoladamente.
2. Acurácia isolada pode enganar, especialmente em bases desbalanceadas (Problema 1): um modelo pode acertar bastante só por favorecer a classe majoritária. O **F1-Score é a métrica mais confiável** para decidir qual modelo usar, pois só é alto quando Precisão e Recall estão ambos em nível razoável.
3. Modelos com **baixo gap treino-teste e F1 consistente** (Regressão Logística no Problema 2) são os que oferecem maior garantia de que o "treinamento surtiu efeito real" e que o desempenho observado se repetirá em dados futuros — critério decisivo para aprovação em produção.
