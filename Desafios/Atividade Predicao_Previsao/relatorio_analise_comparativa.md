# Exercício de Fixação — Comparativo de Algoritmos de Predição (Classificação Binária)

**Fonte dos dados:** repositório `alexandrezamberlan/tias`, pasta `3_predicao_previsao_codigos_exemplos`

---

## 1. Metodologia aplicada aos dois problemas

Para permitir comparação justa, os dois conjuntos de dados (250 linhas cada) foram tratados com o mesmo pipeline:

1. **Divisão treino/teste:** *holdout* estratificado 70%/30% (`random_state=42`), preservando a proporção da classe alvo nos dois subconjuntos.
2. **Padronização:** `StandardScaler` aplicado às features usadas por modelos sensíveis à escala (Regressão Logística, KNN, SVM).
3. **Modelos comparados:** Regressão Logística, Árvore de Decisão, Random Forest, KNN, SVM (kernel RBF) e Naive Bayes Gaussiano.
4. **Métricas coletadas por modelo:**
   - Acurácia em treino **e** em teste (para detectar overfitting via o *gap* entre as duas).
   - Acurácia média em validação cruzada de 5 *folds* (estimativa mais robusta, menos dependente de uma única divisão).
   - Precisão, Recall e F1-Score na base de teste.
   - Matriz de confusão na base de teste.

---

## 2. Problema 1 — Predição de Compra de Produto (`Compro_Produto`)

**Features:** Idade, Renda_Anual_K, Score_Credito, Pontuacao_Engajamento
**Classes:** 60,4% não compraram (0) / 39,6% compraram (1) — desbalanceamento leve, tratável com estratificação.

| Modelo | Acc. Treino | Acc. Teste | Gap Treino-Teste | Acc. CV5 (média) | Precisão | Recall | F1-Score |
|---|---|---|---|---|---|---|---|
| SVM | 0,777 | **0,680** | 0,097 | 0,674 | 0,688 | 0,367 | 0,478 |
| Regressão Logística | 0,709 | 0,653 | 0,055 | 0,709 | 0,600 | 0,400 | 0,480 |
| Naive Bayes | 0,720 | 0,653 | 0,067 | 0,709 | 0,600 | 0,400 | 0,480 |
| KNN | 0,766 | 0,600 | 0,166 | 0,680 | 0,500 | 0,400 | 0,444 |
| Random Forest | **0,966** | 0,613 | **0,352** | 0,674 | 0,526 | 0,333 | 0,408 |
| Árvore de Decisão | 0,863 | 0,600 | 0,263 | 0,669 | 0,500 | 0,233 | 0,318 |

### Leitura das matrizes de confusão (Problema 1)
Em todos os modelos, o padrão dominante é o mesmo: muitos **falsos negativos** (clientes que compraram, mas o modelo previu que não comprariam) — por exemplo, a Regressão Logística acerta 12 de 30 compradores reais e erra 18. Isso mostra que a classe "Comprou" é a mais difícil de capturar com essas quatro variáveis; o sinal disponível não separa bem as duas classes.

### Diagnóstico de overfitting
O ponto crítico é o **Random Forest**: acurácia de **96,6% em treino** contra apenas **61,3% em teste** — um *gap* de 35 pontos percentuais, o maior de toda a tabela. A Árvore de Decisão isolada também sofre do mesmo problema (86,3% treino vs. 60,0% teste). Esse padrão é a assinatura clássica de overfitting: modelos baseados em árvore, com pouca regularização, memorizam ruído específico das 175 observações de treino e não generalizam.
Já Regressão Logística e Naive Bayes têm gaps pequenos (~6-7 p.p.) e a validação cruzada confirma a estabilidade (desvio-padrão baixo, ~0,03-0,06). Isso indica que **esses modelos não estão overfitados** — o problema deles é outro: a **base tem sinal fraco** (dataset sintético com ruído estatístico proposital, como o enunciado descreve), então mesmo sem overfitting a acurácia máxima obtida fica na faixa de 65-68%.

### Acurácia e F1-Score — o que dizem no Problema 1
- **Acurácia** (proporção total de acertos) fica em torno de 60-68% para a maioria dos modelos — pouco acima de um "chute" informado pela classe majoritária (60,4%), o que já é um sinal de alerta sobre a capacidade preditiva real.
- **F1-Score** (média harmônica entre precisão e recall, mais sensível ao desbalanceamento) é baixo em todos os modelos (0,32 a 0,48), sempre puxado para baixo pelo **recall** da classe "Comprou" (0,23 a 0,40). Isso confirma, de forma numérica, o que a matriz de confusão já mostrava visualmente: os modelos erram sistematicamente ao tentar identificar quem vai comprar.

---

## 3. Problema 2 — Predição de Risco de Internação (`Risco_Internacao`)

**Features:** Idade, Pressao_Arterial, Colesterol_Total, Frequencia_Cardiaca_Max
**Classes:** 50% / 50% — perfeitamente balanceado.

| Modelo | Acc. Treino | Acc. Teste | Gap Treino-Teste | Acc. CV5 (média) | Precisão | Recall | F1-Score |
|---|---|---|---|---|---|---|---|
| **Naive Bayes** | 0,766 | **0,813** | **-0,048** | 0,726 | 0,848 | 0,757 | **0,800** |
| Regressão Logística | 0,766 | 0,760 | 0,006 | 0,737 | 0,744 | 0,784 | 0,763 |
| SVM | 0,817 | 0,747 | 0,070 | 0,703 | 0,737 | 0,757 | 0,747 |
| Random Forest | **0,989** | 0,747 | **0,242** | 0,703 | 0,737 | 0,757 | 0,747 |
| Árvore de Decisão | 0,891 | 0,720 | 0,171 | 0,623 | 0,700 | 0,757 | 0,727 |
| KNN | 0,800 | 0,693 | 0,107 | 0,720 | 0,667 | 0,757 | 0,709 |

### Leitura das matrizes de confusão (Problema 2)
Diferente do Problema 1, aqui os erros ficam distribuídos de forma equilibrada entre falsos positivos e falsos negativos, e o Naive Bayes se destaca: 33 acertos em 38 casos de baixo risco e 28 em 37 de alto risco, com apenas 5 falsos positivos e 9 falsos negativos. As quatro variáveis clínicas (idade, pressão, colesterol, frequência cardíaca) discriminam as classes de forma bem mais consistente do que as variáveis comportamentais do Problema 1.

### Diagnóstico de overfitting
Novamente o **Random Forest** exibe o maior *gap* (98,9% treino vs. 74,7% teste = 24,2 p.p.), reforçando que ensembles de árvore sem *tuning* de profundidade/regularização tendem a decorar a base de treino quando há poucas amostras (175 linhas de treino para 4 variáveis). A Árvore de Decisão isolada tem o mesmo problema em menor escala (17,1 p.p.).
Por outro lado, o **Naive Bayes tem gap negativo** (-4,8 p.p., ou seja, performou até melhor no teste que no treino) e a Regressão Logística tem gap praticamente nulo (0,6 p.p.) — ambos sinais de que **não há overfitting** nesses dois modelos, e que o padrão aprendido é genuíno e generaliza.

### Acurácia e F1-Score — o que dizem no Problema 2
- **Acurácia** do melhor modelo (Naive Bayes) chega a 81,3% no teste, bem acima do "chute" da classe majoritária (50%), evidenciando que o modelo de fato capturou relação real entre as variáveis clínicas e o risco de internação.
- **F1-Score** de 0,80 no Naive Bayes mostra equilíbrio entre precisão (0,848) e recall (0,757): o modelo não só acerta quando prevê "alto risco", como também consegue identificar a maior parte dos pacientes que realmente estão em risco — isso é especialmente relevante em contexto de saúde, onde deixar de identificar um paciente de alto risco (falso negativo) tem custo maior que um alarme falso.

---

## 4. Comparativo entre os dois problemas

| Aspecto | Problema 1 (Compra) | Problema 2 (Saúde) |
|---|---|---|
| Melhor F1-Score | 0,480 (Regressão Logística/Naive Bayes) | **0,800 (Naive Bayes)** |
| Melhor Acurácia teste | 0,680 (SVM) | **0,813 (Naive Bayes)** |
| Maior gap treino-teste | 0,352 (Random Forest) | 0,242 (Random Forest) |
| Sinal das features em relação ao alvo | Fraco/ruidoso | Forte e consistente |
| Overfitting presente? | Sim, em modelos de árvore (Árvore de Decisão e Random Forest) | Sim, em modelos de árvore (mesmo padrão) |
| Modelos sem overfitting relevante | Regressão Logística, Naive Bayes | Regressão Logística, Naive Bayes, SVM |

---

## 5. Defesa para o professor: os dados de treinamento são válidos?

**Sim, com ressalvas — e a resposta depende de olhar para o modelo certo, não para "o treinamento" de forma genérica.**

1. **O treinamento surtiu efeito real**, mas de forma desigual entre os modelos. A prova está na comparação entre acurácia de treino e de teste (o *gap*) combinada com a validação cruzada:
   - Em **Regressão Logística e Naive Bayes**, o desempenho em treino, teste e validação cruzada é consistente (gaps de poucos pontos percentuais, ou até negativos). Isso é a evidência estatística de que esses modelos **aprenderam um padrão genuíno**, generalizável, e não decoraram a base.
   - Em **Árvore de Decisão e Random Forest**, o gap de 24 a 35 pontos percentuais entre treino e teste é a assinatura clássica de **overfitting**: o modelo memorizou ruído específico das 175 amostras de treino em vez de aprender a relação real entre as variáveis. Isso não invalida a base de dados — invalida o uso *daqueles modelos sem regularização* nessa base pequena.

2. **A base de dados em si está adequada ao propósito didático**: por ser sintética e conter "ruído estatístico" proposital (como descrito no enunciado), ela não deveria (e não permite) que nenhum modelo atinja 100% de acurácia — e de fato nenhum atingiu. Isso é desejável pedagogicamente: obriga o aluno a comparar modelos e reconhecer overfitting em vez de aceitar acurácia perfeita sem questionar.

3. **Quanto ao uso em produção:**
   - **Problema 2 (Risco de Internação) — Naive Bayes ou Regressão Logística poderiam ser considerados protótipos viáveis para produção**, desde que validados com uma base real (não sintética) e maior volume de dados, e submetidos a validação clínica formal. F1-Score de 0,80 e acurácia de 81% com gap treino-teste próximo de zero são indicadores de generalização real — mas 250 linhas é uma amostra pequena para decisões clínicas, então o recomendável é tratar isso como **prova de conceito**, não como modelo final.
   - **Problema 1 (Compra de Produto) — nenhum modelo deveria ir para produção neste estado.** Mesmo os modelos sem overfitting (Regressão Logística, Naive Bayes) atingem apenas ~65% de acurácia e F1 de 0,48, pouco acima do baseline da classe majoritária. Isso indica que **as quatro variáveis disponíveis não têm poder preditivo suficiente** para a tarefa — o problema não é overfitting, é **underfitting/sinal insuficiente**. A recomendação técnica seria buscar mais variáveis (histórico de compras, canal de aquisição, sazonalidade) antes de considerar produção.
   - **Random Forest e Árvore de Decisão, nos dois problemas, devem ser descartados para produção da forma como foram treinados** (sem tuning de profundidade, poda ou regularização), pois o gap treino-teste comprova que sua acurácia de treino é enganosa.

**Conclusão da defesa:** o treinamento foi válido enquanto *processo* (a metodologia — holdout estratificado, padronização, validação cruzada, comparação de múltiplos algoritmos — está correta e permitiu diagnosticar overfitting onde ele existe). A validade de *cada modelo para produção* é diferente: modelos lineares/probabilísticos generalizam bem nos dois problemas, mas só no Problema 2 o sinal dos dados é forte o suficiente para justificar acurácia e F1 realmente úteis em um cenário aplicado.
