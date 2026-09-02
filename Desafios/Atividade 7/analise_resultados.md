# Análise de Resultados

## 1. Matriz de confusão — houve overfitting?

**Problema 1 (Compra de Produto):**
O Random Forest mostrou overfitting evidente: 96,6% de acurácia no treino contra apenas 61,3% no teste (gap de 35 pontos). A Árvore de Decisão também overfitou (86,3% vs. 60,0%). Esses dois modelos decoraram particularidades das 175 amostras de treino em vez de aprender um padrão real. Já Regressão Logística e Naive Bayes tiveram gaps pequenos (6-7 pontos) — sem overfitting, mas também sem muito o que aprender, porque o sinal nas variáveis é fraco.

**Problema 2 (Risco de Internação):**
Mesmo padrão no Random Forest (98,9% treino vs. 74,7% teste, gap de 24 pontos) — overfitting confirmado. Mas o Naive Bayes teve gap **negativo** (76,6% treino vs. 81,3% teste) e a Regressão Logística teve gap quase nulo (0,6 pontos). Essa é a evidência mais forte de que o treinamento funcionou de verdade: o padrão aprendido generaliza, não é decoreba.

**Conclusão:** os dados foram adequados para treinar modelos lineares/probabilísticos nos dois problemas (sem overfitting), mas modelos de árvore sem regularização overfitaram nos dois casos — o problema não está na base de dados, está na escolha/configuração do modelo.

## 2. Acurácia e F1-Score

**O que cada métrica significa:**
- **Acurácia** = proporção total de previsões corretas (acertos/total). É intuitiva, mas engana quando as classes são desbalanceadas — um modelo que sempre prevê a classe majoritária pode ter acurácia "alta" sem ser útil.
- **F1-Score** = média harmônica entre Precisão (dos que o modelo disse "sim", quantos eram realmente "sim") e Recall (dos que realmente eram "sim", quantos o modelo capturou). É mais rigoroso porque só fica alto se o modelo acerta nas duas frentes ao mesmo tempo.

**Problema 1:** acurácia máxima de 68% (SVM), F1 máximo de apenas 0,48. A diferença grande entre acurácia (razoável) e F1 (baixo) mostra que os modelos acertam bem a classe majoritária ("não comprou") mas erram muito a classe minoritária ("comprou") — recall de 0,23 a 0,40. Ou seja, a acurácia está mascarando um desempenho ruim.

**Problema 2:** acurácia de 81,3% e F1 de 0,80 no Naive Bayes — aqui as duas métricas concordam e são altas, com precisão (0,848) e recall (0,757) equilibrados. Isso indica um modelo genuinamente bom, não um efeito de desbalanceamento (as classes aqui são 50/50).
