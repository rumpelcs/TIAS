"""
Exercício de Fixação - Comparativo de Algoritmos de Predição (Classificação)
Problema 1: Predição de Compra de Produto (Compro_Produto)

Fonte de dados:
https://github.com/alexandrezamberlan/tias/blob/main/3_predicao_previsao_codigos_exemplos/dados_predicao_modelos.csv

X (features): Idade, Renda_Anual_K, Score_Credito, Pontuacao_Engajamento
y (target):   Compro_Produto (0 = não comprou, 1 = comprou)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score,
    precision_score, recall_score, classification_report
)

# ---------------------------------------------------------------
# 1. Carga dos dados
# ---------------------------------------------------------------
df = pd.read_csv("dados_predicao_modelos.csv")
print("Dimensão do dataset:", df.shape)
print(df.head())
print("\nBalanceamento da classe alvo:")
print(df["Compro_Produto"].value_counts(normalize=True))

X = df[["Idade", "Renda_Anual_K", "Score_Credito", "Pontuacao_Engajamento"]]
y = df["Compro_Produto"]

# ---------------------------------------------------------------
# 2. Divisão treino/teste (holdout estratificado)
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Padronização (necessária para KNN, SVM e Regressão Logística)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 3. Modelos comparativos
# ---------------------------------------------------------------
modelos = {
    "Regressao_Logistica": LogisticRegression(max_iter=1000, random_state=42),
    "Arvore_Decisao": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random_Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "Naive_Bayes": GaussianNB(),
}

resultados = []
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, (nome, modelo) in enumerate(modelos.items()):
    # Modelos sensíveis à escala usam dados padronizados
    usa_escala = nome in ["Regressao_Logistica", "KNN", "SVM"]
    Xtr = X_train_scaled if usa_escala else X_train
    Xte = X_test_scaled if usa_escala else X_test

    modelo.fit(Xtr, y_train)

    # Predição em treino (para checar overfitting) e teste
    y_pred_train = modelo.predict(Xtr)
    y_pred_test = modelo.predict(Xte)

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test)
    prec_test = precision_score(y_test, y_pred_test)
    rec_test = recall_score(y_test, y_pred_test)

    # Validação cruzada (5 folds) para robustez da estimativa
    cv_scores = cross_val_score(modelo, Xtr, y_train, cv=5, scoring="accuracy")

    resultados.append({
        "Modelo": nome,
        "Acuracia_Treino": round(acc_train, 3),
        "Acuracia_Teste": round(acc_test, 3),
        "Gap_Treino_Teste": round(acc_train - acc_test, 3),
        "Acuracia_CV5_Media": round(cv_scores.mean(), 3),
        "Acuracia_CV5_Std": round(cv_scores.std(), 3),
        "Precisao_Teste": round(prec_test, 3),
        "Recall_Teste": round(rec_test, 3),
        "F1_Teste": round(f1_test, 3),
    })

    print(f"\n===== {nome} =====")
    print(classification_report(y_test, y_pred_test, target_names=["Nao_Comprou", "Comprou"]))

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                xticklabels=["Pred: Nao", "Pred: Sim"],
                yticklabels=["Real: Nao", "Real: Sim"])
    axes[i].set_title(f"{nome}\nAcc treino={acc_train:.2f} | Acc teste={acc_test:.2f}")

plt.tight_layout()
plt.savefig("matrizes_confusao_problema1.png", dpi=150)
print("\nFigura salva: matrizes_confusao_problema1.png")

# ---------------------------------------------------------------
# 4. Tabela comparativa final
# ---------------------------------------------------------------
df_resultados = pd.DataFrame(resultados).sort_values("F1_Teste", ascending=False)
print("\n\n========== TABELA COMPARATIVA - PROBLEMA 1 ==========")
print(df_resultados.to_string(index=False))
df_resultados.to_csv("resultados_problema1.csv", index=False)
