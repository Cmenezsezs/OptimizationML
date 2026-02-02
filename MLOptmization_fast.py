"""
Otimização de Modelo de Machine Learning - Versão Rápida

"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("OTIMIZAÇÃO DE MODELO DE MACHINE LEARNING")
print("=" * 80)

# Gerar dados
X, y = make_classification(
    n_samples=5000, n_features=30, n_informative=20,
    n_redundant=5, n_classes=2, random_state=42, flip_y=0.1
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nDados gerados:")
print(f"  - Training set: {X_train.shape}")
print(f"  - Test set: {X_test.shape}")
print(f"  - Distribuição de classes: {np.bincount(y_train)}")

# BASELINE
print("\n" + "=" * 80)
print("ETAPA 1: MODELO BASELINE")
print("=" * 80)

start = time.time()
baseline = RandomForestClassifier(n_estimators=50, random_state=42)
baseline.fit(X_train, y_train)
baseline_time = time.time() - start
baseline_pred = baseline.predict(X_test)
baseline_score = roc_auc_score(y_test, baseline.predict_proba(X_test)[:, 1])
baseline_acc = accuracy_score(y_test, baseline_pred)

print(f"\nBaseline Performance:")
print(f"  - AUC-ROC: {baseline_score:.4f}")
print(f"  - Accuracy: {baseline_acc:.4f}")
print(f"  - Tempo: {baseline_time:.2f}s")
print(f"  - Features: {X_train.shape[1]}")

# FEATURE SELECTION
print("\n" + "=" * 80)
print("ETAPA 2: FEATURE SELECTION")
print("=" * 80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

selector = SelectKBest(f_classif, k=20)
X_train_sel = selector.fit_transform(X_train_scaled, y_train)
X_test_sel = selector.transform(X_test_scaled)

model_sel = RandomForestClassifier(n_estimators=50, random_state=42)
model_sel.fit(X_train_sel, y_train)
sel_score = roc_auc_score(y_test, model_sel.predict_proba(X_test_sel)[:, 1])
sel_acc = accuracy_score(y_test, model_sel.predict(X_test_sel))

print(f"\nFeature Selection:")
print(f"  - Features: {X_train.shape[1]} → {X_train_sel.shape[1]} (-{(1-X_train_sel.shape[1]/X_train.shape[1])*100:.0f}%)")
print(f"  - AUC-ROC: {sel_score:.4f} ({(sel_score-baseline_score)*100:+.2f}%)")
print(f"  - Accuracy: {sel_acc:.4f}")

# HYPERPARAMETER TUNING
print("\n" + "=" * 80)
print("ETAPA 3: HYPERPARAMETER TUNING")
print("=" * 80)

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2']
}

start = time.time()
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=3, scoring='roc_auc', n_jobs=-1
)
grid.fit(X_train_sel, y_train)
grid_time = time.time() - start

grid_score = roc_auc_score(y_test, grid.best_estimator_.predict_proba(X_test_sel)[:, 1])
grid_acc = accuracy_score(y_test, grid.best_estimator_.predict(X_test_sel))

print(f"\nMelhores hiperparâmetros:")
for k, v in grid.best_params_.items():
    print(f"  - {k}: {v}")

print(f"\nPerformance:")
print(f"  - AUC-ROC: {grid_score:.4f} ({(grid_score-baseline_score)*100:+.2f}%)")
print(f"  - Accuracy: {grid_acc:.4f}")
print(f"  - Tempo de busca: {grid_time:.2f}s")

# ENSEMBLE
print("\n" + "=" * 80)
print("ETAPA 4: ENSEMBLE DE MODELOS")
print("=" * 80)

rf_opt = grid.best_estimator_
gb = GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42)
gb.fit(X_train_sel, y_train)

rf_proba = rf_opt.predict_proba(X_test_sel)[:, 1]
gb_proba = gb.predict_proba(X_test_sel)[:, 1]
ensemble_proba = 0.6 * rf_proba + 0.4 * gb_proba
ensemble_pred = (ensemble_proba >= 0.5).astype(int)
ensemble_score = roc_auc_score(y_test, ensemble_proba)
ensemble_acc = accuracy_score(y_test, ensemble_pred)

print(f"\nResultados do Ensemble:")
print(f"  - Random Forest AUC: {roc_auc_score(y_test, rf_proba):.4f}")
print(f"  - Gradient Boosting AUC: {roc_auc_score(y_test, gb_proba):.4f}")
print(f"  - Ensemble AUC: {ensemble_score:.4f} ({(ensemble_score-baseline_score)*100:+.2f}%)")
print(f"  - Ensemble Accuracy: {ensemble_acc:.4f}")

# RESUMO
print("\n" + "=" * 80)
print("RESUMO COMPARATIVO")
print("=" * 80)

results = pd.DataFrame({
    'Modelo': ['Baseline', 'Feature Selection', 'Grid Search', 'Ensemble'],
    'AUC-ROC': [baseline_score, sel_score, grid_score, ensemble_score],
    'Accuracy': [baseline_acc, sel_acc, grid_acc, ensemble_acc],
    'Melhoria (%)': [
        0,
        (sel_score-baseline_score)/baseline_score*100,
        (grid_score-baseline_score)/baseline_score*100,
        (ensemble_score-baseline_score)/baseline_score*100
    ]
})

print("\n", results.to_string(index=False))

print("\n" + "=" * 80)
print("MODELO FINAL - RELATÓRIO DE CLASSIFICAÇÃO")
print("=" * 80)
print("\n", classification_report(y_test, ensemble_pred, target_names=['Classe 0', 'Classe 1']))

print("=" * 80)
print("OTIMIZAÇÃO CONCLUÍDA!")
print("=" * 80)

# Salvar resultados
with open('/home/claude/resultados.txt', 'w') as f:
    f.write("RESULTADOS DA OTIMIZAÇÃO\n")
    f.write("="*50 + "\n\n")
    f.write(f"AUC-ROC Baseline: {baseline_score:.4f}\n")
    f.write(f"AUC-ROC Feature Selection: {sel_score:.4f}\n")
    f.write(f"AUC-ROC Grid Search: {grid_score:.4f}\n")
    f.write(f"AUC-ROC Ensemble: {ensemble_score:.4f}\n\n")
    f.write(f"Accuracy Baseline: {baseline_acc:.4f}\n")
    f.write(f"Accuracy Ensemble: {ensemble_acc:.4f}\n\n")
    f.write(f"Melhoria Total: {(ensemble_score-baseline_score)/baseline_score*100:.2f}%\n")
    f.write(f"Features: {X_train.shape[1]} → {X_train_sel.shape[1]}\n")

print("\nResultados salvos em: /home/results.txt")
