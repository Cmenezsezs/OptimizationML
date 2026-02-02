"""
Otimização de Modelo de Machine Learning
Demonstração de técnicas avançadas para um Engenheiro de IA
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. GERAÇÃO E PREPARAÇÃO DOS DADOS
# ============================================================================

def generate_data(n_samples=10000, n_features=50, n_informative=30):
    """Gera dataset sintético para classificação"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=10,
        n_repeated=5,
        n_classes=2,
        random_state=42,
        flip_y=0.1
    )
    return X, y

print("=" * 80)
print("OTIMIZAÇÃO DE MODELO DE MACHINE LEARNING")
print("=" * 80)

# Gerar dados
X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nDados gerados:")
print(f"  - Training set: {X_train.shape}")
print(f"  - Test set: {X_test.shape}")
print(f"  - Distribuição de classes: {np.bincount(y_train)}")

# ============================================================================
# 2. BASELINE MODEL (sem otimização)
# ============================================================================

print("\n" + "=" * 80)
print("ETAPA 1: MODELO BASELINE (sem otimização)")
print("=" * 80)

start_time = time.time()
baseline_model = RandomForestClassifier(random_state=42)
baseline_model.fit(X_train, y_train)
baseline_time = time.time() - start_time

baseline_pred = baseline_model.predict(X_test)
baseline_score = roc_auc_score(y_test, baseline_model.predict_proba(X_test)[:, 1])

print(f"\nBaseline Performance:")
print(f"  - AUC-ROC: {baseline_score:.4f}")
print(f"  - Tempo de treino: {baseline_time:.2f}s")
print(f"  - Número de features: {X_train.shape[1]}")

# ============================================================================
# 3. FEATURE ENGINEERING E SELEÇÃO
# ============================================================================

print("\n" + "=" * 80)
print("ETAPA 2: OTIMIZAÇÃO DE FEATURES")
print("=" * 80)

# Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature Selection - SelectKBest
selector = SelectKBest(f_classif, k=30)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

print(f"\nFeature Selection:")
print(f"  - Features originais: {X_train_scaled.shape[1]}")
print(f"  - Features selecionadas: {X_train_selected.shape[1]}")
print(f"  - Redução: {(1 - X_train_selected.shape[1]/X_train_scaled.shape[1])*100:.1f}%")

# Treinar modelo com features selecionadas
model_selected = RandomForestClassifier(random_state=42)
model_selected.fit(X_train_selected, y_train)
selected_score = roc_auc_score(
    y_test, 
    model_selected.predict_proba(X_test_selected)[:, 1]
)

print(f"  - AUC-ROC após seleção: {selected_score:.4f}")
print(f"  - Melhoria: {((selected_score - baseline_score)/baseline_score)*100:+.2f}%")

# ============================================================================
# 4. HYPERPARAMETER TUNING - GRID SEARCH
# ============================================================================

print("\n" + "=" * 80)
print("ETAPA 3: HYPERPARAMETER TUNING - GRID SEARCH")
print("=" * 80)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

print(f"\nGrid Search configurado:")
print(f"  - Combinações a testar: {np.prod([len(v) for v in param_grid.values()])}")

start_time = time.time()
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train_selected, y_train)
grid_time = time.time() - start_time

grid_score = roc_auc_score(
    y_test,
    grid_search.best_estimator_.predict_proba(X_test_selected)[:, 1]
)

print(f"\nMelhores hiperparâmetros encontrados:")
for param, value in grid_search.best_params_.items():
    print(f"  - {param}: {value}")

print(f"\nResultados Grid Search:")
print(f"  - AUC-ROC: {grid_score:.4f}")
print(f"  - Tempo de busca: {grid_time:.2f}s")
print(f"  - Melhoria vs baseline: {((grid_score - baseline_score)/baseline_score)*100:+.2f}%")

# ============================================================================
# 5. HYPERPARAMETER TUNING - RANDOMIZED SEARCH (mais eficiente)
# ============================================================================

print("\n" + "=" * 80)
print("ETAPA 4: HYPERPARAMETER TUNING - RANDOMIZED SEARCH")
print("=" * 80)

param_distributions = {
    'n_estimators': [50, 100, 150, 200, 250, 300],
    'max_depth': [5, 10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

start_time = time.time()
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=50,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=0
)
random_search.fit(X_train_selected, y_train)
random_time = time.time() - start_time

random_score = roc_auc_score(
    y_test,
    random_search.best_estimator_.predict_proba(X_test_selected)[:, 1]
)

print(f"\nMelhores hiperparâmetros (Random Search):")
for param, value in random_search.best_params_.items():
    print(f"  - {param}: {value}")

print(f"\nResultados Randomized Search:")
print(f"  - AUC-ROC: {random_score:.4f}")
print(f"  - Tempo de busca: {random_time:.2f}s")
print(f"  - Melhoria vs Grid Search: {((random_score - grid_score)/grid_score)*100:+.2f}%")
print(f"  - Speedup vs Grid Search: {grid_time/random_time:.2f}x mais rápido")

# ============================================================================
# 6. ENSEMBLE DE MODELOS
# ============================================================================

print("\n" + "=" * 80)
print("ETAPA 5: ENSEMBLE DE MODELOS")
print("=" * 80)

# Treinar múltiplos modelos
rf_optimized = random_search.best_estimator_
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb_model.fit(X_train_selected, y_train)

# Predições
rf_proba = rf_optimized.predict_proba(X_test_selected)[:, 1]
gb_proba = gb_model.predict_proba(X_test_selected)[:, 1]

# Ensemble por média ponderada
ensemble_proba = 0.6 * rf_proba + 0.4 * gb_proba
ensemble_pred = (ensemble_proba >= 0.5).astype(int)
ensemble_score = roc_auc_score(y_test, ensemble_proba)

print(f"\nResultados do Ensemble:")
print(f"  - Random Forest AUC: {roc_auc_score(y_test, rf_proba):.4f}")
print(f"  - Gradient Boosting AUC: {roc_auc_score(y_test, gb_proba):.4f}")
print(f"  - Ensemble AUC: {ensemble_score:.4f}")
print(f"  - Melhoria vs melhor modelo individual: {((ensemble_score - max(roc_auc_score(y_test, rf_proba), roc_auc_score(y_test, gb_proba)))/max(roc_auc_score(y_test, rf_proba), roc_auc_score(y_test, gb_proba)))*100:+.2f}%")

# ============================================================================
# 7. RESUMO COMPARATIVO
# ============================================================================

print("\n" + "=" * 80)
print("RESUMO COMPARATIVO DE TODAS AS OTIMIZAÇÕES")
print("=" * 80)

results = pd.DataFrame({
    'Abordagem': [
        'Baseline',
        'Feature Selection',
        'Grid Search',
        'Randomized Search',
        'Ensemble'
    ],
    'AUC-ROC': [
        baseline_score,
        selected_score,
        grid_score,
        random_score,
        ensemble_score
    ],
    'Tempo (s)': [
        baseline_time,
        '-',
        grid_time,
        random_time,
        '-'
    ],
    'Melhoria (%)': [
        0,
        ((selected_score - baseline_score)/baseline_score)*100,
        ((grid_score - baseline_score)/baseline_score)*100,
        ((random_score - baseline_score)/baseline_score)*100,
        ((ensemble_score - baseline_score)/baseline_score)*100
    ]
})

print("\n", results.to_string(index=False))

print("\n" + "=" * 80)
print("MELHOR MODELO FINAL")
print("=" * 80)

print(f"\nModelo: Ensemble (Random Forest + Gradient Boosting)")
print(f"AUC-ROC: {ensemble_score:.4f}")
print(f"Melhoria total vs baseline: {((ensemble_score - baseline_score)/baseline_score)*100:.2f}%")

# Relatório de classificação final
print("\nRelatório de Classificação (Conjunto de Teste):")
print(classification_report(y_test, ensemble_pred, target_names=['Classe 0', 'Classe 1']))

# ============================================================================
# 8. DICAS DE OTIMIZAÇÃO ADICIONAL
# ============================================================================

print("\n" + "=" * 80)
print("TÉCNICAS ADICIONAIS DE OTIMIZAÇÃO")
print("=" * 80)

print("""
1. OTIMIZAÇÃO DE DADOS:
   - Cross-validation estratificada
   - Data augmentation (para dados limitados)
   - Tratamento de outliers e valores faltantes
   - Balanceamento de classes (SMOTE, undersampling)

2. OTIMIZAÇÃO DE FEATURES:
   - Feature engineering baseado em domínio
   - Recursive Feature Elimination (RFE)
   - PCA para redução de dimensionalidade
   - Feature importance analysis

3. OTIMIZAÇÃO DE HIPERPARÂMETROS:
   - Bayesian Optimization (Optuna, Hyperopt)
   - Evolutionary algorithms
   - Early stopping para modelos iterativos
   - Learning rate scheduling

4. OTIMIZAÇÃO DE MODELO:
   - Stacking de modelos
   - Blending
   - Calibração de probabilidades
   - Quantização para inferência rápida

5. OTIMIZAÇÃO COMPUTACIONAL:
   - Paralelização de treinamento
   - GPU acceleration (XGBoost, LightGBM)
   - Model pruning
   - Knowledge distillation

6. OTIMIZAÇÃO DE PRODUÇÃO:
   - Model compression
   - ONNX conversion
   - Batch prediction
   - Caching de features
""")

print("\n" + "=" * 80)
print("OTIMIZAÇÃO CONCLUÍDA!")
print("=" * 80)
