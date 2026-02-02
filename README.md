# 🎯 Otimização de Modelo de Machine Learning

## 📋 Sumário 

Este projeto demonstra um pipeline completo de **otimização de modelos de Machine Learning**, aplicando técnicas avançadas de engenharia de features, tuning de hiperparâmetros e ensemble learning. O objetivo é maximizar a performance preditiva através de metodologias sistemáticas e cientificamente fundamentadas.

**Resultado Principal:** Desenvolvimento de um modelo de classificação binária com **AUC-ROC de 0.9378** e **accuracy de 88.1%**, demonstrando técnicas profissionais de otimização aplicáveis a problemas reais de produção.

---

## 🎓 Objetivo do Projeto

### Propósito
Demonstrar competências essenciais de um **Engenheiro de IA especialista em otimização**, incluindo:

- ✅ Estabelecimento de baseline para comparação
- ✅ Feature engineering e seleção de variáveis relevantes
- ✅ Hyperparameter tuning sistemático
- ✅ Técnicas de ensemble learning
- ✅ Avaliação comparativa e documentação de resultados

### Problema de Negócio
Criar um modelo de classificação binária otimizado para um dataset com 30 features, maximizando a capacidade preditiva enquanto mantém eficiência computacional e interpretabilidade.

---

## 🏗️ Arquitetura da Solução

### Pipeline de Otimização

```
┌─────────────────────┐
│   Dados Brutos      │
│   (5000 samples)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Split Train/Test   │
│   (80% / 20%)       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ ETAPA 1: Baseline   │
│ Random Forest       │
│ AUC: 0.9427         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ ETAPA 2: Feature    │
│ Selection (30→20)   │
│ AUC: 0.9334         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ ETAPA 3: Grid       │
│ Search CV           │
│ AUC: 0.9384         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ ETAPA 4: Ensemble   │
│ RF + GBM            │
│ AUC: 0.9378 ✓       │
└─────────────────────┘
```

---

## 📊 Resultados Detalhados

### Comparativo de Performance

| Modelo              | AUC-ROC | Accuracy | Features | Observações                    |
|---------------------|---------|----------|----------|--------------------------------|
| **Baseline**        | 0.9427  | 89.7%    | 30       | Random Forest sem otimização   |
| **Feature Selection** | 0.9334  | 88.0%    | 20       | Redução de 33% nas features    |
| **Grid Search**     | 0.9384  | 88.9%    | 20       | Hiperparâmetros otimizados     |
| **Ensemble Final**  | 0.9378  | 88.1%    | 20       | RF + Gradient Boosting         |

### Métricas do Modelo Final (Ensemble)

#### Matriz de Confusão - Conjunto de Teste
```
                 Predito
                 Neg    Pos
Real   Neg      437     66
       Pos       53    444
```

#### Relatório de Classificação

| Classe    | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Classe 0  | 0.89      | 0.87   | 0.88     | 503     |
| Classe 1  | 0.87      | 0.90   | 0.88     | 497     |
| **Média** | **0.88**  | **0.88** | **0.88** | **1000** |

---

## 🔬 Metodologia Aplicada

### 1️⃣ Modelo Baseline
**Objetivo:** Estabelecer linha de base para comparação

- **Algoritmo:** Random Forest (50 árvores)
- **Configuração:** Parâmetros padrão do scikit-learn
- **Resultado:** AUC-ROC 0.9427, Accuracy 89.7%
- **Tempo de treinamento:** 1.03s

**Insight:** Modelo baseline já apresenta excelente performance, indicando que o problema tem boa separabilidade.

### 2️⃣ Feature Selection
**Objetivo:** Reduzir dimensionalidade e eliminar ruído

- **Técnica:** SelectKBest com teste F-ANOVA
- **Redução:** 30 → 20 features (-33%)
- **Resultado:** AUC-ROC 0.9334
- **Impacto:** Pequena redução em performance (-0.99%), mas ganho em interpretabilidade e velocidade

**Insight:** A leve queda sugere que algumas features removidas continham informação útil, porém a redução de dimensionalidade facilita deployment.

### 3️⃣ Hyperparameter Tuning
**Objetivo:** Maximizar performance através de otimização sistemática

- **Técnica:** Grid Search com 3-fold Cross-Validation
- **Espaço de busca:** 16 combinações de hiperparâmetros
- **Tempo de busca:** 15.55s
- **Resultado:** AUC-ROC 0.9384

**Melhores hiperparâmetros encontrados:**
```python
{
    'n_estimators': 100,
    'max_depth': 20,
    'min_samples_split': 2,
    'max_features': 'sqrt'
}
```

**Insight:** Aumento de árvores (50→100) e profundidade controlada melhoraram a generalização.

### 4️⃣ Ensemble Learning
**Objetivo:** Combinar múltiplos modelos para melhor performance

- **Arquitetura:** Blending de Random Forest + Gradient Boosting
- **Pesos:** 60% RF + 40% GBM
- **Resultado:** AUC-ROC 0.9378, Accuracy 88.1%

**Performance individual:**
- Random Forest: 0.9384
- Gradient Boosting: 0.9286
- **Ensemble:** 0.9378 (robusto e estável)

**Insight:** Ensemble oferece maior robustez e estabilidade, embora com performance similar ao melhor modelo individual.

---

## 🎯 Análise Crítica

### Pontos Fortes ✅
1. **Pipeline estruturado e reprodutível**
2. **Validação cruzada** para evitar overfitting
3. **Múltiplas técnicas de otimização** aplicadas sistematicamente
4. **Documentação completa** de cada etapa
5. **Modelos com alta performance** (AUC > 0.93)

### Limitações e Considerações ⚠️
1. **Dataset sintético:** Resultados podem variar em dados reais
2. **Classe balanceada:** Performance pode cair em datasets desbalanceados
3. **Trade-off interpretabilidade vs. performance:** Ensemble é menos interpretável
4. **Tempo de treinamento:** Grid Search pode ser custoso em produção

### Próximos Passos 🚀

#### Otimizações Adicionais Recomendadas:

1. **Hyperparameter Tuning Avançado**
   - Implementar Bayesian Optimization (Optuna/Hyperopt)
   - Testar Randomized Search para exploração mais ampla
   - Early stopping em modelos iterativos

2. **Feature Engineering**
   - Engenharia de features baseada em domínio
   - Interações polinomiais entre features
   - Recursive Feature Elimination (RFE)

3. **Modelos Avançados**
   - XGBoost / LightGBM para maior velocidade
   - Stacking de múltiplos níveis
   - Neural Networks para comparação

4. **Validação Robusta**
   - Stratified K-Fold CV (k=5 ou k=10)
   - Validação em dataset holdout separado
   - Análise de curvas de aprendizado

5. **Otimização para Produção**
   - Model compression e quantização
   - ONNX conversion para deployment
   - Monitoramento de drift de dados
   - A/B testing de modelos

---

## 💻 Tecnologias Utilizadas

### Stack Principal
- **Python 3.12**
- **scikit-learn 1.5+** - Machine Learning
- **NumPy** - Computação numérica
- **Pandas** - Manipulação de dados

### Algoritmos Implementados
- Random Forest Classifier
- Gradient Boosting Classifier
- SelectKBest (Feature Selection)
- Grid Search CV
- Ensemble Learning (Blending)

---

## 📁 Estrutura do Projeto

```
.
├── ml_optimization_fast.py    # Script principal de otimização
├── ml_optimization.py          # Versão completa com mais técnicas
├── resultados.txt              # Resultados numéricos salvos
└── README.md                   # Este arquivo
```

---

## 🚀 Como Executar

### Pré-requisitos
```bash
pip install scikit-learn numpy pandas --break-system-packages
```

### Execução
```bash
python ml_optimization_fast.py
```

### Saída Esperada
O script irá:
1. ✅ Gerar dataset sintético
2. ✅ Treinar modelo baseline
3. ✅ Aplicar feature selection
4. ✅ Executar grid search
5. ✅ Criar ensemble
6. ✅ Exibir comparativo completo
7. ✅ Salvar resultados em `resultados.txt`

---

## 📈 Conclusões

Este projeto demonstra um **workflow profissional de otimização de ML**, aplicando técnicas state-of-the-art que são essenciais para um Engenheiro de IA:

### Principais Aprendizados:
1. ✅ **Sempre estabeleça um baseline** antes de otimizar
2. ✅ **Feature selection** pode melhorar eficiência sem sacrificar muito a performance
3. ✅ **Hyperparameter tuning** é essencial, mas deve ser balanceado com tempo computacional
4. ✅ **Ensembles** oferecem robustez, mas adicioram complexidade
5. ✅ **Documentação e experimentação** são fundamentais para projetos de ML em produção

### Impacto para Produção:
- **Modelo robusto** com AUC-ROC superior a 0.93
- **Pipeline reprodutível** e bem documentado
- **Features reduzidas** facilitam manutenção e deployment
- **Metodologia científica** permite iteração e melhoria contínua

---

## 👤 Autor

Desenvolvido como demonstração de competências em **Otimização de Modelos de Machine Learning** para posição de Engenheiro de IA.

### Competências Demonstradas:
- 🎯 Feature Engineering e Selection
- 🎯 Hyperparameter Optimization
- 🎯 Ensemble Learning
- 🎯 Model Evaluation e Validation
- 🎯 Python e scikit-learn
- 🎯 Documentação Técnica

---

## 📄 Licença

Este projeto é disponibilizado para fins educacionais e de demonstração técnica.

---

## 📚 Referências

- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Ensemble Methods in Machine Learning](https://link.springer.com/chapter/10.1007/3-540-45014-9_1)
- [Feature Selection Methods](https://jmlr.org/papers/v3/guyon03a.html)
- [Hyperparameter Optimization](https://www.jmlr.org/papers/v13/bergstra12a.html)

---

**Data de criação:** Fevereiro 2026  
**Versão:** 1.0  
**Status:** ✅ Produção
