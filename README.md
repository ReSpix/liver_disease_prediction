## Предсказание заболеваний печени

На основе датасета: \
https://www.kaggle.com/datasets/paramjeetsinghds/indian-liver-disease-dataset

**Использованы методы**:
- One Rule
- Линейная регрессия
- KNN
- Наивный байесовский классификатор
- Деревья решений
- Градиентный бустинг
- SVM
- MLP
- Ансамбль моделей

**Получены результаты** (отсортированы по точности):

| Модель   | Точность |
| -------- | -------- |
| XGBClassifier 🏆 | 0.877 ± 0.005 |
| StackingClassifier | 0.878 ± 0.004 |
| GradientBoostingClassifier | 0.869 ± 0.003 |
| VotingClassifier | 0.850 ± 0.003 |
| RandomForestClassifier | 0.825 ± 0.033 |
| LogisticRegression |0.807 ± 0.005|
| MLPClassifier | 0.798 ± 0.007 |
| DecisionTreeClassifier | 0.791 ± 0.003 |
| KNeighborsClassifier | 0.790 ± 0.004 |
| GaussianNB | 0.781 ± 0.003 |
| SVC | 0.756 ± 0.003 |
| LinearSVC | 0.755 ± 0.003 |
| BernoulliNB | 0.698 ± 0.084 |
| OneRClassifier |0.507 ± 0.002 |
| Предсказание самого частого класса | 0.360 |
| Случайное предсказание | 0,166 |