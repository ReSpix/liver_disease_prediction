# Предсказание заболеваний печени

На основе датасета: [Indian Liver Disease Dataset](https://www.kaggle.com/datasets/paramjeetsinghds/indian-liver-disease-dataset)

### Использованы методы:
- One Rule
- Линейная регрессия
- KNN
- Наивный байесовский классификатор
- Деревья решений
- Градиентный бустинг
- SVM
- MLP
- Ансамбль моделей

**Способ тестирования моделей**:\
Так как целевая переменная имеет дисбаланс классов был выбран метод кросс-валидации на основе `StratifiedKFold(n_splits=5)`\
Использовались метрики:
- F1-macro
- F1-weighted

### Получены результаты
Отсортированы по F1-macro:

| Модель   | F1-macro ⬇️ | F1-weighted |
| -------- | -------- | ----------- |
| StackingClassifier | 0.835 ± 0.006 | 0.828 ± 0.003 |
| XGBClassifier 🏆 | 0.834 ± 0.006 | **0.878 ± 0.004** |
| GradientBoostingClassifier | 0.814 ± 0.005 | 0.868 ± 0.003 |
| VotingClassifier | 0.808 ± 0.005 | 0.859 ± 0.003 |
| RandomForestClassifier | 0.793 ± 0.005 | 0.856 ± 0.003 |
| MLPClassifier | 0.764 ± 0.003 | 0.828 ± 0.003 |
| LogisticRegression | 0.748 ± 0.004 | 0.817 ± 0.004 |
| SVC | 0.727 ± 0.005 | 0.803 ± 0.004 |
| GaussianNB | 0.724 ± 0.003 | 0.790 ± 0.003 |
| DecisionTreeClassifier | 0.718 ± 0.005 | 0.791 ± 0.004 |
| KNeighborsClassifier | 0.688 ± 0.004 | 0.774 ± 0.004 |
| LinearSVC | 0.645 ± 0.007 | 0.742 ± 0.004 |
| BernoulliNB | 0.464 ± 0.003 | 0.561 ± 0.001 |
| OneRClassifier | 0.334 ± 0.003 | 0.448 ± 0.002 |

Средняя точность `StackingClassifier` выше чем у `XGBClassifier`, однако стандартное отклонение делает это различие статистически незначимым. Доверительные интервалы точности у этих моделей почти полностью пересекаются.\
Поэтому финальной, наиболее точной моделью можно считать именно `XGBClassifier`, так как его F1-weighted заметно выше.

### Запуск
```bash
git clone https://github.com/ReSpix/liver_disease_prediction.git
cd liver_disease_prediction/
uv sync
```