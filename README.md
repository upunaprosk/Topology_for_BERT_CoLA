# Topology_for_BERT_CoLA


5_* - Расчет характеристик для каждого признака каждого слоя/головы:
коэффициента корреляции значений признака с вектором ответов, p_value значение теста [Манна-Уитни](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html)

6_* - Расчет характеристик для каждого признака каждого слоя/головы: Matthews correlation coefficient(MCC)/Accuracy в результате обучения LogRegression на ```in_domain_train``` (X = значение признака) с отбором оптимального порога по roc-кривой

7_1* - Обзор константных фичей + код для отрисовки графов и матриц внимания

7_Feature_selection - Отбор признаков (по найденным в 5_* значениям + перебор по параметрам LogRegression) + метрики результатов по группам major
