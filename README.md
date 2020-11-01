# Хакатон "Лидеры цифровой трансформации"
# Трек: Разработка инструмента оценки качества работы алгоритмов медицинских изображений.

## Short Summary


Архитектура репозитория

```
data/
   Dataset/
      Origin/
      .../
    masks_aggregate/
    masks012/
 src/
 notebooks/
```


## Этапы решения

#### Сегментация легких на снимках.

Линк на предобученную сегментационную модель с открытого репозитория [here](https://github.com/IlliaOvcharenko/lung-segmentation). Прогнали через наш датасет и сохранили полученные маски в ./data/masks_aggregate.

#### Препроцессинг сегментационных масок.

С помощью ./notebooks/lung_clustering.ipynb :

  - Очистили маски от выбросов.
  - Получение anatomical prior shape.
  - Выделение масок левого, правого легкого и области вне легких.

Сохранили полученные маски и среднее в ./data/masks012.

#### Тренировка модели и получение предсказания.

  - Features mining & Feature Selection.
  - Model training.
  - Pseudo-labeling.
  - Model training on pseudo-labeling.
  - Постпроцессинг.
