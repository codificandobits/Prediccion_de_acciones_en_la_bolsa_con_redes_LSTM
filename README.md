# Predicción de acciones en la bolsa usando redes LSTM

Código fuente de [este](https://youtu.be/3kXj6VgxbP8) video, en donde se muestra cómo usar una red LSTM para predecir el valor máximo de la acción de Apple en un instante de tiempo determinado.

El set de datos contiene registros diarios de la variación de la acción de Apple entre el año 2006 y 2017. El modelo se implementa en Keras usando la función LSTM.

Tras 20 iteraciones de entrenamiento se obtiene una predicción bastante cercana al valor real, con una diferencia que en la mayoría de los casos no supera los 10 dólares.

## Dependencias
Keras==2.2.4
numpy==1.16.3