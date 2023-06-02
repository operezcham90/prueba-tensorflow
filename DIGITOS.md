# Tensorflow reconocimiento digitos

A continuación, aquí tienes un ejemplo de cómo crear y entrenar una red neuronal 
para clasificar imágenes utilizando el [conjunto de datos MNIST](https://github.com/mbornet-hl/MNIST/tree/master/IMAGES/GROUPS), 
que contiene dígitos escritos a mano:

```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Cargar el conjunto de datos MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar los valores de píxeles al rango [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Crear el modelo de red neuronal
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Evaluar el modelo
model.evaluate(x_test, y_test)
```

En este ejemplo, primero cargamos el conjunto de datos MNIST y lo dividimos en conjuntos de entrenamiento y prueba. A continuación, normalizamos los valores de los píxeles al rango [0, 1].

Luego, creamos un modelo secuencial, que es una pila lineal de capas. En este caso, utilizamos una capa de aplanamiento para convertir las imágenes 2D en un vector 1D, seguida de una capa densa con activación ReLU y una capa de salida densa con activación softmax para la clasificación de las 10 posibles clases de dígitos.

Después de crear el modelo, lo compilamos especificando el optimizador, la función de pérdida y las métricas que se utilizarán durante el entrenamiento.

Luego, entrenamos el modelo utilizando el método `fit` y especificamos el número de épocas, el tamaño del lote y los datos de validación.

Finalmente, evaluamos el modelo en el conjunto de prueba utilizando el método `evaluate` y obtenemos la precisión de clasificación.

Para visualizar algunos ejemplos de salida generados por tu red neuronal entrenada, puedes utilizar la función `predict` del modelo para obtener las predicciones y luego mostrar las imágenes junto con sus etiquetas y las predicciones correspondientes. Aquí tienes un ejemplo de cómo hacerlo:

```python
import numpy as np
import matplotlib.pyplot as plt

# Obtener predicciones para algunos ejemplos
predictions = model.predict(x_test[:10])

# Obtener las etiquetas predichas
predicted_labels = np.argmax(predictions, axis=1)

# Mostrar las imágenes junto con las etiquetas y las predicciones correspondientes
for i in range(10):
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f'Etiqueta verdadera: {y_test[i]}, Etiqueta predicha: {predicted_labels[i]}')
    plt.axis('off')
    plt.show()
```

En este ejemplo, utilizamos el método `predict` del modelo para obtener las predicciones para los primeros 10 ejemplos en el conjunto de prueba (`x_test[:10]`). Luego, utilizamos la función `np.argmax` para obtener las etiquetas predichas, que corresponden al índice de la clase con el valor máximo en cada predicción.

Finalmente, utilizamos la biblioteca Matplotlib para mostrar las imágenes junto con las etiquetas verdaderas y las etiquetas predichas. Cada imagen se muestra en una ventana separada utilizando la función `imshow`, y se muestra el título que indica la etiqueta verdadera y la etiqueta predicha. La función `axis('off')` se utiliza para eliminar los ejes de coordenadas en la visualización.

Ejecuta este código después de entrenar tu modelo y verás una serie de imágenes con sus etiquetas y las predicciones generadas por tu red neuronal. Esto te permitirá tener una idea visual de cómo está funcionando tu modelo.
