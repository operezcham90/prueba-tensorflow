# Prueba con TensorFlow

Enlace de Google Colab: [Tutorial Clasificación](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb)

Para dibujar el modelo:

```python
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True)
```

Para guardar el modelo:

```python
from google.colab import files
model.save("model.h5")
files.download("model.h5")
```

Instalar herramienta para dibujar modelo:
```python
!pip install visualkeras
import visualkeras
visualkeras.layered_view(model, legend=True, scale_xy=2.0)
```
