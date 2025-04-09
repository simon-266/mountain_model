from tensorflow.keras.utils import plot_model
import tensorflow as tf

model = tf.keras.models.load_model('model.keras')
plot_model(model, "model.png", show_shapes=True)