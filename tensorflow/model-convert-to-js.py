import tensorflow as tf
import tensorflowjs as tfjs

model = tf.keras.models.load_model("model/1605955873")
tfjs.converters.save_keras_model(model, "modeljs/1605955873")