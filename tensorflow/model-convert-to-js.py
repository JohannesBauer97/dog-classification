import tensorflow as tf
import tensorflowjs as tfjs

model = tf.keras.models.load_model("model/1605955873")
model = model.layers.pop(1)
model.summary()

# model.save("model/prod")

tfjs.converters.save_keras_model(model, "modeljs/1605955873")