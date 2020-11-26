import tensorflow as tf

model = tf.keras.models.load_model("model/1605955873")
model.summary()
model = model.layers.pop(1)
model.summary()

model.save("model/prod")