import tensorflow as tf
import json
import requests

test_url = "https://www.telegraph.co.uk/content/dam/science/2017/09/10/TELEMMGLPICT000107300056_trans_NvBQzQNjv4BqyuLFFzXshuGqnr8zPdDWXiTUh73-1IAIBaONvUINpkg.jpeg"
img_height=299
img_width=299

test_path = tf.keras.utils.get_file("test", origin=test_url)

img = tf.keras.preprocessing.image.load_img(test_path, target_size=(img_height, img_width))
img_array = tf.keras.preprocessing.image.img_to_array(img)

data = json.dumps({"instances":[img_array.tolist()]})

response = requests.post("http://localhost:8501/v1/models/model:predict", data)


print(response.text)