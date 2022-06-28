from importlib.util import module_for_loader
from flask import Flask, render_template,request
# import tensorflow as tf
import keras
from tensorflow.keras import utils
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd


app = Flask(__name__)
#model = VGG16()
#model = keras.models.load_model("best_model.h5")
ref = pd.read_csv('plants.csv', header=None, index_col=0, squeeze=True).to_dict()
@app.route("/", methods=["GET"])
def hello():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def predictio():
    model = load_model("best_model.h5")
    imagefile=request.files["imagefile"]
    image_path= "./images/" + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(256,256))
    image = img_to_array(image)
    #image = image.reshape((1, image.shape[0],image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
  
    pred = np.argmax(model.predict(image))
    #yhat = model.predict(image)
    #label = decode_predictions(pred)
    #label = label[0][0]

    predi= {ref[pred]}

    return render_template("index.html", prediction=predi)




if __name__ == "__main__":
    app.run(port=3000,debug=True)