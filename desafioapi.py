import pandas as pd
from flask import Flask, jsonify,request
from flask_cors import CORS
import pickle
import numpy as np 
from tensorflow import keras
import os
import random

import cv2
import numpy as np
# from tensorflow import keras

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

app = Flask(__name__)
cors = CORS(app)


@app.route('/')
def home():
    return jsonify({'message': 'ONGI ETORRI!!'})


@app.route('/prediccion')

def prediccion():

    # Cargar el modelo desde el archivo YAML
    with open("./model.json", "r") as yaml_file:
        loaded_model_yaml = yaml_file.read()

    loaded_model = keras.models.model_from_json(loaded_model_yaml)

    # Cargar los pesos del modelo
    loaded_model.load_weights("model.h5")

### Lo nuevo
    # Obtener la lista de nombres de archivos en la carpeta
    nombres_archivos = os.listdir('./fotos/caras/')

# Seleccionar un archivo aleatorio de la lista
    archivo_aleatorio = random.choice(nombres_archivos)
    ruta_completa = os.path.join('./fotos/caras/', archivo_aleatorio)

#### Lo nuevo

    #tu_foto_path = './fotos/caras/lobezno.jpg'
   

    # Función para preprocesar la imagen antes de pasarla al modelo
    #img = cv2.imread(tu_foto_path, cv2.IMREAD_GRAYSCALE)
    
    img= cv2.imread(ruta_completa, cv2.IMREAD_GRAYSCALE )

    img = cv2.resize(img, (48, 48))
    img = img / 255.0  # Normalizar
    img = np.reshape(img, (1, 48, 48, 1))  # Agregar dimensión adicional para el batch
   
    
    # Realizar la predicción
    prediction = loaded_model.predict(img)

    # Obtener la emoción predicha
    predicted_class = np.argmax(prediction)

    # Crear un diccionario para mapear clases a emociones (personaliza según tus clases)
    mapper = {0: 'Anger', 1: 'Fear', 2: 'Happy', 3: 'Sad'}

    return jsonify ({'Emoción predicha:' : mapper[predicted_class]})

@app.route('/prediccion_varios')
def prediccion_varios():


# Cargar el modelo desde el archivo YAML
    with open("model.json", "r") as yaml_file:
        loaded_model_yaml = yaml_file.read()

    loaded_model = keras.models.model_from_json(loaded_model_yaml)

# Cargar los pesos del modelo
    loaded_model.load_weights("model.h5")


  # Obtener la lista de nombres de archivos en la carpeta
    nombres_archivos = os.listdir('./fotos/caras/')

# Seleccionar un archivo aleatorio de la lista
    archivo_aleatorio = random.choice(nombres_archivos)
    ruta_completa = os.path.join('./fotos/caras/', archivo_aleatorio)

# Cargar la foto
    img = cv2.imread(ruta_completa)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Inicializar el clasificador de caras de OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detectar caras en la imagen
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

# Procesar cada cara detectada
    for (x, y, w, h) in faces:
    # Extraer la región facial
        face_roi = gray[y:y+h, x:x+w]

    # Preprocesar la imagen y realizar la predicción
        img = cv2.resize(face_roi, (48, 48))
        img = img / 255.0  # Normalizar
        img = np.reshape(img, (1, 48, 48, 1))
        
        prediction = loaded_model.predict(img)

    # Obtener la emoción predicha
    predicted_class = np.argmax(prediction)

    # Crear un diccionario para mapear clases a emociones (personaliza según tus clases)
    mapper = {0: 'Anger', 1: 'Fear', 2: 'Happy', 3: 'Sad'}

    # Imprimir la emoción predicha para cada cara
    return jsonify ({'Emoción predicha para cara en:' : mapper[predicted_class]})

    #return jsonify ({'Emoción predicha:' : mapper[predicted_class]})


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")