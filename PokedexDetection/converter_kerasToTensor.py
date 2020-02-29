"""
#import keras to tensorflow code and necessary packages
import keras_to_tensorflow
import argparse
import pickle

#construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-m","--model",required=True,help="path to the trained model")
ap.add_argument("-lb","--labelbin",required=True,help="path the label binarizer")
args=vars(ap.parse_args())

#load the label binarizer
lb = pickle.loads(open(args["labelbin"],"rb").read())
class_labels = lb.classes_.tolist()
print("[INFO] class labels :{}".format(class_labels))

#load the trained model
print("[INFO] loading the model...")
model = load_model(args["model"])
"""
#convert the Keras to TensorFlow using following code
from tensorflow import lite
converter = lite.TFLiteConverter.from_keras_model_file( 'pokedex.hdf5')
tfmodel = converter.convert()
open ("pokedex.tflite" , "wb") .write(tfmodel)
