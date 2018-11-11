# Load VGG-net and save the image features in a dictionary
import os
import numpy as np
import tensorflow as tf
from keras.applications import vgg16
from keras.preprocessing import image
import keras.backend.tensorflow_backend as KTF
from keras.applications.vgg16 import preprocess_input

KTF.set_session(tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))))

vgg_model = vgg16.VGG16(weights='imagenet', include_top=True)
vgg_model.layers.pop()
vgg_model.layers[-1].outbound_nodes = []
vgg_model.outputs = [vgg_model.layers[-1].output]
vgg_model.summary()

sketch_root = "/home/interns/sasi/Sketchy/sketch/tx_000100000000"

sketch_paths = []
for path, subdirs, files in os.walk(sketch_root):
    for fileName in files:
        sketch_paths.append(path + '/' + fileName)
    pass

np.save('sketch_paths', np.array(sketch_paths))

# for sketches
sketch_paths = np.load('sketch_paths.npy')

BATCH_SIZE = 41
X_out = np.zeros((len(sketch_paths), 4096))
X_in = np.zeros((BATCH_SIZE, 224, 224, 3))

for ii in range(len(sketch_paths) // BATCH_SIZE):
    print('Batch ' + str(ii) + ' in progress...')
    for jj in range(BATCH_SIZE):
        X_in[jj, :, :, :] = image.img_to_array(image.load_img(
            sketch_paths[ii * BATCH_SIZE + jj], target_size=(224, 224)))
        pass
    X_in = preprocess_input(X_in)
    X_out[ii * BATCH_SIZE:(ii + 1) * BATCH_SIZE, :] = vgg_model.predict_on_batch(X_in)
    pass

# store the image paths and vgg_features
np.save('vgg_sketch_features_mod', X_out)
np.save('sketch_paths', np.array(sketch_paths))
