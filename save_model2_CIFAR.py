# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models

print('import complete')
input("Press Enter to continue...")

#load le dataset dans les array de train et de test
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# normalise les array
train_images, test_images = train_images / 255.0, test_images / 255.0


#réseau de neurones a 1 couche
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.summary()

print('import neural network complete')
input("Press Enter to continue...")


#compile le modèle
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
              


#entrainement du modèle
model.fit(train_images, train_labels, epochs=10)


print('trainning fini')
input("Press Enter to continue...")

model.save('my_model')

converter = tf.lite.TFLiteConverter.from_saved_model('my_model')
tflite_model = converter.convert()
open("converted_model", "wb").write(tflite_model)

input("Press Enter to continue...")

tflite_interpreter = tf.lite.Interpreter(model_path='converted_model')

input_details = tflite_interpreter.get_input_details()
output_details = tflite_interpreter.get_output_details()

print("== Input details ==")
print("shape:", input_details[0]['shape'])
print("type:", input_details[0]['dtype'])
print("\n== Output details ==")
print("shape:", output_details[0]['shape'])
print("type:", output_details[0]['dtype'])

input("Press Enter to continue...")
