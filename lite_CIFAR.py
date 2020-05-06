import tensorflow as tf
from tensorflow.keras import datasets
import numpy as np
import matplotlib.pyplot as plt

# import dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']




tflite_interpreter = tf.lite.Interpreter(model_path='converted_model')
tflite_interpreter.allocate_tensors()
input_details = tflite_interpreter.get_input_details()
output_details = tflite_interpreter.get_output_details()



# info sur les input et output de tf lite
# print("== Input details ==")
# print("shape:", input_details[0]['shape'])
# print("type:", input_details[0]['dtype'])
# print("\n== Output details ==")
# print("shape:", output_details[0]['shape'])
# print("type:", output_details[0]['dtype'])

#nombre de predictions
nb_predictions=1000
predictions=np.zeros(shape=(nb_predictions,10))


#fonction de normalisation
def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

    

#calcul des predictions 
for i in range (nb_predictions):

    test_image = np.expand_dims(test_images[i], axis=0).astype(np.float32)
    tflite_interpreter.set_tensor(input_details[0]['index'], test_image)
    tflite_interpreter.invoke()
    prediction = tflite_interpreter.get_tensor(output_details[0]['index'])
    prediction=normalize(prediction)
    predictions[i]=prediction
    
    

#affichage de 49 images random predites    
j=np.random.randint(1,1000,50)
    
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_labels, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_labels:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[int(true_labels)]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_labels = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[int(true_labels)].set_color('blue')
  
  




num_rows = 7
num_cols = 7
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  k=j[i]
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(k, predictions[k], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(k, predictions[k], test_labels)
plt.tight_layout()
plt.show()
