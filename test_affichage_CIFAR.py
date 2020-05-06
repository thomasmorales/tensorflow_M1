import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

print('import complete')
input("Press Enter to continue...")

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

j=0
plt.figure(figsize=(7,7))
#plt.colorbar()
input("Press Enter to continue...")

for i in range(64):
    plt.subplot(8,8,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    
    while class_names[train_labels[j][0]] != 'airplane':
        j+=1
    
    plt.imshow(train_images[j], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[j][0]])
    j+=1
plt.show()

input("Press Enter to continue...")