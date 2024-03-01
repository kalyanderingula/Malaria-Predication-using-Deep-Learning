from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MaxPooling2D
from matplotlib import pyplot as plt

classifier = Sequential()

# 1st model

classifier.add(Convolution2D(512, (3, 3), input_shape=(64, 64, 3), activation='relu',strides=(3,3)))

classifier.add(MaxPooling2D(pool_size=(3, 3)))
# again use to gain remaining pexel values
classifier.add(Convolution2D(256, (2, 2), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(1, 1)))
# now converting 2d into 1d 
classifier.add(Flatten())
# now connecting all the layer into single layer
classifier.add(Dense(units=64, activation='relu'))
# now softmax is use to max the predication accuracy
classifier.add(Dense(units=2, activation='softmax'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)
print("\nTraining the data...\n")
training_set = train_datagen.flow_from_directory('data\\train',
                                                 target_size=(64,64),
                                                 batch_size=32,
                                                 class_mode='categorical'
                                                 )

test_set = test_datagen.flow_from_directory('data\\test',
                                            target_size=(64,64),
                                            batch_size=32,
                                            class_mode='categorical'
                                            )
print("\n Testing the data.....\n")

history=classifier.fit_generator(training_set,steps_per_epoch =30,epochs = 20,validation_data = test_set,verbose = 1)
# data will save in .h5 file with is trained data and tested data
classifier.save(r"visualizations\model.h5")
# ggplot is used to bulid data visualization
plt.style.use("ggplot")
plt.figure()
plt.plot(history.history['accuracy'],'r',label='training accuracy',color='green')
plt.plot(history.history['val_accuracy'],label='validation accuracy',color='red')
plt.xlabel('# epochs')
plt.ylabel('accuracy')
plt.legend()
plt.savefig(r"visualizations\model_1_acc.png")
plt.show()

plt.style.use("ggplot")
plt.figure()
plt.plot(history.history['loss'],'r',label='training loss',color='green')
plt.plot(history.history['val_loss'],label='validation loss',color='red')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig(r"visualizations\model_1_loss.png")
plt.show()


vgg_acc=history.history['val_accuracy'][-1]
print(vgg_acc)
