import tensorflow as tf
import keras
import numpy as np
import os
import zipfile

from keras.applications import MobileNet
from keras.layers import Dense,GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing import image
from IPython.display import Image

os.environ['KMP_DUPLICATE_LIB_OK']='True'

base_model = MobileNet(weights='imagenet',include_top=False)

len(base_model.layers)

# base_model.summary()

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024,activation='relu')(x)
x = Dense(1024,activation='relu')(x) 
x = Dense(1024,activation='relu')(x) 
x = Dense(1024,activation='relu')(x) 
preds = Dense(2,activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=preds)
for layer in model.layers[:87]:
    layer.trainable = False
for layer in model.layers[87:]:
    layer.trainable = True

len(model.layers)

#Download Data Set
# local_zip = './cats_and_dogs_filtered.zip'
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall('./')
# zip_ref.close()

Image(filename="./cats_and_dogs_filtered/train/dogs/dog.40.jpg")

#Data Generators
base_dir = './cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) 

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224,224), color_mode='rgb', batch_size=32, class_mode='categorical', shuffle=True)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) 

val_generator = val_datagen.flow_from_directory(validation_dir, target_size=(224,224), color_mode='rgb', batch_size=32, class_mode='categorical', shuffle=True)

#Training the Model
model.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

step_size_train = train_generator.n // train_generator.batch_size
step_size_val = val_generator.n // val_generator.batch_size

history = model.fit_generator(generator=train_generator, steps_per_epoch=step_size_train,validation_data=val_generator, validation_steps=step_size_val, epochs=5)

#Testing the Model
def prepare_image(file):
    img_path = ''
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded_dims)

#Test Images with Validation DataSet
Image(filename=validation_dir+"/cats/cat.2000.jpg")

img = prepare_image(validation_dir+"/cats/cat.2000.jpg")
model.predict(img)

#Test Images with Validation DataSet
Image(filename=validation_dir+"/dogs/dog.2000.jpg")

img = prepare_image(validation_dir+"/dogs/dog.2003.jpg")
model.predict(img)

#Take a file name and predict output
def predict(filen):
      img = prepare_image(filen)
  probs = model.predict(img)[0]
  pred_class = probs.argmax()
  classes = ["cat", "dog"]
  print("%s -> %.1f%% confident - %s" %(filen,probs[pred_class]*100, classes[pred_class]))

predict(validation_dir+"/dogs/dog.2003.jpg")

