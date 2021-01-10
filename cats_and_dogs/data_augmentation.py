from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

home_path = os.getcwd()
train_cats_dir = os.path.join(home_path, 'train/cats')

fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]

img_path = fnames[26]
img = image.load_img(img_path, target_size=(150,150))

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=.2,
        height_shift_range=.2,
        shear_range=.2,
        zoom_range=.2,
        horizontal_flip=True,
        fill_mode='nearest'
        )

x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)

i=0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i%4==0:
        break

plt.show()
