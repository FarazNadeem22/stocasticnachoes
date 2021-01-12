# StochasticNachos
For Computer Vision: 
Readin images, resizing images, one hot encoding, normalizing images and flattening images if needed. 
Once you import this package all you need to do is set up your pants and then you can simply call functions.

Driver.

from StocasticNachos import *
import os


path = 'c:\\Users\\User\\Documents\\Python & R\\Machine Learning\\Project\\data\\'
os.chdir(path)
print(os.listdir())

train_path = path +"train"
validation_path = path + "validation"

train_list = get_list(train_path)
validate_list = get_list(validation_path)

labels = get_labels(train_path)
print(labels)

#create label index for easy lookup

label2index = dict((name, index) for index, name in enumerate(labels))
index2label = dict((index, name) for index, name in enumerate(labels))
num_classes = len(labels)

train_x,train_y = read_images(train_list,train_path, label2index,"train")
val_x,val_y = read_images(validate_list, validation_path, label2index, "validate")

plot_sample_images(train_x, train_y,index2label,"Images from Train List")

#Normlalize 

train_x, val_x = normalize_image_2(train_x,val_x)

#One hot encode

train_y,val_y = one_hot_encode(train_y,val_y, num_classes)

display_info(train_x,val_x,train_y,val_y)
