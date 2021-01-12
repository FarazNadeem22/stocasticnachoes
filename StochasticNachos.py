import os
import cv2 
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical

def get_lables(path):
    """
    Takes an os path as an argument and returns a list of directories in the path provided
    Dependency: OS 
    """
    
    #save the original path and set the working directory at the end back to the original path
    original_path = os.getcwd()

    #change directory to path provided
    os.chdir(path)

    #create empty list
    lst = []

    #populate the list with names of directories 
    for dirs in os.listdir():
        lst.append(dirs)
    
    #change path back to original path
    os.chdir(original_path)

    return lst 

def label2index(lables):
    """
    return a label to index dictionary. It takes a list - such as the output from the get_lables function.
    Dependency: None
    """
    label2index = dict((name, index) for index, name in enumerate(labels))
    return label2index

def index2label(lables):
    """
    return a index to label  dictionary. It takes a list - such as the output from the get_lables function.
    Dependency: None
    """
    index2label = dict((index, name) for index, name in enumerate(labels))
    return index2label

def get_numClasses(lables):
    """
    Returns the number of classes
    It takes a list - such as the output from the get_lables function.
    Dependency: None
    """
    return len(lables)

def get_list(working_path):
    """
    Takes one argument as the working path.This function goes through each folder one by one and reads each file and saves it as a tuple of two 
    items such as (imageclass<directoryname>, filename). read_images uses the output of this function as its first input
    Dependency: os
    """
    #save the original path and set the working directory at the end back to the original path
    original_path = os.getcwd()

    #change directory to path provided
    os.chdir(working_path)

    #create empty list
    lst = []

    #primize counter 
    counter = 0

    #read and populate the list, lst
    for dirs in os.listdir():
        tmp_path = working_path +"/"+ str(dirs)
        #os.chdir(tmp_path) #this is redundent 
        for images in os.listdir(tmp_path):
            lst.append((dirs,images))
            counter += 1
        #print information about how many images were found in each directory 
        print(dirs, " count = ", counter )
        #set counter back to zero
        counter = 0
    return lst

def read_images(lst, PATH, label2index, name="list", width = 50, height = 50, printdetails = True):
    """
    Takes three mandatory and one optional argument. The first three mandatory arguments are 1. list - this is the out put from the get_list function, 
    the path where the base directory is and the label2index again which is the output from label2index function. The last argument is the name like train test validate.
    Dependency: os, numpy and cv2
    Ammendment 1: Default height and width of 50,50 have been added. If you  want your images to be of different height and width please pass those sizes when calling.
    Ammendment 2: Print Detials is set to True but you can disable it by passing false.
    this function.
    """
    #Set x and y as empty lists. x will hold the images and y their corresponding index
    x,y = [],[]
    
    #primeize a counter 
    counter = 0 

    #start reading data
    for labels, files in lst:
        image_path = os.path.join(PATH,str(labels),files)
        #check if for some reason image is gif
        if image_path[:-3] == 'gif':
            print("Data Clearning Run Time Handeling: Not appending -> ", image_path)
        else:
            try:
                image = cv2.imread(image_path)
                #Since cv2 reads image in as Blue Green Red We will convert it back to our traditional Red Green Blue
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                # we will call our own function to resize the images as per your requirement
                image = resize_image(image, width=width, height=height)
                #Append the images to x and their coresponding index to y 
                x.append(image)
                y.append([label2index[labels]])
                counter += 1
            except NameError:
                print("Error in Try block ->", image_path)

    #This will not show if printdetails argument as been set to false
    if printdetails:
        print(len(x))
        print(x[0].shape)
    
    #Convert to numpy array
    x = np.asarray(x)
    y = np.asarray(y)

    #This will not show if printdetails argument as been set to false
    if printdetails:
        print("x_{} shape:".format(name),x.shape)
        print("y_{} shape:".format(name),y.shape)
        print(x[0][:5,:5,0])
        print(y[:5])
        print("*********"*5,"\n")
    return x,y

def resize_image(image, height=200, width=200):
    """
    Takes and image and resizes it to 200,200 by default, but you can give it other dimensions if needed
    Dependency: cv2
    """
    dim = (height,width)
    image = cv2.resize(image,dim, interpolation= cv2.INTER_AREA)
    return image

def plot_sample_images(x_list,y_list,index2label, title="Sample Images", axis=False):
    """VOID FUNCTION: PLOTS A SET OF IMAGES
    Mandatory arguments: Takes three arguments as input x_list, y_list amd index2label. 
    Optional arguments: The name of the tile is an optional parameter and axis if off by default it can be turned on with passing True for axis
    Dependency: matplotlib
    """
    # Generate a sample of index 
    train_image_sample = np.random.randint(0, high=x_list.shape[0], size=50)

    #Train images 
    fig = plt.figure(figsize=(15,10))
    for i,img_idx in enumerate(train_image_sample):
        axs = fig.add_subplot(5,10,i+1)
        axs.set_title(index2label[y_list[img_idx][0]])
        plt.imshow(x_list[img_idx])
        if axis:
            continue
        else:
            plt.axis('off')

    plt.suptitle(title)
    plt.show()
    
def one_hot_encode(train_y,validate_y,num_classes):
    """
    Takes three inpurts train_y,validate_y, num_classes encodes in a one hot encode matrix
    Dependncy: from tensorflow.keras.utils import to_categorical
    """
    train_y = to_categorical(train_y,num_classes=num_classes, dtype='float32')
    validate_y = to_categorical(validate_y, num_classes=num_classes, dtype='float32')
    return train_y, validate_y

def normalize_image(train_x,validate_x, test_x):
    """
    Takes three inputs and returns noralized train validate and test x, suggested names to assing to return values train_x, validate_x, test_x
    Dependency:  
    """
    train_x = train_x.astype("float")/255.0
    validate_x = validate_x.astype("float")/255.0
    test_x = test_x.astype("float")/255.0
    return train_x,validate_x,test_x

def flatten_image(train_x,validate_x,test_x,image_width,image_height,num_channels): 
    """Takes six inputs train,validate,test,image_width,image_height,num_channels which should be NORMALIZED i.e is between 0 and 1 
    and returns normalized train validate and test suggestes names to assign to return values train_x, 
    validate_x, test_"""
    flatten_shape = image_width * image_height * num_channels
    train_x = train_x.reshape(train_x.shape[0],flatten_shape)
    validate_x = validate_x.reshape(validate_x.shape[0],flatten_shape)
    test_x = test_x.reshape(test_x.shape[0],flatten_shape)
    return train_x,validate_x,test_x

def drop_wrongsize(lst,PATH, height=200, width=200):
    """
    VOID FUCNTION: Takes a lst of images and its path as an argument (MANDATORY) and drops the images that are not of correct size
    can also take height and width as argument default height and width = 200 each
    """
    counter = 0
    for labels, file in lst:
        image_path = os.path.join(PATH,str(labels),file)
        image=cv2.imread(image_path)
        if image.shape[0] != height:
            print("Removing: ",image_path)
            os.remove(image_path)
            counter += 1
        if image.shape[1] != width:
            print(image_path)
            os.remove("Removing: ",image_path)
            counter += 1
    print("\n {} images dropped from lst".format(counter))

def get_labels(working_path):
    """returns a list of directories in the path provided"""
    os.chdir(working_path)
    counter = 0
    lst = []
    for dirs in os.listdir():
        lst.append(dirs)
    return lst

def display_info(train_x,validate_x,train_y,validate_y):
    """VOID FUNCTION: Takes six inputs train_x,validate_x,test_x,train_y,validate_y,test_y
    and prints shapes and hot one encoded matrix """
    print("Inputs x:")
    print("train_x shape:",train_x.shape)
    print("validate_x shape:",validate_x.shape)

    print("Outputs y:")
    print("train_y shape:",train_y.shape)
    print("validate_y shape:",validate_y.shape)
   
    print("Inputs:\n",train_x[:5])
    print("Outputs:\n",train_y[:5])

def normalize_image_2(train_x,validate_x):
    """
    Takes two inputs and returns noralized train validate , suggested names to assing to return values train_x, validate_x
    Dependency:  
    """
    train_x = train_x.astype("float")/255.0
    validate_x = validate_x.astype("float")/255.0
    return train_x,validate_x