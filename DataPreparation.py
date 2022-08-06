import os
import random
import shutil
from shutil import copyfile
labels = []

# folders for resource of image

path_labels = []

split_size = 0.8


def datapreparation(source_path):
    labels = os.listdir(source_path)
    print(labels)
    #path for each subdirectory
    for label in labels:
       path_labels.append(source_path+'/'+label)
    print(path_labels)
    # print the no of images in each label
    i = 0
    for path in path_labels:
        print(f"There are {len(os.listdir(path))} images of {labels[i]} .")
        i = i+1


def create_train_test_dirs(root_path):
    # If we are running the code again and again we might need to delete the directory
    if os.path.exists(root_path):
        shutil.rmtree(root_path)
    train = os.path.join(root_path,"training")
    os.makedirs(train)
    for l in labels:
        os.makedirs(train+'/'+label)
    test = os.path.join(root_path, "testing")
    os.makedirs(test)
    for l in labels:
        os.makedirs(test+'/'+l)


def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):

    all_files = []
    # files whose size is not zero there filenames will be stored as list

    # if the testing directory is already present then it will be deleted

    if os.path.exists(TESTING):
        shutil.rmtree(TESTING)
    if os.path.exists(TRAINING):
        shutil.rmtree(TRAINING)
    for file_name in os.listdir(SOURCE):
        file_path = SOURCE + '/' + file_name

        if os.path.getsize(file_path):
            all_files.append(file_name)
        else:
            print('{} is zero length, so ignoring'.format(file_name))

    n_files = len(all_files)
    split_point = int(n_files * SPLIT_SIZE)

    shuffled = random.sample(all_files, n_files)

    train_set = shuffled[:split_point]
    test_set = shuffled[split_point:]
    os.mkdir(TRAINING)
    os.mkdir(TESTING)
    for file_name in test_set:
        #print(file_name)
        copyfile(SOURCE+"/"+file_name, TESTING+"/"+file_name)
    for file_name in train_set:
        #print(file_name)
        copyfile(SOURCE + "/" + file_name, TRAINING + "/" + file_name)


Source_files = "C:/Users/91992/Downloads/DigitDataset/Sign-Language-Digits-Dataset-master/Dataset"
root_path = "C:/Users/91992/Documents/Project Documents/Data"

datapreparation(Source_files)
#create_train_test_dirs(root_path)

TRAINING_DIR = "C:/Users/91992/Documents/Project Documents/Data/training"
TESTING_DIR = "C:/Users/91992/Documents/Project Documents/Data/testing"

training_dir = []
testing_dir = []
labels = os.listdir(Source_files)
for label in labels:
    training_dir.append(TRAINING_DIR+'/'+label)
    testing_dir.append(TESTING_DIR+'/'+label)

for i in range(len(labels)):
    print(path_labels[i],training_dir[i],testing_dir[i])
    split_data(path_labels[i], training_dir[i], testing_dir[i],split_size)



