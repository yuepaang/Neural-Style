import pickle
import numpy as np
import os
import imageio
import scipy
import skimage.transform as st
from sklearn.utils import shuffle


num_char = 200
data_dir = "D:/BaiduNetdiskDownload/HWDB1/"
train_data_dir = os.path.join(data_dir, 'train')
test_data_dir = os.path.join(data_dir, 'test')

f = open('D:/BaiduNetdiskDownload/HWDB1/char_dict','rb')
char_dict = pickle.load(f)
f.close()
char_set = []
for i in range(num_char):
    char_set.append(list(char_dict.keys())[list(char_dict.values()).index(i)])


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def image_preprocessing(img):
    pad_size = abs(img.shape[0]-img.shape[1]) // 2
    if img.shape[0] < img.shape[1]:
        pad_dims = ((pad_size, pad_size), (0, 0))
    else:
        pad_dims = ((0, 0), (pad_size, pad_size))
    img = np.lib.pad(img, pad_dims, mode='constant', constant_values=255)
    img = st.resize(img, (64 - 4*2, 64 - 4*2), mode='constant')
    img = np.lib.pad(img, ((4, 4), (4, 4)), mode='constant', constant_values=255)
    assert img.shape == (64, 64)
    img = img.flatten()
    img = (img - 128) / 128
    return img


def one_hot(char):
    
    vector = np.zeros(len(char_set), dtype=float)
    vector[char_set.index(char)] = 1
    return vector


def DataLoading(path):
    x, y = [], []
    for i, v in enumerate(os.listdir(path)):
        if i == num_char:
            break
        files= os.listdir(os.path.join(path, v))
        for f in files:
            img=imageio.imread(path + '/'+ v + '/'+ f)
            gray = rgb2gray(img)
            img = image_preprocessing(gray)
            x.append(img)
            y.append(one_hot(char_set[i]))
    print("Completed! Total is %d characters and %d files" % (i, len(x)))
    return np.array(x, dtype=float), np.array(y, dtype=float)


train_data_x, train_data_y = DataLoading(train_data_dir)
test_data_x, test_data_y = DataLoading(test_data_dir)

# Data Shuffle
train_data_x, train_data_y = shuffle(train_data_x, train_data_y, random_state=0)
test_data_x, test_data_y = shuffle(test_data_x, test_data_y, random_state=0)

f1 = open('trainX2','wb')
pickle.dump(train_data_x, f1, protocol=2)
f1.close()
print("Save train_data_x Completed!")

f2 = open('trainY2','wb')
pickle.dump(train_data_y, f2, protocol=2)
f2.close()
print("Save train_data_y Completed!")

f3 = open('testX2','wb')
pickle.dump(test_data_x, f3, protocol=2)
f3.close()
print("Save test_data_x Completed!")

f4 = open('testY2','wb')
pickle.dump(test_data_y, f4, protocol=2)
f4.close()
print("Save test_data_y Completed!")