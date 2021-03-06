# caffe_root = "/hpc/sw/caffe-2015.11.30-gpu/"
# caffe_source = "/home/ml0501/erdi/caffe/"
# data_root = "/home/ml0501/statefarm/"
caffe_root = "/Users/erdicalli/dev/tools/caffe/"
caffe_source = "/Users/erdicalli/dev/tools/caffe/"
data_root = "/Users/erdicalli/dev/workspace/statefarm-data/"

import os
import sys

import numpy as np

sys.path.insert(0, caffe_root + 'python')

import caffe

## Use GPU
# caffe.set_device(0)
# caffe.set_mode_gpu()

# LAYERS = ['fc7']
# LAYERS = ['fc6']
LAYERS = ['fc6', 'fc7']
file_identifier = "".join(LAYERS)
LAYER_SIZE = 4096
DATA_POINTS = LAYER_SIZE * len(LAYERS)


def extract_features(images, layers):
    net = caffe.Net(caffe_source + 'models/bvlc_reference_caffenet/deploy.prototxt',
                    caffe_source + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                    caffe.TEST)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(
        1))  # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB]]

    num_images = len(images)
    net.blobs['data'].reshape(num_images, 3, 227, 227)
    net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data', caffe.io.load_image(x)), images)
    out = net.forward()
    combined_features = None
    if len(layers) > 1:
        for layer in layers:
            if combined_features is None:
                combined_features = net.blobs[layer].data
            else:
                combined_features = np.append(combined_features, net.blobs[layer].data, axis=1)
        normalizing_constant = np.linalg.norm(combined_features, axis=1)
        combined_features = np.divide(combined_features, normalizing_constant[:, None])
    else:
        combined_features = np.array(net.blobs[layers[0]].data)
    return combined_features

# extract image features and save it to .h5

# Initialize files
import h5py

# f.close()
f = h5py.File(data_root + 'train_image_' + file_identifier + 'features.h5', 'w')
filenames = f.create_dataset('photo_id', (0,), maxshape=(None,), dtype='|S54')
classes = f.create_dataset('class', (0,), maxshape=(None,), dtype='|S54')
feature = f.create_dataset('feature', (0, DATA_POINTS), maxshape=(None, DATA_POINTS))
f.close()

import pandas as pd

train_photos = pd.read_csv(data_root + 'driver_imgs_list.csv')
train_folder = data_root + 'train/'

train_images = np.array([[os.path.join(train_folder, x[0], x[1]), x[0], x[1]] for x in
                         zip(train_photos['classname'], train_photos['img'])])

num_train = len(train_images)
print "Number of training images: ", num_train
batch_size = 500

# Training Images
for i in range(0, num_train, batch_size):
    images = train_images[i: min(i + batch_size, num_train)]
    features = extract_features(images[:, 0], layers=LAYERS)
    num_done = i + features.shape[0]
    f = h5py.File(data_root + 'train_image_' + file_identifier + 'features.h5', 'r+')
    f['photo_id'].resize((num_done,))
    f['photo_id'][i: num_done] = np.array(images[:, 2])
    f['class'].resize((num_done,))
    f['class'][i: num_done] = np.array(images[:, 1])
    f['feature'].resize((num_done, features.shape[1]))
    f['feature'][i: num_done, :] = features
    f.close()
    # if num_done % 2000 == 0 or num_done == num_train:
    print "Train images processed: ", num_done

### Check the file content

f = h5py.File(data_root + 'train_image_' + file_identifier + 'features.h5', 'r')
print 'train_image_features.h5:'
for key in f.keys():
    print key, f[key].shape

print "\nA photo:", f['photo_id'][0]
print "Its feature vector (first 10-dim): ", f['feature'][0][0:10], " ..."
f.close()

f = h5py.File(data_root + 'test_image_' + file_identifier + 'features.h5', 'w')
filenames = f.create_dataset('photo_id', (0,), maxshape=(None,), dtype='|S54')
feature = f.create_dataset('feature', (0, DATA_POINTS), maxshape=(None, DATA_POINTS))
f.close()

test_folder = data_root + 'test/'
test_images = np.array([[os.path.join(test_folder, x), x] for x in os.listdir(test_folder)])

num_test = len(test_images)
print "Number of test images: ", num_test

# Test Images
for i in range(0, num_test, batch_size):
    images = test_images[i: min(i + batch_size, num_test)]
    features = extract_features(images[:, 0], layers=LAYERS)
    num_done = i + features.shape[0]

    f = h5py.File(data_root + 'test_image_' + file_identifier + 'features.h5', 'r+')
    f['photo_id'].resize((num_done,))
    f['photo_id'][i: num_done] = np.array(images[:, 1])
    f['feature'].resize((num_done, features.shape[1]))
    f['feature'][i: num_done, :] = features
    f.close()
    if num_done % 20000 == 0 or num_done == num_test:
        print "Test images processed: ", num_done

### Check the file content
f = h5py.File(data_root + 'test_image_' + file_identifier + 'features.h5', 'r')
for key in f.keys():
    print key, f[key].shape

print "\nA photo:", f['photo_id'][0]
print "feature vector: (first 10-dim)", f['feature'][0][0:10], " ..."
f.close()
