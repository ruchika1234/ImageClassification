from __future__ import print_function
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter
import time
import numpy as np
import matplotlib.pyplot as plt
import random
from random import shuffle
from six.moves import cPickle as pickle
import numpy as np
import os
from imageio import imread
import platform

def extract_features(imgs, feature_fns, verbose=False):
  num_images = imgs.shape[0]
  if num_images == 0:
    return np.array([])
  feature_dims = []
  first_image_features = []
  for feature_fn in feature_fns:
    feats = feature_fn(imgs[0].squeeze())
    assert len(feats.shape) == 1, 'Feature functions must be one-dimensional'
    feature_dims.append(feats.size)
    first_image_features.append(feats)
  total_feature_dim = sum(feature_dims)
  imgs_features = np.zeros((num_images, total_feature_dim))
  imgs_features[0] = np.hstack(first_image_features).T
  for i in range(1, num_images):
    idx = 0
    for feature_fn, feature_dim in zip(feature_fns, feature_dims):
      next_idx = idx + feature_dim
      imgs_features[i, idx:next_idx] = feature_fn(imgs[i].squeeze())
      idx = next_idx
    if verbose and i % 1000 == 0:
      print('Done extracting features for %d / %d images' % (i, num_images))

  return imgs_features


def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


def hog_feature(im):
  if im.ndim == 3:
    image = rgb2gray(im)
  else:
    image = np.at_least_2d(im)

  sx, sy = image.shape 
  orientations = 9 
  cx, cy = (8, 8) 
  gx = np.zeros(image.shape)
  gy = np.zeros(image.shape)
  gx[:, :-1] = np.diff(image, n=1, axis=1) 
  gy[:-1, :] = np.diff(image, n=1, axis=0) 
  grad_mag = np.sqrt(gx ** 2 + gy ** 2) 
  grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90 

  n_cellsx = int(np.floor(sx / cx))  
  n_cellsy = int(np.floor(sy / cy))  
  orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
  for i in range(orientations):
    temp_ori = np.where(grad_ori < 180 / orientations * (i + 1),
                        grad_ori, 0)
    temp_ori = np.where(grad_ori >= 180 / orientations * i,
                        temp_ori, 0)
    cond2 = temp_ori > 0
    temp_mag = np.where(cond2, grad_mag, 0)
    orientation_histogram[:,:,i] = uniform_filter(temp_mag, size=(cx, cy))[int(cx/2)::cx, int(cy/2)::cy].T
  
  return orientation_histogram.ravel()


def color_histogram_hsv(im, nbin=10, xmin=0, xmax=255, normalized=True):
  ndim = im.ndim
  bins = np.linspace(xmin, xmax, nbin+1)
  hsv = matplotlib.colors.rgb_to_hsv(im/xmax) * xmax
  imhist, bin_edges = np.histogram(hsv[:,:,0], bins=bins, density=normalized)
  imhist = imhist * np.diff(bin_edges)
  return imhist
pass

class LinearClassifier(object):

  def __init__(self):
    self.W = None

  def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):

    num_train, dim = X.shape[0], X.shape[1]
    num_classes = np.max(y) + 1 
    if self.W is None:
      self.W = 0.001 * np.random.randn(dim, num_classes)

    loss_history = []
    for it in range(num_iters):
      X_batch = None
      y_batch = None
      batch_indices = np.random.choice(num_train, batch_size, replace=False)
      X_batch = X[batch_indices]
      y_batch = y[batch_indices]
      loss, grad = self.loss(X_batch, y_batch, reg)
      loss_history.append(loss)
      self.W = self.W - learning_rate * grad

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

    return loss_history

  def predict(self, X):
    y_pred = np.zeros(X.shape[0])
    scores = X.dot(self.W)
    y_pred = scores.argmax(axis=1)
    return y_pred
  
  def loss(self, X_batch, y_batch, reg):
    pass


class LinearSVM(LinearClassifier):

  def loss(self, X_batch, y_batch, reg):
      loss = 0.0
      W = self.W
      X = X_batch
      y = y_batch
      
      dW = np.zeros(W.shape) 
      num_classes = W.shape[1]
      num_train = X.shape[0]
      scores = X.dot(W)
      correct_class_scores = scores[ np.arange(num_train), y].reshape(num_train,1)
      margin = np.maximum(0, scores - correct_class_scores + 1)
      margin[ np.arange(num_train), y] = 0 # do not consider correct class in loss
      loss = margin.sum() / num_train
      loss += reg * np.sum(W * W)
      margin[margin > 0] = 1
      valid_margin_count = margin.sum(axis=1)
      margin[np.arange(num_train),y ] -= valid_margin_count
      dW = (X.T).dot(margin) / num_train

      # Regularization gradient
      dW = dW + reg * 2 * W
      return loss, dW



def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = load_pickle(f)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    cifar10_dir = './datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

try:
   del X_train, y_train
   del X_test, y_test
   print('Clear previously loaded data.')
except:
   pass
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print(type(X_train))
num_color_bins = 10 
feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
X_train_feats = extract_features(X_train, feature_fns, verbose=True)
X_val_feats = extract_features(X_val, feature_fns)
X_test_feats = extract_features(X_test, feature_fns)
mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)
X_train_feats -= mean_feat
X_val_feats -= mean_feat
X_test_feats -= mean_feat
std_feat = np.std(X_train_feats, axis=0, keepdims=True)
X_train_feats /= std_feat
X_val_feats /= std_feat
X_test_feats /= std_feat
X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])
X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))])


learning_rates = [1e-3, 1e-2]
regularization_strengths = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
results = {}
best_val = -1
best_svm = None
np.random.seed(0)

grid_search = [ (lr,reg) for lr in learning_rates for reg in regularization_strengths ]

for lr, reg in grid_search:
    svm = LinearSVM()
    svm.train(X_train_feats, y_train, learning_rate=lr, reg=reg, num_iters=2000,
            batch_size=200, verbose=False)
    y_train_pred = svm.predict(X_train_feats)
    train_accuracy = np.mean( y_train_pred == y_train )
    y_val_pred = svm.predict(X_val_feats)
    val_accuracy = np.mean( y_val_pred == y_val )
    results[lr,reg] = (train_accuracy,val_accuracy)
    if val_accuracy > best_val:
        best_val = val_accuracy
        best_svm = svm
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_accuracy, val_accuracy))
    
print('best validation accuracy achieved during cross-validation: %f' % best_val)



y_test_pred = best_svm.predict(X_test_feats)
test_accuracy = np.mean(y_test == y_test_pred)
print(test_accuracy)


examples_per_class = 8
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for cls, cls_name in enumerate(classes):
    idxs = np.where((y_test != cls) & (y_test_pred == cls))[0]
    idxs = np.random.choice(idxs, examples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt.subplot(examples_per_class, len(classes), i * len(classes) + cls + 1)
        plt.imshow(X_test[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls_name)
plt.show()