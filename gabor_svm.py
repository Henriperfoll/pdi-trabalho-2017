import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel

import glob
from skimage import io
import os

from sklearn import svm
from sklearn import metrics

def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
    return filtered

# prepare filter bank kernels
kernels = []
for theta in range(10):
    theta = theta / 10. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)

classes = []
features = []

for filename in glob.glob(os.path.join('practice/*')):
	aux = io.imread(filename)
	filename = os.path.basename(filename)
	classes.append(filename.split("_")[0])
	gabor_item = compute_feats(aux, kernels)
	(hist, _) = np.histogram(gabor_item.ravel(),bins=np.arange(0,11),range=(0,10))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
    	features.append(hist)


model = svm.LinearSVC(C=100.0, random_state=42)
model.fit(features,classes)

classes = []
features = []

for filename in glob.glob(os.path.join('practice/*')):
	aux = io.imread(filename)
	filename = os.path.basename(filename)
	classes.append(filename.split("_")[0])
	gabor_item = compute_feats(aux, kernels)
	(hist, _) = np.histogram(gabor_item.ravel(),bins=np.arange(0,11),range=(0,10))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
    	features.append(hist)

prediction = model.predict(features)

print metrics.classification_report(classes,prediction)
print metrics.confusion_matrix(classes,prediction)
