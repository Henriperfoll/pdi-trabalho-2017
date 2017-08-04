import numpy as np
from skimage import io
import os
import glob
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

classes =  []
features = []

radius = 2
n_points = 10 * radius
eps = 1e-7

for filename in glob.glob(os.path.join('practice/*')):
	aux = io.imread(filename)
	filename = os.path.basename(filename)
	classes.append(filename.split("_")[0])
	lbp_item = local_binary_pattern(aux,n_points, radius, 'uniform')
	(hist, _) = np.histogram(lbp_item.ravel(),bins=np.arange(0, n_points + 3),range=(0, n_points + 2))
	# normalize the histogram
	hist = hist.astype("float")
	hist /= (hist.sum() + eps)
	features.append(hist)

# print images[0]
model = KNeighborsClassifier(n_neighbors=3)
model.fit(features,classes)

classes =  []
features = []

for filename in glob.glob(os.path.join('test/*')):
	aux = io.imread(filename)
	filename = os.path.basename(filename)
	classes.append(filename.split("_")[0])
	lbp_item = local_binary_pattern(aux, n_points, radius, 'uniform')
	(hist, _) = np.histogram(lbp_item.ravel(),bins=np.arange(0, n_points + 3),range=(0, n_points + 2))
	# normalize the histogram
	hist = hist.astype("float")
	hist /= (hist.sum() + eps)
	features.append(hist)

prediction = model.predict(features)
# print prediction.shape
# print prediction
print metrics.classification_report(classes,prediction)
print metrics.confusion_matrix(classes,prediction)
