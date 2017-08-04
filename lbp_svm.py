import numpy as np
from skimage import io
import os
import glob
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from sklearn import svm
from sklearn import metrics

images = []
label =  []
lbp = []
radius = 2
n_points = 24 * radius
eps = 1e-7

for filename in glob.glob(os.path.join('practice/*')):
	aux = io.imread(filename)
	images.append(aux)
	filename = os.path.basename(filename)
	label.append(filename.split("_")[0])
	lbp_item = local_binary_pattern(aux,n_points, radius, 'uniform')
	(hist, _) = np.histogram(lbp_item.ravel(),bins=np.arange(0, n_points + 3),range=(0, n_points + 2))
	# normalize the histogram
	hist = hist.astype("float")
	hist /= (hist.sum() + eps)
	lbp.append(hist)

# print images[0]
model = svm.LinearSVC(C=100.0, random_state=42)
model.fit(lbp,label)

label =  []
lbp = []

for filename in glob.glob(os.path.join('test/*')):
	aux = io.imread(filename)
	images.append(aux)
	filename = os.path.basename(filename)
	label.append(filename.split("_")[0])
	lbp_item = local_binary_pattern(aux, n_points, radius, 'uniform')
	(hist, _) = np.histogram(lbp_item.ravel(),bins=np.arange(0, n_points + 3),range=(0, n_points + 2))
	# normalize the histogram
	hist = hist.astype("float")
	hist /= (hist.sum() + eps)
	lbp.append(hist)

prediction = model.predict(lbp)

print metrics.classification_report(label,prediction)
print metrics.confusion_matrix(label,prediction)
