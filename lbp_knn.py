import numpy as np
from skimage import io
import os
import glob
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

def extract_lbp_features(image):
	#raio do lbp
	radius = 2
	#quantidade de pontos no raio
	n_points = 10 * radius
	lbp = local_binary_pattern(image,n_points, radius, 'uniform')
	(hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, n_points + 3),range=(0, n_points + 2))
	# normalize the histogram
	eps = 1e-7
	hist = hist.astype("float")
	hist /= (hist.sum() + eps)
	return hist

classes =  []
features = []

for filename in glob.glob(os.path.join('../practice/*')):
	aux = io.imread(filename)
	filename = os.path.basename(filename)
	classes.append(filename.split("_")[0])
	features.append(extract_lbp_features(aux))

# print images[0]
model = KNeighborsClassifier(n_neighbors=3)
model.fit(features,classes)

classes =  []
features = []

for filename in glob.glob(os.path.join('../test/*')):
	aux = io.imread(filename)
	filename = os.path.basename(filename)
	classes.append(filename.split("_")[0])
	features.append(extract_lbp_features(aux))

prediction = model.predict(features)

print metrics.classification_report(classes,prediction)
print metrics.confusion_matrix(classes,prediction)
