from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from personalCV.preprocesser_methods import Resizer
from personalCV.data import DataLoader
from imutils import paths
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", required=True,help="path to the dataset")
parser.add_argument("-k", "--neighbors", type=int, default=5,help="No. of neighbors")
parser.add_argument("-j", "--jobs", type=int, default=-1)

arguments = vars(parser.parse_args())

print("[INFO] Loading.....")

path_to_images = list(paths.list_images(arguments["dataset"]))

rs = Resizer(32,32)
dl = DataLoader(preprocesser_methods = [rs])
images,classes = dl.loader(path_to_images,verbose = 100)
images = images.reshape(images[0],3072)

le = LabelEncoder()
classes = le.fit_transform(classes)

X_train,y_train,X_test,y_test = train_test_split(images,classes,test_size = 0.25,random_state = 42)
print("[INFO] Analysing KNN Algorithm")

model = KNeighborsClassifier(n_neighbors = arguments["neighbors"],n_jobs = arguments["jobs"])
model.fit(X_train,y_train)

print(classification_report(y_test, model.predict(X_test),target_names=le.classes_))

