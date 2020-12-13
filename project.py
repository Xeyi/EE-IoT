from time import time
from PIL import Image
import glob
import numpy as np
import sys
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

PICTURE_PATH = r"C:\Users\Yi\Desktop\Assignment\695-Applied Machine Learning\project\data\jm"

all_data_set = []  
all_data_label = [] 


def get_data():
    label = 1
    for name in glob.glob(PICTURE_PATH + "\\*.pgm"):
        img = Image.open(name)
        all_data_set.append(list(img.getdata()))
        all_data_label.append(label)
        label += 1
    train_data,test_data = train_test_split(all_data_set, test_size = 0.2)
    train_label,test_label = train_test_split(all_data_label, test_size = 0.2)
    return train_data,test_data,train_label,test_label

train_data,test_data,train_label,test_label=get_data()

n_components = 8 
pca = PCA(n_components=n_components, svd_solver='auto',
          whiten=True).fit(all_data_set)
n_components = 8 
pca = PCA(n_components=n_components, svd_solver='auto',
          whiten=True).fit(all_data_set)
# PCA降维后的总数据集
all_data_pca = pca.transform(all_data_set)
train_data_pca = pca.transform(train_data)
test_data_pca = pca.transform(test_data)
# X为降维后的数据，y是对应类标签
X = np.array(all_data_pca)
X_train = np.array(train_data_pca)
X_test = np.array(test_data_pca)
y = np.array(all_data_label)
y_train = np.array(train_label)
y_test = np.array(test_label)

clf = tree.DecisionTreeClassifier()
clf.fit(X, y)

# Cross Validation
cv = KFold(n_splits=10)            
misclassification_rate = 999
tree_size = 0
max_attributes = 10
depth_range = range(1, max_attributes + 1)

for depth in depth_range:
    tree_model = tree.DecisionTreeClassifier(max_depth = depth)
    model = tree_model.fit(X, y) 
    valid_acc = model.score(X_test, y_test)
    if(misclassification_rate > valid_acc):
        misclassification_rate = valid_acc
        tree_size = depth
        
final_tree = tree.DecisionTreeClassifier(max_depth = tree_size)
final_tree.fit(X, y)

def get_pic():
    data = []
    for i in glob.glob(r"C:\Users\Yi\Desktop\Assignment\695-Applied Machine Learning\project\pic\11.pgm"):
        img = Image.open(i)
        data.append(img.getdata())
    data_pca = pca.transform(data)
    X_pic = np.array(data_pca)
    return X_pic

X_pic = get_pic()
print(clf.predict(X_pic))
