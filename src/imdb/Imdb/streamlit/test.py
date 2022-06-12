import os

#The architecture of a project in Django

#root/

#app1/
#app2/
#...
#main/
#settings.py
#Inside settings.py:

#SITE_ROOT = os.path.dirname(os.path.realpath(__file__)) -> gives the path of the file settings.py: root/main/. This is NOT THE ROOT OF THE PROJECT

#PROJECT_PATH = os.path.abspath(os.path.dirname(__name__)) -> gives the root of the project: root/. This is THE ROOT OF THE PROJECT.

#PROJECT_PATH = os.path.abspath(os.path.dirname(__name__))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
BASE_DIR = "/".join(BASE_DIR.split('/')[:-3])

PATCH_DIR = os.path.join(BASE_DIR, "ML", "patch.py")
KNN_DIR = os.path.join(BASE_DIR, "ML", "knn_predict.py")
print(KNN_DIR)

import sys

sys.path.append(KNN_DIR)

from importlib.machinery import SourceFileLoader
  
# imports the module from the given path
foo = SourceFileLoader("knn_predict", PATCH_DIR).load_module()
foo = SourceFileLoader("knn_predict", KNN_DIR).load_module()