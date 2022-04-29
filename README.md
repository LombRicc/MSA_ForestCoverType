# MSA Forest Cover Type Classification with AdaBooost

This git repository contains the project done for the Statistical Methods for Machine Learning Course of Università di Milano. It's an evaluation of a made from scratch implementation of the AdaBoost algorithm on a real-life dataset of observations of four wilderness areas located in the Roosevelt National Forest of northern Colorado. Each observation is a 30m x 30m patch and define seven different types of forest cover. 
The seven types are:
1) Spruce/Fir;
2) Lodgepole Pine;
3) Ponderosa Pine;
4) Cottonwood/Willow;
5) Aspen;
6) Douglas-fir;
7) Krummholz.

The observations are defined by 54 features:
* Elevation - Elevation in meters;
* Aspect - Aspect in degrees azimuth;
* Horizontal Distance To Hydrology - Hori-
zontal Distance to nearest surface water fea-
tures;
* Vertical Distance To Hydrology - Vertical
Distance to nearest surface water features;
* Horizontal Distance To Roadways - Hori-
zontal Distance to nearest roadway;
* Hillshade 9am (0 to 255 index) - Hillshade
index at 9am, summer solstice;
* Hillshade Noon (0 to 255 index) - Hillshade
index at noon, summer solstice;
* Hillshade 3pm (0 to 255 index) - Hillshade
index at 3pm, summer solstice;
* Horizontal Distance To Fire Points - Hor-
izontal Distance to nearest wildfire ignition
points;
* Wilderness Area (4 binary columns, 0 = ab-
sence or 1 = presence) - Wilderness area des-
ignation;
* Soil Type (40 binary columns, 0 = absence or
1 = presence) - Soil Type designation;
* Cover Type (7 types, integers 1 to 7) - Forest
Cover Type designation.

The training set contains both features and the
Cover Type. The test set contains only the features.

Class repository contains three files: 

* **covtype.csv** contains the Forest Cover Type dataset composed of 581012 observations with 54 attributes that define 7 different classes;   

* **adaboost.py** contains the full implementation of the Adaboost algorithm from scratch in python. The implementation of the decision stump, the fit method that trains the decision stumps accordingly to the number of boosting rounds, as well as the predict method which is used to make the class predictions for the 7 classes;

* **test.py** is the file to test the algorithm, it's possible to change the parameteres such as K the number of folds for the external cross-validation and T which is the number of boosting rounds to use in the algorithm. It also shows the results in terms of classification accuracy.
