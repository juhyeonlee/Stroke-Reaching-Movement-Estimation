**Project Description**
-

Classify the quality of movements for stroke survivors.
Using reaching / retracting 2D movement data with one wrist sensor (affected side).


**Code Description**
-
- data_struc.py: class of the data structure
- load_data.py: load data and put data aligned with the data structure
- plot_raw_acc_vel_pos.py: preprocess data and visualize
- extract_features.py: the function extracting features
- classify_Gaussian.py: extract features and classify with Gaussian process classifier
- classify.py: extract features and classify with features selection and SVM
- roc_curve.py: draw roc curve and find an operating point
- utils.py: mics. functions