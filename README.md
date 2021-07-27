**Project Description**
-

Classify the quality of movements for stroke survivors.
Using reaching / retracting 2D movement data with one wrist sensor (affected side).


**Code Description**
-
Main codes
- load_data.py: load data and put data aligned with the data structure
- plot_raw_acc_vel_pos.py: preprocess data and visualize
- classify.py: extract features and classify with different classifiers
- roc_curve.py: draw roc curve and find an operating point

Function codes
- data_struct.py: class of the data structure
- extract_features.py: the function extracting features
- utils.py: mics. functions