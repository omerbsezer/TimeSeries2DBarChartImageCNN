# TimeSeries2DBarChartImageCNN
Conversion of the time series values to 2-D stock bar chart images and prediction using CNN.

1. download stock historical data from "finance.yahoo.com" (there is example files in \data directory).
2. run sol.py to create training and test image files.
3. run cnn.py with training.csv and test.csv, and create cnn_result file with prediction, price and ground-truth data.
4. evalute with evaluateFinancial.java
