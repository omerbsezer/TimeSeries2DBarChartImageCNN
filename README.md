# TimeSeries2DBarChartImageCNN

"In this study, we developed an out-of-the-box algorithmic trading strategy which was based on identifying Buy-Sell decisions based on triggers that are generated from a trained deep CNN model using stock bar chart images. Almost all existing strategies in literature used the stock time series data directly or indirectly, however in this study we chose a different path by using the chart images directly as 2-D images without introducing any other time series data. To best of our knowledge, this is the first attempt in literature to adapt such an unconventional approach. The results indicate that the proposed model was able to produce generally consistent outcomes and was able to beat BaH strategy depending on the market conditions."

Conversion of the time series values to 2-D stock bar chart images and prediction using CNN. Numpy, Pandas, Tensorflow and Keras libraries are used to implement algorithm.

1. Download stock historical data from "finance.yahoo.com" (there is example files in \data directory).
2. Run sol.py to create training and test image files.
3. Run cnn.py with training.csv and test.csv, and create cnn_result file with prediction, price and ground-truth data.
4. Evalute with evaluateFinancial.java

Details will be announced later..
