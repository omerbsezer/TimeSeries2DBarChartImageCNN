# TimeSeries2DBarChartImageCNN: Financial Trading Model with Stock Bar Chart Image Time Series with Deep Convolutional Neural Networks

## CNN-BI
"In this study, we developed an out-of-the-box algorithmic trading strategy which was based on identifying Buy-Sell decisions based on triggers that are generated from a trained deep CNN model using stock bar chart images. Almost all existing strategies in literature used the stock time series data directly or indirectly, however in this study we chose a different path by using the chart images directly as 2-D images without introducing any other time series data. To best of our knowledge, this is the first attempt in literature to adapt such an unconventional approach. The results indicate that the proposed model was able to produce generally consistent outcomes and was able to beat BaH strategy depending on the market conditions."

## Method
![image](https://user-images.githubusercontent.com/10358317/211334577-699c7bc8-7f63-415f-9e9a-cff3b55176dd.png)

## Generated Images
![image](https://user-images.githubusercontent.com/10358317/211334749-7bffca30-5ab9-4c2f-a0d6-fc121a9ae47c.png)

## Sample Result:
![image](https://user-images.githubusercontent.com/10358317/211334963-641951de-545c-43e6-a447-9e05115cbd37.png)


## Phases in the algorithm:

Conversion of the time series values to 2-D stock bar chart images and prediction using CNN. Numpy, Pandas, Tensorflow and Keras libraries are used to implement algorithm.

1. Download stock historical data from "finance.yahoo.com" (there is example files in \data directory).
2. Run sol.py to create training and test image files.
3. Run cnn.py with training.csv and test.csv, and create cnn_result file with prediction, price and ground-truth data.
4. Evalute with evaluateFinancial.java


**Paper Link:** https://arxiv.org/pdf/1903.04610.pdf

**Note:** If you are using code or idea from paper, please refer to this paper. 

**Bibtex:**

```
@article{sezer2019financial,
  title={Financial trading model with stock bar chart image time series with deep convolutional neural networks},
  author={Sezer, Omer Berat and Ozbayoglu, Ahmet Murat},
  journal={arXiv preprint arXiv:1903.04610},
  year={2019}
}
```
