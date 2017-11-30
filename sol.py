import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as pl
import matplotlib.image as mpimg


training_input_file="data/MSFT19972012.csv"
test_input_file="data/MSFT20122017.csv"
training_output_file="training.csv"
test_output_file="test.csv"

# calculate slopeReference (slopeRef) and find the distribution of the labels
# find the first,second seperation points..
def get_histogram(input_name,roc):
    df = pd.read_csv(input_name, header=None, index_col=None, delimiter=',')
    for r in range(len(df) - 45):
        all_y = df[5].values.tolist()
        price45 = all_y[r + 33]
        price30 = all_y[r + 29]
        dif_ratio = ((price45 - price30) / price30) * 100
        roc.append(dif_ratio)
    roc.sort()
    len_roc=len(roc)
    f_sliding=roc.__getitem__(2*int(len_roc/5))
    s_sliding = roc.__getitem__(3*int(len_roc/5))
    print("len_roc:",len_roc, "f_sliding:",f_sliding,"s_sliding:",s_sliding)
    return round(f_sliding,2),round(s_sliding,2)


# imagesFile
def imagesFileCreation(input_name,output_name,f_sliding,s_sliding):
    # read csv file
    df = pd.read_csv(input_name, header=None, index_col=None, delimiter=',')

    # for all value, create image file
    for r in range(len(df) - 45):
        print('r:', r)
        img = [[]]
        # all pixels are set value 255 (white) in 30x30 pixel image
        img = [[255 for i in range(30)] for i in range(30)]
        all_y = df[5].values.tolist()
        sub_y = all_y[r:r + 30]

        current_price = round(all_y[r+29], 2)

        price45 = all_y[r + 44]
        price30 = all_y[r + 29]

        # calculate ratio
        dif_ratio = ((price45 - price30) / price30) * 100


        #
        max_y = (all_y[r+15] * 130) / 100
        min_y = (all_y[r+15] * 70) / 100

        label_avg_y = 0
        label_sum_y = 0

        print("f_sliding,s_sliding:",f_sliding,s_sliding)
        #test1
        if (dif_ratio >= s_sliding): # slopeCurrent>slopeRef[s_sliding]
            predictLabel = 1   # label=Buy
        elif (dif_ratio > f_sliding and dif_ratio < s_sliding):
            predictLabel = 0   # label = Hold
        elif (dif_ratio <= f_sliding): # slopeCurrent<slopeRef[s_sliding]
            predictLabel = 2   # label = Sell

        print("predictLabel:", predictLabel, " dif_ratio:", dif_ratio, "price:", current_price)
        print("max:",max_y, "min:",min_y)

        # calculate coefficient to normalize data
        coef = 30 / (int(max_y - min_y)+1)
        j = 0
        print(max_y, min_y)

        # calculate the stock price and create black bar graphics for 30 days
        for i in range(30):
            val = (sub_y[i] - min_y) * coef
            #print(val)
            for k in range(int(val)):
                if(k<30):
                    img[29 - k][j] = 0
            j += 1

        my_df = pd.DataFrame(img)
        label_price = ';'.join((str(predictLabel), str(current_price)))

        # append image values in file
        my_df.to_csv(output_name, index=False, header=False, mode='a', line_terminator=';', sep=';')
        with open(output_name, 'a') as file:
            file.write(label_price)
            file.write('\n')
        del label_sum_y, predictLabel

        # print image
        # print(img)
        # imgplot = pl.imshow(img, cmap=pl.get_cmap('gray'))
        # pl.show()

if __name__=="__main__":
    roc = []
    f_sliding,s_sliding=get_histogram(training_input_file,roc)
    print("Starting Training File Generation..")
    imagesFileCreation(training_input_file, training_output_file,f_sliding,s_sliding)
    print("Ending Training File Generation...")
    print("Starting Test File Generation..")
    imagesFileCreation(test_input_file, test_output_file,f_sliding,s_sliding)
    print("Ending Test File Generation...")






