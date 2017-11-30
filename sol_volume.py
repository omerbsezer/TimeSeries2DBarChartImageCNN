import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as pl
import matplotlib.image as mpimg


training_input_file="BA19972007.csv"
test_input_file="BA20072017.csv"
training_output_file="training.csv"
test_output_file="test.csv"

# use volume values when creating stock price bar images
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
    f_sliding=roc.__getitem__(3*int(len_roc/7))
    s_sliding = roc.__getitem__(4*int(len_roc/7))
    print("len_roc:",len_roc, "f_sliding:",f_sliding,"s_sliding:",s_sliding)
    return round(f_sliding,2),round(s_sliding,2)

def file_generation(input_name,output_name,f_sliding,s_sliding):
    # read csv file
    df = pd.read_csv(input_name, header=None, index_col=None, delimiter=',')

    # for all value, create image file
    for r in range(len(df) - 45):
        print('r:', r)
        img = [[]]

        img = [[255 for i in range(30)] for i in range(30)]
        all_y = df[5].values.tolist()
        all_volume=df[6].values.tolist()
        color=df[6].values.tolist()
        values = df[6].values.tolist()
        all_volume_mean=(df[6].mean(),2)
        all_volume_min=df[6].min()
        all_volume_max = df[6].max()
        sub_volume=all_volume[r:r+30]
        print("all_volume:",all_volume)
        print("all_volume_min:", all_volume_min)
        print("all_volume_max", all_volume_max)
        sub_y = all_y[r:r + 30]

        current_price = round(all_y[r+29], 2)
        price33 = all_y[r + 44]
        price30 = all_y[r + 29]

        dif_ratio = ((price33 - price30) / price30) * 100
        max_volume=0
        min_volume=1000000000
        for i in range(30):
            if(sub_volume[i]>max_volume):
                max_volume=sub_volume[i]
            if (sub_volume[i] < min_volume):
                min_volume = sub_volume[i]

        for i in range(30):
            volume_coef=((sub_volume[i]-min_volume)/(max_volume-min_volume))*100
            color[i]=volume_coef

        max_y = (all_y[r+15] * 130) / 100
        min_y = (all_y[r+15] * 70) / 100

        label_avg_y = 0
        label_sum_y = 0

        if (dif_ratio >= s_sliding):
            predictLabel = 1
        elif (dif_ratio > f_sliding and dif_ratio < s_sliding):
            predictLabel = 0
        elif (dif_ratio <= f_sliding):
            predictLabel = 2

        print("predictLabel:", predictLabel, " dif_ratio:", dif_ratio, "price:", current_price)

        coef = 30 / int(max_y - min_y)
        j = 0
        print(max_y, min_y)
        #
        for i in range(30):
            values[i] = (sub_y[i] - min_y) * coef
            #print("values[",i,"]:",values[i])
            for k in range(int(values[i])):
                if (k < 30):
                    img[29 - k][i] = 200-(int(color[i])*2)


        print("values:", int(values[r]))
        my_df = pd.DataFrame(img)
        label_price = ';'.join((str(predictLabel), str(current_price)))

        my_df.to_csv(output_name, index=False, header=False, mode='a', line_terminator=';', sep=';')
        with open(output_name, 'a') as file:
            file.write(label_price)
            file.write('\n')
        del label_sum_y, predictLabel
        # print(img)
        #imgplot = pl.imshow(img, cmap=pl.get_cmap('gray'),vmin=0,vmax=255)
        #pl.show()

if __name__=="__main__":
    roc = []
    f_sliding, s_sliding = get_histogram(training_input_file, roc)
    print("Starting Training File Generation..")
    file_generation(training_input_file, training_output_file, f_sliding, s_sliding)
    print("Ending Training File Generation...")
    print("Starting Test File Generation..")
    file_generation(test_input_file, test_output_file, f_sliding, s_sliding)
    print("Ending Test File Generation...")




