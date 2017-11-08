import os
from os import listdir
from os.path import isfile, join
import pandas as pd
from sample_firm import get_files

def produce_sentences_per_receipt(folder):
    files = get_files(folder, ".csv", "filtered")

    directory = "./sentences"
    if not os.path.exists(directory):
        os.makedirs(directory)
    output_file = directory + "/per_receipt.txt"
    with open(output_file, mode='a') as f:
        f.write("name" + "\n")
        for file in files:
            df = pd.read_csv(folder + "/" + file, decimal=',')
            names = df.groupby('r_id')['name'].apply(list)
            for name_per_receipt in names:
                to_string = [str(i) for i in name_per_receipt]
                f.write(" ".join(to_string) + "\n")

def produce_sentences_per_item(folder):
    files = get_files(folder, ".csv", "filtered")

    directory = "./sentences"
    if not os.path.exists(directory):
        os.makedirs(directory)
    output_file = directory + "/per_item.txt"
    with open(output_file, mode='a') as f:
        for file in files:
            df = pd.read_csv(folder + "/" + file, decimal=',')
            names = df['name'].tolist()
            for name_per_item in names:
                f.write(str(name_per_item) + "\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f",help="sample firm folder")
    args = parser.parse_args()

    folder = args.f
    produce_sentences_per_receipt(folder)



