import os
from os import listdir
from os.path import isfile, join
import pandas as pd

def get_files(folder, file_type):
    files = []
    if os.path.isdir(folder):
        only_files = [f for f in listdir(folder) if isfile(join(folder, f))]
        for file in only_files:
            if file.endswith(file_type):
                files.append(file)
    return files

def filter_sample_firms(receipt_file, firm_file):
    receipts = pd.read_csv(receipt_file, delimiter=',')
    firms = pd.read_csv(firm_file, delimiter=',')
    firms_list = firms['tin'].tolist()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f",help="receipt file")
    parser.add_argument("-s", help="sample firm file")
    args = parser.parse_args()

    datafile = args.f
    #print(datafile)
    directory = "./sample_firm"
    index = datafile.rfind("/")
    filename = datafile[index + 1:]
    output_file = directory + "/processed_" + filename

    if not os.path.exists(directory):
        os.makedirs(directory)
