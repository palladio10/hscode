import os
from os import listdir
from os.path import isfile, join
import pandas as pd

def get_files(folder, suffix, prefix):
    files = []
    if os.path.isdir(folder):
        only_files = [f for f in listdir(folder) if isfile(join(folder, f))]
        for csv_file in only_files:
            if csv_file.endswith(suffix) and csv_file.startswith(prefix):
                files.append(csv_file)
    return files

def filter_sample_firms(receipt_file, firm_file, output_file):
    receipts_df = pd.read_csv(receipt_file, delimiter=',')
    firms_df = pd.read_csv(firm_file, delimiter=',')
    firms_list = firms_df['tin'].tolist()
    filtered_df = receipts_df[receipts_df['tin'].isin(firms_list)]
    filtered_df.to_csv(output_file, sep=',', encoding='utf-8')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f",help="receipt file folder")
    parser.add_argument("-s", help="sample firm file")
    args = parser.parse_args()

    folder = args.f
    files = get_files(folder, ".csv", "processed")
    #print(datafile)
    firm_file = args.s
    directory = "./sample_firm"
    if not os.path.exists(directory):
        os.makedirs(directory)
    for file in files:
        print(file)
        output_file = directory + "/filtered_" + file
        filter_sample_firms(folder + "/" + file, firm_file, output_file)
