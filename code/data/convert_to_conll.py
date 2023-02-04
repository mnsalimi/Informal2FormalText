import pickle
import random
import numpy as np
from sklearn.model_selection import train_test_split

def convert_all_csvs_to_conll():
    labels = []
    lines = []
    with open("Data - Sheet1.csv", "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open("Data - Sheet2.csv", "r", encoding="utf-8") as f:
        lines += [",,"] + f.readlines()
    with open("Data - Sheet3.csv", "r", encoding="utf-8") as f:
        lines += [",,"] + f.readlines() + [",,"]
    newlines = []
    lines = [
        line.replace("\n", "").split(",")
        for line in lines
        if len(line) and line[0] not in ["<", ">", "<", ">"]
    ]
    for i in range(len(lines)):
        if lines[i][0] != "token":
            newlines.append(lines[i])
    lines = newlines
    end = 0
    data = []
    for i in range(len(lines)):
        if lines[i][0] == "" and lines[i][1] == "":
            data.append(
                [[lines[j][0], lines[j][1]] for j in range(end, i)]
            )
            end = i+1
    for i, line in enumerate(data):
        for j in range(len(data[i])):
            data[i][j][1] = data[i][j][1].replace(" F", "F").replace("M", "N")
            if data[i][j][1] == "":
                if data[i][j][0] != "":
                    print("here", data[i])
            if data[i][j][1] not in labels:
                labels.append(data[i][j][1])
    random.shuffle(data)
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j][1] == "token_tag":
                print(data[i])
    train = data[:790]
    dev = data[790:]
    with open("dev.txt", "w", encoding="utf-8") as f:
        for line in dev:
            for x, y in line:
                f.write(x+"\t"+y+"\n")
            f.write("\n")
    with open("train.txt", "w", encoding="utf-8") as f:
        for line in train:
            for x, y in line:
                f.write(x+"\t"+y+"\n")
            f.write("\n")
    
    return data

def assign_label_based_freq(data):
    freq = {}
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j][0] not in freq:
                freq[data[i][j][0]] = {}

                # try:
                if data[i][j][1] not in freq[data[i][j][0]]:
                    freq[data[i][j][0]] = {data[i][j][1]: 1}
                # except:
                #     print(freq)
                #     print(data[i][j][0])
                #     print(data[i][j][1])
                #     print()
                else:
                    freq[data[i][j][0]][data[i][j][1]] += 1

            else:
                if data[i][j][1] not in freq[data[i][j][0]]:
                    freq[data[i][j][0]] = {data[i][j][1]: 1}
                else:
                    freq[data[i][j][0]][data[i][j][1]] += 1
                # print("else")
                # print(freq)
                # print(data[i][j][0])
                # print(data[i][j][1])
                # freq[data[i][j][0]][data[i][j][1]] += 1
    with open("freq.pickle", "wb") as f:
        pickle.dump(freq, f)
    print(freq)

def get_ambiguity(data):
    ambgs = {}
    for row in data:
        for x, y in row:
            if x not in ambgs:
                ambgs[x] = []
                ambgs[x].append(y)
            else:
                if y not in ambgs[x]:
                    ambgs[x].append(y)
    ambgs_list = [(x, y) for x, y in ambgs.items() if len(y) > 1]
    # print(len(ambgs_list))
    # [print(ambg) for ambg in ambgs_list]

if __name__ == "__main__":
    data = convert_all_csvs_to_conll()
    ambgs = get_ambiguity(data)
    assign_label_based_freq(data)