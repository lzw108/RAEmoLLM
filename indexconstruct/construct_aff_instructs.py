import pandas as pd
import argparse

# 1. Task EI-reg: Detecting Emotion Intensity (regression)
def construct_EIreg(x, y):
    pre_instruct = "Calculate the intensity of emotion E in the tweet as a decimal value ranging from 0 to 1, with 0 representing the lowest intensity and 1 representing the highest intensity."
    if x[-1] == "." or x[-1] == "!" or x[-1] == "?":
        Instruction = "Task: " + pre_instruct + " Tweet: " + x + " Emotion E: " + y + ". " + "Intensity Score: "
    else:
        Instruction = "Task: " + pre_instruct + " Tweet: " + x + ". Emotion E: " + y + ". " + "Intensity Score: "
    return Instruction.replace("\n", " ")

def EIreg_instructioins(data_name, data_path,output_path):
    df = pd.read_csv(data_path)
    dfs = []
    emotions = ["anger", "fear", "joy", "sadness"]
    for epoch in emotions:
        print(len(df["text"]))
        df['Instruction'] = df.apply(lambda row: construct_EIreg(row['text'], epoch), axis=1)
        df['Answer'] = 9999
        df[['Instruction', 'Answer']].to_csv(output_path + './%s-EIreg-%s.csv' % (data_name,epoch), index=False)
        dfs.append(output_path + './%s-EIreg-%s.csv' % (data_name,epoch))

    df_all = pd.read_csv(dfs[0])
    for i in range(1, len(dfs)):
        df_new = pd.read_csv(dfs[i])
        df_all = pd.concat([df_all, df_new])
    print(len(df_all["Instruction"]))
    df_all.reset_index(drop=True)
    df_all.to_csv(output_path + './%s-EIreg.csv'%data_name, index=False)


# 2. Task EI-oc: Detecting Emotion Intensity (ordinal classification)
def construct_EIoc(x, y):
    pre_instruct = "Classify the tweet into one of four ordinal classes of intensity of emotion E that best represents the mental state of the tweeter. 0: no E can be inferred. 1: low amount of E can be inferred. 2: moderate amount of E can be inferred. 3: high amount of E can be inferred."
    if x[-1] == "." or x[-1] == "!" or x[-1] == "?":
        Instruction = "Task: " + pre_instruct + " Tweet: " + x + " Emotion E: " + y + ". " + "Intensity Class: "
    else:
        Instruction = "Task: " + pre_instruct + " Tweet: " + x + ". Emotion E: " + y + ". " + "Intensity Class: "
    return Instruction.replace("\n", " ")

def EIoc_instructioins(data_name, data_path,output_path):
    df = pd.read_csv(data_path)
    dfs = []
    emotions = ["anger", "fear", "joy", "sadness"]
    for epoch in emotions:
        print(len(df["text"]))
        df['Instruction'] = df.apply(lambda row: construct_EIoc(row['text'], epoch), axis=1)
        df['Answer'] = 9999

        df[['Instruction', 'Answer']].to_csv(output_path + './%s-EIoc-%s.csv' % (data_name, epoch), index=False)
        dfs.append(output_path + './%s-EIoc-%s.csv' % (data_name, epoch))

    df_all = pd.read_csv(dfs[0])
    for i in range(1, len(dfs)):
        df_new = pd.read_csv(dfs[i])
        df_all = pd.concat([df_all, df_new])
    print(len(df_all["Instruction"]))
    df_all.reset_index(drop=True)
    df_all.to_csv(output_path + './%s-EIoc.csv'%data_name, index=False)

# 3. Task V-reg: Detecting Valence or Sentiment Intensity (regression)
def construct_Vreg(x):
    pre_instruct = "Calculate the sentiment intensity or valence score of the tweet, which should be a real number between 0 (extremely negative) and 1 (extremely positive)."
    if x[-1] == "." or x[-1] == "!" or x[-1] == "?":
        Instruction = "Task: " + pre_instruct + " Tweet: " + x + " " + "Intensity Score: "
    else:
        Instruction = "Task: " + pre_instruct + " Tweet: " + x + ". " + "Intensity Score: "
    return Instruction.replace("\n"," ")

def Vreg_instructioins(data_name, data_path,output_path):
    df = pd.read_csv(data_path)
    df['Instruction'] = df.apply(lambda row: construct_Vreg(row['text']), axis=1)
    df['Answer'] = 9999
    df[['Instruction','Answer']].to_csv(output_path + './%s-Vreg.csv'%data_name, index=False)


# 4 Task V-oc: Detecting Valence (ordinal classification) -- This is the traditional Sentiment Analysis Task
def construct_Voc(x):
    pre_instruct = "Classify the tweet into one of seven ordinal classes, corresponding to various levels of positive and negative sentiment intensity, that best represents the mental state of the tweeter. 3: very positive mental state can be inferred. 2: moderately positive mental state can be inferred. 1: slightly positive mental state can be inferred. 0: neutral or mixed mental state can be inferred. -1: slightly negative mental state can be inferred. -2: moderately negative mental state can be inferred. -3: very negative mental state can be inferred."
    if x[-1] == "." or x[-1] == "!" or x[-1] == "?":
        Instruction = "Task: " + pre_instruct + " Tweet: " + x + " " + "Intensity Class: "
    else:
        Instruction = "Task: " + pre_instruct + " Tweet: " + x + ". " + "Intensity Class: "
    return Instruction.replace("\n"," ")

def Vroc_instructioins(data_name, data_path,output_path):
    df = pd.read_csv(data_path)
    print(len(df["text"]))
    df['Instruction'] = df.apply(lambda row: construct_Voc(row['text']), axis=1)
    df['Answer'] = 9999
    df[['Instruction','Answer']].to_csv(output_path + './%s-Voc.csv'%data_name, index=False)


# 5. Task E-c: Detecting Emotions (multi-label classification) -- This is a traditional Emotion Classification Task
def construct_Ec(x):
    pre_instruct = "Categorize the tweet's emotional tone as either 'neutral or no emotion' or identify the presence of one or more of the given emotions (anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, trust)."
    if x[-1] == "." or x[-1] == "!" or x[-1] == "?":
        Instruction = "Task: " + pre_instruct + " Tweet: " + x + " " + "This tweet contains emotions: "
    else:
        Instruction = "Task: " + pre_instruct + " Tweet: " + x + ". " + "This tweet contains emotions: "
    return Instruction.replace("\n", " ")

def Ec_instructioins(data_name, data_path, output_path):
    df = pd.read_csv(data_path)
    print(len(df["text"]))
    df['Instruction'] = df.apply(lambda row: construct_Ec(row['text']), axis=1)
    df['Answer'] = 9999
    df[['Instruction', 'Answer']].to_csv(output_path + './%s-Ec.csv'%data_name, index=False)

def combine_aff_instructs(data_name, data_path, output_path):

    df1 = pd.read_csv(output_path + "%s-EIreg.csv"%data_name)
    df2 = pd.read_csv(output_path + "%s-EIoc.csv"%data_name)
    df3 = pd.read_csv(output_path + "%s-Vreg.csv"%data_name)
    df4 = pd.read_csv(output_path + "%s-Voc.csv"%data_name)
    df5 = pd.read_csv(output_path + "%s-Ec.csv"%data_name)

    print(len(df1),len(df2),len(df3),len(df4),len(df5))
    merged_df = pd.concat([df1, df2, df3, df4, df5])
    merged_df.to_csv(output_path + '%s-affective-all.csv'%data_name, index=False)

def csv2json(input_file, output_file):
    # Get json format
    df = pd.read_csv(input_file)
    import json
    with open(output_file, 'w', encoding="utf-8") as write_f:
        for index, row in df.iterrows():
            data_one = {}

            data_one["instruction"] = row["Instruction"]
            data_one["input"] = ""
            data_one["output"] = row["Answer"]
            write_f.write(json.dumps(data_one, indent=None, ensure_ascii=False) + "\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    print("Begin to add affective instructs for data......")

    EIreg_instructioins(args.data_name, args.data_path, args.output_path)
    EIoc_instructioins(args.data_name, args.data_path, args.output_path)

    Vreg_instructioins(args.data_name, args.data_path, args.output_path)
    Vroc_instructioins(args.data_name, args.data_path, args.output_path)

    Ec_instructioins(args.data_name, args.data_path, args.output_path)

    combine_aff_instructs(args.data_name, args.data_path, args.output_path)
    input_file = args.output_path + '%s-affective-all.csv'%args.data_name
    output_file = args.output_path + '%s-affective-all.json'%args.data_name
    csv2json(input_file, output_file)

