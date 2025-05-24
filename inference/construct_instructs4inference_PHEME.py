import torch
import json
import os
import pandas as pd
import argparse

def csv2json(input_file, output_file):
    df = pd.read_csv(input_file)
    with open(output_file, 'w',encoding="utf-8") as write_f:
        for index, row in df.iterrows():
            data_one = {}
            data_one["instruction"] = row["Instruction"]
            data_one["input"] = ""
            data_one["output"] = row["Answer"]
            write_f.write(json.dumps(data_one, indent=None, ensure_ascii=False) + "\n")

def construct_instructs_VregVocEc(dim_name, num_examples, aff_name, data_name, Task_prompt, input_path, output_path):
    sydneysiege = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/sydneysiege-Retribasedon-%s.pth" % (data_name, aff_name, dim_name, aff_name))
    ottawashooting = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/ottawashooting-Retribasedon-%s.pth" % (data_name, aff_name, dim_name, aff_name))
    charliehebdo = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/charliehebdo-Retribasedon-%s.pth" % (data_name, aff_name, dim_name, aff_name))
    ferguson = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/ferguson-Retribasedon-%s.pth" % (data_name, aff_name, dim_name, aff_name))
    germanwings = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/germanwings-Retribasedon-%s.pth" % (data_name, aff_name, dim_name, aff_name))
    prince = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/prince-Retribasedon-%s.pth" % (data_name, aff_name, dim_name, aff_name))
    putinmissing = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/putinmissing-Retribasedon-%s.pth" % (data_name, aff_name, dim_name, aff_name))
    gurlitt = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/gurlitt-Retribasedon-%s.pth" % (data_name, aff_name, dim_name, aff_name))
    ebola = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/ebola-Retribasedon-%s.pth" % (data_name, aff_name, dim_name, aff_name))

    test_datas = []
    test_datas.extend(sydneysiege)
    test_datas.extend(ottawashooting)
    test_datas.extend(charliehebdo)
    test_datas.extend(ferguson)
    test_datas.extend(germanwings)
    test_datas.extend(prince)
    test_datas.extend(putinmissing)
    test_datas.extend(gurlitt)
    test_datas.extend(ebola)
    len(test_datas)

    os.makedirs(os.path.join(output_path, "data4inference", data_name, aff_name, dim_name), exist_ok=True)

    with open(output_path + "./data4inference/%s/%s/%s/test.json"%(data_name, aff_name,dim_name), 'w', encoding="utf-8") as write_f:
        for index, test_data in enumerate(test_datas):
            data_one = {}

            Examples = "\nHere are a few examples:\n"
            for data_ in test_data["retried_data"][:num_examples]:
                Examples += " Text: " + data_["content"] + " The label of this text: " + data_["label"] + ".\n"
            Instruction = Task_prompt + "Target text: " + test_data[
                "content"] + Examples + " \nAccording to the above information, the label of target text:"

            data_one["Domain"] = test_data["domain"]
            data_one["Instruction"] = Instruction
            data_one["Answer"] = test_data["label"]
            write_f.write(json.dumps(data_one, indent=None, ensure_ascii=False) + "\n")
    metadata_test = pd.read_json(output_path + "./data4inference/%s/%s/%s/test.json"%(data_name, aff_name,dim_name), lines=True, dtype={'num': str})
    metadata_test.to_csv(output_path + "./data4inference/%s/%s/%s/test.csv"%(data_name, aff_name,dim_name), index=False)

    # Get json format
    input_file = output_path + "./data4inference/%s/%s/%s/test.csv"%(data_name, aff_name,dim_name)
    output_file = output_path + "./data4inference/%s/%s/%s/%s-%s-%s.json"%(data_name, aff_name, dim_name,data_name, aff_name, dim_name)
    csv2json(input_file, output_file)
    if not os.path.exists(output_path + "./data4inference/%s/all/"%(data_name)):
        os.mkdir(output_path + "./data4inference/%s/all/"%(data_name))
    path = output_file
    to_path = output_path + "./data4inference/%s/all/"%(data_name)
    os.system("cp %s %s"%(path,to_path))

def construct_instructs_VregVocEc_addexpl(dim_name, num_examples, aff_name, data_name, Task_prompt, input_path, output_path):
    sydneysiege = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/sydneysiege-Retribasedon-%s.pth" % (data_name, aff_name, dim_name, aff_name))
    ottawashooting = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/ottawashooting-Retribasedon-%s.pth" % (data_name, aff_name, dim_name, aff_name))
    charliehebdo = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/charliehebdo-Retribasedon-%s.pth" % (data_name, aff_name, dim_name, aff_name))
    ferguson = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/ferguson-Retribasedon-%s.pth" % (data_name, aff_name, dim_name, aff_name))
    germanwings = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/germanwings-Retribasedon-%s.pth" % (data_name, aff_name, dim_name, aff_name))
    prince = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/prince-Retribasedon-%s.pth" % (data_name, aff_name, dim_name, aff_name))
    putinmissing = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/putinmissing-Retribasedon-%s.pth" % (data_name, aff_name, dim_name, aff_name))
    gurlitt = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/gurlitt-Retribasedon-%s.pth" % (data_name, aff_name, dim_name, aff_name))
    ebola = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/ebola-Retribasedon-%s.pth" % (data_name, aff_name, dim_name, aff_name))

    test_datas = []
    test_datas.extend(sydneysiege)
    test_datas.extend(ottawashooting)
    test_datas.extend(charliehebdo)
    test_datas.extend(ferguson)
    test_datas.extend(germanwings)
    test_datas.extend(prince)
    test_datas.extend(putinmissing)
    test_datas.extend(gurlitt)
    test_datas.extend(ebola)
    len(test_datas)

    os.makedirs(os.path.join(output_path, "data4inference", data_name, aff_name, dim_name), exist_ok=True)

    with open(output_path + "./data4inference/%s/%s/%s/test.json"%(data_name, aff_name,dim_name), 'w', encoding="utf-8") as write_f:
        for index, test_data in enumerate(test_datas):
            data_one = {}
            if aff_name == "Vreg":
                Examples = ".\nHere are a few examples retrieved through sentiment intensity:\n"
                for data_ in test_data["retried_data"][:num_examples]:
                    Examples += " Text: " + data_["content"] + " Sentiment intensity: " + data_["aff_info"] + ". The label of this text: " + data_["label"] + ".\n"
                Instruction = Task_prompt + "Target text: " + test_data["content"]  + " Sentiment intensity: " + test_data["aff_info"] + Examples + " \nAccording to the above information, the label of target text:"
            elif aff_name == "Voc":
                Examples = ".\nHere are a few examples retrieved through sentiment classification:\n"
                for data_ in test_data["retried_data"][:num_examples]:
                    Examples += " Text: " + data_["content"] + " Sentiment classification: " + data_[
                        "aff_info"] + ". The label of this text: " + data_["label"] + ".\n"
                Instruction = Task_prompt + "Target text: " + test_data["content"] + " Sentiment classification: " + \
                              test_data[
                                  "aff_info"] + Examples + " \nAccording to the above information, the label of target text:"
            elif aff_name == "Ec":
                Examples = "\nHere are a few examples retrieved through emotion classification:\n"
                for data_ in test_data["retried_data"][:num_examples]:
                    Examples += " Text: " + data_["content"] + " Emotion classification: " + data_[
                        "aff_info"] + " The label of this text: " + data_["label"] + ".\n"
                Instruction = Task_prompt + "Target text: " + test_data["content"] + " Emotion classification: " + \
                              test_data["aff_info"] + Examples + " \nAccording to the above information, the label of target text:"
            else:
                print("affective name err....")

            data_one["Domain"] = test_data["domain"]
            data_one["Instruction"] = Instruction
            data_one["Answer"] = test_data["label"]
            write_f.write(json.dumps(data_one, indent=None, ensure_ascii=False) + "\n")
    metadata_test = pd.read_json(output_path + "./data4inference/%s/%s/%s/test.json"%(data_name, aff_name,dim_name), lines=True, dtype={'num': str})
    metadata_test.to_csv(output_path + "./data4inference/%s/%s/%s/test.csv"%(data_name, aff_name,dim_name), index=False)

    # Get json format
    input_file = output_path + "./data4inference/%s/%s/%s/test.csv"%(data_name, aff_name,dim_name)
    output_file = output_path + "./data4inference/%s/%s/%s/%s-%s-%s-addexpl.json"%(data_name, aff_name, dim_name,data_name, aff_name, dim_name)
    csv2json(input_file, output_file)
    if not os.path.exists(output_path + "./data4inference/%s/all/"%(data_name)):
        os.mkdir(output_path + "./data4inference/%s/all/"%(data_name))
    path = output_file
    to_path = output_path + "./data4inference/%s/all/"%(data_name)
    os.system("cp %s %s"%(path,to_path))

def construct_instructs_EIregEIoc(dim_name, num_examples, aff_name, data_name, Task_prompt, input_path, output_path):
    aff_name, emotion_name = aff_name.split("/")

    sydneysiege = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/sydneysiege-Retribasedon-%s.pth" % (data_name, aff_name+"/"+emotion_name, dim_name, aff_name+"-"+emotion_name))
    ottawashooting = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/ottawashooting-Retribasedon-%s.pth" % (data_name, aff_name+"/"+emotion_name, dim_name, aff_name+"-"+emotion_name))
    charliehebdo = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/charliehebdo-Retribasedon-%s.pth" % (data_name, aff_name+"/"+emotion_name, dim_name, aff_name+"-"+emotion_name))
    ferguson = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/ferguson-Retribasedon-%s.pth" % (data_name, aff_name+"/"+emotion_name, dim_name, aff_name+"-"+emotion_name))
    germanwings = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/germanwings-Retribasedon-%s.pth" % (data_name, aff_name+"/"+emotion_name, dim_name, aff_name+"-"+emotion_name))
    prince = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/prince-Retribasedon-%s.pth" % (data_name, aff_name+"/"+emotion_name, dim_name, aff_name+"-"+emotion_name))
    putinmissing = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/putinmissing-Retribasedon-%s.pth" % (data_name, aff_name+"/"+emotion_name, dim_name, aff_name+"-"+emotion_name))
    gurlitt = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/gurlitt-Retribasedon-%s.pth" % (data_name, aff_name+"/"+emotion_name, dim_name, aff_name+"-"+emotion_name))
    ebola = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/ebola-Retribasedon-%s.pth" % (data_name, aff_name+"/"+emotion_name, dim_name, aff_name+"-"+emotion_name))

    test_datas = []
    test_datas.extend(sydneysiege)
    test_datas.extend(ottawashooting)
    test_datas.extend(charliehebdo)
    test_datas.extend(ferguson)
    test_datas.extend(germanwings)
    test_datas.extend(prince)
    test_datas.extend(putinmissing)
    test_datas.extend(gurlitt)
    test_datas.extend(ebola)
    len(test_datas)

    os.makedirs(os.path.join(output_path, "data4inference", data_name, aff_name, emotion_name, dim_name), exist_ok=True)

    with open(output_path + "./data4inference/%s/%s/%s/%s/test.json"%(data_name, aff_name, emotion_name, dim_name), 'w', encoding="utf-8") as write_f:
        for index, test_data in enumerate(test_datas):
            data_one = {}

            Examples = "\nHere are a few examples:\n"
            for data_ in test_data["retried_data"][:num_examples]:
                Examples += " Text: " + data_["content"] + " The label of this text: " + data_["label"] + ".\n"
            Instruction = Task_prompt + "Target text: " + test_data[
                "content"] + Examples + " \nAccording to the above information, the label of target text:"


            data_one["Domain"] = test_data["domain"]
            data_one["Instruction"] = Instruction
            data_one["Answer"] = test_data["label"]
            write_f.write(json.dumps(data_one, indent=None, ensure_ascii=False) + "\n")
    metadata_test = pd.read_json(output_path + "./data4inference/%s/%s/%s/%s/test.json"%(data_name, aff_name, emotion_name, dim_name), lines=True, dtype={'num': str})
    metadata_test.to_csv(output_path + "./data4inference/%s/%s/%s/%s/test.csv"%(data_name, aff_name, emotion_name, dim_name), index=False)

    # Get json format
    input_file = output_path + "./data4inference/%s/%s/%s/%s/test.csv"%(data_name,  aff_name, emotion_name,dim_name)
    output_file = output_path + "./data4inference/%s/%s/%s/%s/%s-%s-%s-%s.json"%(data_name, aff_name, emotion_name, dim_name,data_name, aff_name, emotion_name, dim_name)
    csv2json(input_file, output_file)
    if not os.path.exists(output_path + "./data4inference/%s/all/"%(data_name)):
        os.mkdir(output_path + "./data4inference/%s/all/"%(data_name))
    path = output_file
    to_path = output_path + "./data4inference/%s/all/"%(data_name)
    os.system("cp %s %s"%(path,to_path))

def construct_instructs_EIregEIoc_addexpl(dim_name, num_examples, aff_name, data_name, Task_prompt, input_path, output_path):
    aff_name, emotion_name = aff_name.split("/")

    sydneysiege = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/sydneysiege-Retribasedon-%s.pth" % (data_name, aff_name+"/"+emotion_name, dim_name, aff_name+"-"+emotion_name))
    ottawashooting = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/ottawashooting-Retribasedon-%s.pth" % (data_name, aff_name+"/"+emotion_name, dim_name, aff_name+"-"+emotion_name))
    charliehebdo = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/charliehebdo-Retribasedon-%s.pth" % (data_name, aff_name+"/"+emotion_name, dim_name, aff_name+"-"+emotion_name))
    ferguson = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/ferguson-Retribasedon-%s.pth" % (data_name, aff_name+"/"+emotion_name, dim_name, aff_name+"-"+emotion_name))
    germanwings = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/germanwings-Retribasedon-%s.pth" % (data_name, aff_name+"/"+emotion_name, dim_name, aff_name+"-"+emotion_name))
    prince = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/prince-Retribasedon-%s.pth" % (data_name, aff_name+"/"+emotion_name, dim_name, aff_name+"-"+emotion_name))
    putinmissing = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/putinmissing-Retribasedon-%s.pth" % (data_name, aff_name+"/"+emotion_name, dim_name, aff_name+"-"+emotion_name))
    gurlitt = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/gurlitt-Retribasedon-%s.pth" % (data_name, aff_name+"/"+emotion_name, dim_name, aff_name+"-"+emotion_name))
    ebola = torch.load(
        input_path + "./retrieval/data4RAG/%s/%s/%s/ebola-Retribasedon-%s.pth" % (data_name, aff_name+"/"+emotion_name, dim_name, aff_name+"-"+emotion_name))

    test_datas = []
    test_datas.extend(sydneysiege)
    test_datas.extend(ottawashooting)
    test_datas.extend(charliehebdo)
    test_datas.extend(ferguson)
    test_datas.extend(germanwings)
    test_datas.extend(prince)
    test_datas.extend(putinmissing)
    test_datas.extend(gurlitt)
    test_datas.extend(ebola)
    len(test_datas)

    os.makedirs(os.path.join(output_path, "data4inference", data_name, aff_name, emotion_name, dim_name), exist_ok=True)

    with open(output_path + "./data4inference/%s/%s/%s/%s/test.json"%(data_name, aff_name, emotion_name, dim_name), 'w', encoding="utf-8") as write_f:
        for index, test_data in enumerate(test_datas):
            data_one = {}
            if aff_name == "EIreg":
                Examples = ".\nHere are a few examples retrieved through %s intensity:\n" % emotion_name
                for data_ in test_data["retried_data"][:num_examples]:
                    Examples += " Text: " + data_["content"] + " %s intensity: " % emotion_name + data_[
                        "EI_info"] + ". The label of this text: " + data_["label"] + ".\n"
                Instruction = Task_prompt + "Target text: " + test_data["content"] + " %s intensity: " % emotion_name + \
                              test_data["EI_info"] + Examples + " \nAccording to the above information, the label of target text:"
            elif aff_name == "EIoc":
                Examples = ".\nHere are a few examples retrieved through %s intensity classification:\n" % emotion_name
                for data_ in test_data["retried_data"][:num_examples]:
                    Examples += " Text: " + data_["content"] + " %s intensity classification: " % emotion_name + data_[
                        "EI_info"] + ". The label of this text: " + data_["label"] + ".\n"
                Instruction = Task_prompt + "Target text: " + test_data[
                    "content"] + " %s intensity classification: " % emotion_name + test_data[
                                  "EI_info"] + Examples + " \nAccording to the above information, the label of target text:"
            else:
                print("the aff name is err")

            data_one["Domain"] = test_data["domain"]
            data_one["Instruction"] = Instruction
            data_one["Answer"] = test_data["label"]
            write_f.write(json.dumps(data_one, indent=None, ensure_ascii=False) + "\n")
    metadata_test = pd.read_json(output_path + "./data4inference/%s/%s/%s/%s/test.json"%(data_name, aff_name, emotion_name, dim_name), lines=True, dtype={'num': str})
    metadata_test.to_csv(output_path + "./data4inference/%s/%s/%s/%s/test.csv"%(data_name, aff_name, emotion_name, dim_name), index=False)

    # Get json format
    input_file = output_path + "./data4inference/%s/%s/%s/%s/test.csv"%(data_name,  aff_name, emotion_name,dim_name)
    output_file = output_path + "./data4inference/%s/%s/%s/%s/%s-%s-%s-%s-addexpl.json"%(data_name, aff_name, emotion_name, dim_name,data_name, aff_name, emotion_name, dim_name)
    csv2json(input_file, output_file)
    if not os.path.exists(output_path + "./data4inference/%s/all/"%(data_name)):
        os.mkdir(output_path + "./data4inference/%s/all/"%(data_name))
    path = output_file
    to_path = output_path + "./data4inference/%s/all/"%(data_name)
    os.system("cp %s %s"%(path,to_path))

if __name__ == "__main__":
    # dim_name = "3d"
    # num_examples = 4
    # aff_name = "EIreg/anger"
    data_name = "PHEME"

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--aff_name', type=str, required=True)
    parser.add_argument('--dim_name', type=str, required=True)
    parser.add_argument('--num_examples', type=int, required=True)
    args = parser.parse_args()

    Task_prompt = "Task: Determine if the target text is 0. non-rumours or 1. rumours. Please answer the question directly without explanation. You must provide a specific answer. "
    # test
    if args.aff_name == "Vreg" or args.aff_name == "Voc" or args.aff_name == "Ec":
        construct_instructs_VregVocEc(args.dim_name, args.num_examples, args.aff_name, data_name, Task_prompt, args.input_path, args.output_path)
        construct_instructs_VregVocEc_addexpl(args.dim_name, args.num_examples, args.aff_name, data_name, Task_prompt, args.input_path, args.output_path)
    else:
        construct_instructs_EIregEIoc(args.dim_name, args.num_examples, args.aff_name, data_name, Task_prompt, args.input_path, args.output_path)
        construct_instructs_EIregEIoc_addexpl(args.dim_name, args.num_examples, args.aff_name, data_name, Task_prompt, args.input_path, args.output_path)
