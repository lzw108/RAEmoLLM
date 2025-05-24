import torch
import re
import json
from text2vec import semantic_search
import pandas as pd
import numpy as np
import torch
import copy
import os
import argparse

def embedding_search(query, embs, top_k, flag=None):
    hits = semantic_search(query, embs, top_k=top_k)
    hits = hits[0]
    return hits

def get_retri(query, embs, topk, retri_datas, filter_threshold=0.2):
    res_embedding = embedding_search(query, embs, topk)
    res = [i for i in res_embedding if i["score"] >= filter_threshold]
    res_sorted = sorted(res, key=lambda x: x["score"], reverse=True)
    lst_idx = [i["corpus_id"] for i in res_sorted]
    retried_datas = [retri_datas[i] for i in lst_idx]
    return retried_datas

def process_VregVocEc(aff_name,data_name,dim_name,retri_num,output_path):

    datas = []
    for index, row in df_label_text.iterrows():
        one_list = []
        domain = re.sub(r'\d+', '', row["event"])
        label = row["rumor_label"]
        content = row["text"]
        emb = aff_embeds[index]
        aff_info = row[aff_name]
        datas.append({"domain": domain, "label":label,"content":content,"aff_info":aff_info,"embedding":emb})

    sydneysiege = [item for item in datas if item['domain'] == 'sydneysiege']
    ottawashooting = [item for item in datas if item['domain'] == 'ottawashooting']
    charliehebdo = [item for item in datas if item['domain'] == 'charliehebdo']
    ferguson = [item for item in datas if item['domain'] == 'ferguson']
    germanwings = [item for item in datas if item['domain'] == 'germanwings']
    prince = [item for item in datas if item['domain'] == 'prince']
    putinmissing = [item for item in datas if item['domain'] == 'putinmissing']
    gurlitt = [item for item in datas if item['domain'] == 'gurlitt']
    ebola = [item for item in datas if item['domain'] == 'ebola']

    all_data = [sydneysiege, ottawashooting, charliehebdo, ferguson, germanwings, prince, putinmissing, gurlitt, ebola]
    domain_names = ["sydneysiege", "ottawashooting", "charliehebdo", "ferguson", "germanwings", "prince", "putinmissing","gurlitt","ebola"]

    for i in range(len(domain_names)):
        print(i)
        retri_datas = []
        for j, domain_data in enumerate(all_data):
            if j == i:
                test_datas = copy.deepcopy(domain_data)
                domain_name = domain_names[j]
            else:
                retri_datas.extend(domain_data)

        embs = [item['embedding'] for item in retri_datas]
        embs = np.array(embs)

        new_datas = []
        for test_data in test_datas:
            query = test_data['embedding']
            retried_datas = get_retri(query, embs, retri_num, retri_datas)
            test_data["retried_data"] = retried_datas
            new_datas.append(test_data)

        os.makedirs(os.path.join(output_path, "data4RAG", data_name, aff_name, dim_name), exist_ok=True)

        torch.save(new_datas, output_path + "./data4RAG/%s/%s/%s/%s-Retribasedon-%s.pth" % (data_name, aff_name, dim_name, domain_name, aff_name.split("/")[0]))

def process_EI(aff_name,data_name,dim_name,retri_num,output_path):

    EI_anger_embeds = aff_embeds[0:len(df_label_text)]
    EI_fear_embeds = aff_embeds[len(df_label_text):2*len(df_label_text)]
    EI_joy_embeds = aff_embeds[2*len(df_label_text):3*len(df_label_text)]
    EI_sadness_embeds = aff_embeds[3*len(df_label_text):4*len(df_label_text)]

    emotions = ["anger", "fear", "joy", "sadness"]
    all_embeds = [EI_anger_embeds, EI_fear_embeds, EI_joy_embeds, EI_sadness_embeds]
    if aff_name == "EIreg":
        emotion_names = ["anger_reg", "fear_reg", "joy_reg", "sadness_reg"]
    elif aff_name == "EIoc":
        emotion_names = ["anger_oc", "fear_oc", "joy_oc", "sadness_oc"]
    else:
        print("the affective name is not EIreg or EIoc")

    for emotion_index in range(4):
        datas = []
        for index, row in df_label_text.iterrows():
            domain = re.sub(r'\d+', '', row["event"])
            label = row["rumor_label"]
            content = row["text"]
            EI_info = str(row[emotion_names[emotion_index]])
            emb = all_embeds[emotion_index][index]
            datas.append({"domain": domain, "label": label, "content": content, "EI_info": EI_info, "embedding": emb})

        sydneysiege = [item for item in datas if item['domain'] == 'sydneysiege']
        ottawashooting = [item for item in datas if item['domain'] == 'ottawashooting']
        charliehebdo = [item for item in datas if item['domain'] == 'charliehebdo']
        ferguson = [item for item in datas if item['domain'] == 'ferguson']
        germanwings = [item for item in datas if item['domain'] == 'germanwings']
        prince = [item for item in datas if item['domain'] == 'prince']
        putinmissing = [item for item in datas if item['domain'] == 'putinmissing']
        gurlitt = [item for item in datas if item['domain'] == 'gurlitt']
        ebola = [item for item in datas if item['domain'] == 'ebola']

        all_data = [sydneysiege, ottawashooting, charliehebdo, ferguson, germanwings, prince, putinmissing, gurlitt,
                    ebola]
        domain_names = ["sydneysiege", "ottawashooting", "charliehebdo", "ferguson", "germanwings", "prince",
                        "putinmissing", "gurlitt", "ebola"]
        for i in range(0, 9):
            print(i)
            retri_datas = []
            for j, domain_data in enumerate(all_data):
                if j == i:
                    test_datas = copy.deepcopy(domain_data)
                    domain_name = domain_names[j]
                else:
                    retri_datas.extend(domain_data)

            embs = [item['embedding'] for item in retri_datas]
            embs = np.array(embs)

            new_datas = []
            for test_data in test_datas:
                query = test_data['embedding']
                retried_datas = get_retri(query, embs, retri_num, retri_datas)
                test_data["retried_data"] = retried_datas
                new_datas.append(test_data)

            os.makedirs(os.path.join(output_path, "data4RAG", data_name, aff_name, emotions[emotion_index], dim_name),
                        exist_ok=True)

            torch.save(new_datas, output_path + "./data4RAG/%s/%s/%s/%s/%s-Retribasedon-%s.pth" % (data_name, aff_name, emotions[emotion_index], dim_name, domain_name, aff_name+"-"+emotions[emotion_index]))


if __name__ == "__main__":
    data_name = "PHEME"

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--aff_name', type=str, required=True)
    parser.add_argument('--dim_name', type=str, required=True)
    parser.add_argument('--retri_num', type=int, required=True)
    args = parser.parse_args()

    if args.dim_name == "4096d":
        aff_embeds = torch.load(
            args.input_path + "/embedds/%s-%s-embeddings.pth" % (data_name, args.aff_name))
    else:
        aff_embeds = torch.load(
            args.input_path + "/embedds/%s-%s-embeddings-%s.pth" % (data_name, args.aff_name, args.dim_name))

    df_label_text = pd.read_csv(args.input_path + "/%s-add-aff.csv" % data_name)
    if args.aff_name == "Vreg" or args.aff_name == "Voc" or args.aff_name == "Ec":
        process_VregVocEc(args.aff_name, data_name, args.dim_name, args.retri_num, args.output_path)
    elif args.aff_name == "EIreg" or args.aff_name == "EIoc":
        process_EI(args.aff_name, data_name, args.dim_name, args.retri_num, args.output_path)
    else:
        print("the aff_name is err....")