# LLMs
import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, required=True)
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--predict_file', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
args = parser.parse_args()

df = pd.read_csv(args.data_path)
metadata = pd.read_json(args.predict_file, lines=True, dtype={'num': str})
def get_response(output):
    response = output.split("Assistant:")[1].replace("\n"," ").strip()
    return response
metadata['Assistant'] = metadata['output'].apply(get_response)

metadata_Assitant = metadata["Assistant"]

split_num = len(df)

df_EIreg = metadata_Assitant[0:4*split_num].reset_index(drop=True)
df_EIoc = metadata_Assitant[4*split_num:8*split_num].reset_index(drop=True)
df_Vreg = metadata_Assitant[8*split_num:9*split_num].reset_index(drop=True)
df_Voc = metadata_Assitant[9*split_num:10*split_num].reset_index(drop=True)
df_Ec = metadata_Assitant[10*split_num:11*split_num].reset_index(drop=True)

df_EIreg_anger = df_EIreg[0:split_num].reset_index(drop=True)
df_EIreg_fear = df_EIreg[split_num:2*split_num].reset_index(drop=True)
df_EIreg_joy = df_EIreg[2*split_num:3*split_num].reset_index(drop=True)
df_EIreg_sadness = df_EIreg[3*split_num:4*split_num].reset_index(drop=True)

df_EIoc_anger = df_EIoc[0:split_num].reset_index(drop=True)
df_EIoc_fear = df_EIoc[split_num:2*split_num].reset_index(drop=True)
df_EIoc_joy = df_EIoc[2*split_num:3*split_num].reset_index(drop=True)
df_EIoc_sadness = df_EIoc[3*split_num:4*split_num].reset_index(drop=True)

df["anger_reg"] = df_EIreg_anger
df["fear_reg"] = df_EIreg_fear
df["joy_reg"] = df_EIreg_joy
df["sadness_reg"] = df_EIreg_sadness

df["anger_oc"] = df_EIoc_anger
df["fear_oc"] = df_EIoc_fear
df["joy_oc"] = df_EIoc_joy
df["sadness_oc"] = df_EIoc_sadness

df["Vreg"] = df_Vreg
df["Voc"] = df_Voc
df["Ec"] = df_Ec

df.to_csv(args.output_path, index=False)