#export CUDA_VISIBLE_DEVICES='3'
export ABS_PATH=$(pwd) # local project path

# Emollama-chat-7b in huggingface
model_name_or_path=lzw1008/Emollama-chat-7b

data_name=COCO # AMTCele, PHEME, COCO
data_path=$ABS_PATH/datasets/$data_name.csv
output_path=$ABS_PATH/indexconstruct/affective_analysis/instructdata/
# construct aff instructs for emollms
python indexconstruct/construct_aff_instructs.py \
    --data_name $data_name \
    --data_path $data_path \
    --output_path $output_path

# get labels
# data path
infer_file=$ABS_PATH/indexconstruct/affective_analysis/instructdata/$data_name-affective-all.json
# pridict path
predict_file=$ABS_PATH/indexconstruct/affective_analysis/$data_name-predict.json
# inference
python indexconstruct/get_Emolabel.py \
    --model_name_or_path $model_name_or_path \
    --infer_file $infer_file \
    --predict_file $predict_file \
    --batch_size 8 \
    --seed 123


output_path_aff=$ABS_PATH/indexconstruct/affective_analysis/$data_name-add-aff.csv

python indexconstruct/postprocess_label.py \
    --data_name $data_name \
    --predict_file $predict_file \
    --data_path $data_path \
    --output_path $output_path_aff
