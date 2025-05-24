#export CUDA_VISIBLE_DEVICES='1'
export ABS_PATH=$(pwd) # local project path

# get embds
model_name_or_path=lzw1008/Emollama-chat-7b
data_name=PHEME # AMTCele, PHEME, COCO
aff_name=Vreg # Vreg, Voc, Ec, EIreg, EIoc

# data path
infer_file=$ABS_PATH/indexconstruct/affective_analysis/instructdata/$data_name-$aff_name.csv
# pridict path
predict_file=$ABS_PATH/indexconstruct/affective_analysis/embedds/$data_name-$aff_name-embeddings.pth

embd_path=$ABS_PATH/indexconstruct/affective_analysis/embedds/

python indexconstruct/get_embs.py \
    --model_name_or_path $model_name_or_path \
    --infer_file $infer_file \
    --predict_file $predict_file \
    --batch_size 8 \
    --seed 123


