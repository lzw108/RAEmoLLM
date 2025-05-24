#export CUDA_VISIBLE_DEVICES='2'
export ABS_PATH=$(pwd) # local project path

data_name=PHEME # AMTCele, PHEME, COCO
aff_name=Vreg # Vreg, Voc, Ec, EIreg, EIoc
dim_name=4096d
retri_num=4

# data path
input_path=$ABS_PATH/indexconstruct/affective_analysis/
# pridict path
output_path=$ABS_PATH/retrieval/
# inference
python retrieval/retrieval_$data_name.py \
    --input_path $input_path \
    --output_path $output_path \
    --aff_name $aff_name \
    --dim_name $dim_name \
    --retri_num $retri_num
