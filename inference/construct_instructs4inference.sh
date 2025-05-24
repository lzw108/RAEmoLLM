#export CUDA_VISIBLE_DEVICES='2'
export ABS_PATH=$(pwd) # change to your local project path

data_name=PHEME # AMTCele, PHEME, COCO
aff_name=Vreg # Vreg, Voc, Ec, EIreg/anger, EIreg/fear, EIreg/joy, EIreg/sadness, EIoc/anger, EIoc/fear, EIoc/joy, EIoc/sadness
dim_name=4096d # # 8d, 16d, 128d, 512d, 4096d
num_examples=4

# data path
input_path=$ABS_PATH/
# pridict path
output_path=$ABS_PATH/inference/
# inference
python inference/construct_instructs4inference_$data_name.py \
    --input_path $input_path \
    --output_path $output_path \
    --aff_name $aff_name \
    --dim_name $dim_name \
    --num_examples $num_examples

