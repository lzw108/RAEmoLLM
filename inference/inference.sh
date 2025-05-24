#export CUDA_VISIBLE_DEVICES='1'
export ABS_PATH=$(pwd) # local project path

# checkpoint
model_name_or_path=mistralai/Mistral-7B-Instruct-v0.2

data_name=PHEME
aff_name=Vreg #  Vreg, Voc, Ec, EIreg/anger, EIreg/fear, EIreg/joy, EIreg/sadness, EIoc/anger, EIoc/fear, EIoc/joy, EIoc/sadness
dim=4096d 

predict_file=$ABS_PATH/inference/generate_results/$data_name-$aff_name-$dim-predict.json
if [[ $aff_name == EIreg* || $aff_name == EIoc* ]]; then
  infer_file=$ABS_PATH/inference/data4inference/$data_name/$aff_name/$dim/$data_name-${aff_name//\//-}-$dim.json
else
  infer_file=$ABS_PATH/inference/data4inference/$data_name/$aff_name/$dim/$data_name-$aff_name-$dim.json
fi

# inference
python inference/inference.py \
    --model_name_or_path $model_name_or_path \
    --infer_file $infer_file \
    --predict_file $predict_file \
    --batch_size 8 \
    --seed 123

