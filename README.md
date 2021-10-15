# _DSTC10 Track3 SIMMC 2.0 (SKKU-NLP)_

## Overview
- This model is for DSTC10 Track3. Sub-Task 1 model is based on RoBERTa which has a powerful enocding ablity. Sub-Task 2~4 model is based on a pipeline of ResNet and BART.
- The final submission files are in the "final" branch
---
## Enviroment

- CUDA 11.1
- Python 3.7+

Packages:
- torch==1.8.0
- transformers==4.8.0
- nltk==3.6.2
- scikit-learn==0.24.2
- tensorboard==2.6.0
- tensorboardX==2.4
- wordninja==2.0.0
---
## Model Parameters
- Download the Fine-tuned ResNet
Google Drive Link: https://drive.google.com/file/d/1pz804uvlVBsiBM6bDfbvWydKMc7T7HOa/view?usp=sharing
- Download the Fine-tuned RoBERTa (for sub-task 1)
Google Drive Link: https://drive.google.com/file/d/1BLVi4_3yCyh-815M7MQ-cc-zKcqCAKGF/view?usp=sharing
- Download the Fine-tuned BART (for sub-task 2-4)
Google Drive Link: https://drive.google.com/file/d/1KDEYLVm4Ka5WRclANfVEV54IaPGdoBYo/view?usp=sharing
---
## Datasets & Models
```sh
model
  |---data
  |   |--simmc2_dials_dstc10_train.json                                               # The official data released (train)
  |   |--simmc2_dials_dstc10_teststd_public.json                            # The official data released (teststd)    
  |   |-- ...
  |   |--fashion_prefab_metadata_all.json                                           # The meta data released (Fashion)
  |   |--furniture_prefab_metadata_all.json                                        # The meta data released (Furniture)
  |   |--visual_meta_data_predicted_100.pickle                                # Predicted visual metadata of teststd, we got using ResNet
  |   |--simmc2_scene_images_dstc10_teststd
  |   |   |--cloth_store_1416238_woman_12_0_bbox.json
  |   |   |--...
  
|--disambiguate
|   |--data
|   |--RoBERTa
|   |   |--save_model
|   |   |   |--model                                                                                          # Place the downloaded fine-tuned RoBERTa, naemd 'model'

|--mm_dst
|   |--gpt2_dst
|   |--save
|   |   |--model                                                                                                # Place the downloaded fine-tuned BART, naemd 'model'
|   |--data
|   |   |--simmc2_dials_dstc10_teststd_public_predict.txt
|   |   |--...

```
---

## Preprocessing

### For sub task 1
```sh
# (train/dev/devtet)
python format_disambiguation_data.py \
	--simmc_train_json="../data/simmc2_dials_dstc10_train.json" \
	--simmc_dev_json="../data/simmc2_dials_dstc10_dev.json" \
	--simmc_devtest_json="../data/simmc2_dials_dstc10_devtest.json" \
	--disambiguate_save_path="./data/"

# (teststd)
python format_disambiguation_data.py \
	--simmc_teststd_public_json="../data/simmc2_dials_dstc10_teststd_public.json" \
	--disambiguate_save_path="./data/"\
```

### For sub task 2-4
```sh
# (train/dev/devtest)
python -m gpt2_dst.scripts.preprocess_input \
    --input_path_json="../data/simmc2_dials_dstc10_train.json"\
    --output_path_predict="./gpt2_dst/data/simmc2_dials_dstc10_train_predict.txt" \
    --output_path_target="./gpt2_dst/data_parallel/simmc2_dials_dstc10_train_target.txt" \
    --len_context=6 \
    --use_multimodal_contexts=1 \
    --output_path_special_tokens="./gpt2_dst/data/simmc2_special_tokens.json"

python -m gpt2_dst.scripts.preprocess_input \
    --input_path_json="../data/simmc2_dials_dstc10_dev.json"\
    --output_path_predict="./gpt2_dst/data/simmc2_dials_dstc10_dev_predict.txt" \
    --output_path_target="./gpt2_dst/data_parallel/simmc2_dials_dstc10_dev_target.txt" \
    --len_context=6 \
    --use_multimodal_contexts=1 \
    --input_path_special_tokens="./gpt2_dst/data/simmc2_special_tokens.json" \
    --output_path_special_tokens="./gpt2_dst/data/simmc2_special_tokens.json"

python -m gpt2_dst.scripts.preprocess_input \
    --input_path_json="../data/simmc2_dials_dstc10_devtest.json"\
    --output_path_predict="./gpt2_dst/data/simmc2_dials_dstc10_devtest_predict.txt" \
    --output_path_target="./gpt2_dst/data_parallel/simmc2_dials_dstc10_devtest_target.txt" \
    --len_context=6 \
    --use_multimodal_contexts=1 \
    --input_path_special_tokens="./gpt2_dst/data/simmc2_special_tokens.json" \
    --output_path_special_tokens="./gpt2_dst/data/simmc2_special_tokens.json"

# (teststd)
python -m gpt2_dst.scripts.preprocess_input \
    --input_path_json="../data/simmc2_dials_dstc10_teststd_public.json"\
    --output_path_predict="./gpt2_dst/data/simmc2_dials_dstc10_teststd_public_predict.txt" \
    --len_context=6 \
    --use_multimodal_contexts=1 \
    --output_path_special_tokens="./gpt2_dst/data/simmc2_special_tokens.json"
```

---
## Training
### For sub task 1 (./disambiguate/RoBERTa/)
```sh
python main.py \
    --mode='fine'\
    --train_file="../data/simmc2_disambiguate_dstc10_train.json" \
    --dev_file="../data/simmc2_disambiguate_dstc10_dev.json" \
    --devtest_file="../data/simmc2_disambiguate_dstc10_devtest.json" \
    --model="roberta-large"\
    --seed=1\
    --epochs=5\
    --batch_size=6\
    --learning_rate=2e-5\
    --do_train\
    --do_eval\
```

### For sub task 2-4 (./mm_dst/)
```sh
 python -m gpt2_dst.scripts.run_language_modeling_BART \
    --output_dir="./gpt2_dst/save/model" \
    --model_type=facebook/bart-large \
    --model_name_or_path=facebook/bart-large\
    --do_train\
    --line_by_line \
    --add_special_tokens="./gpt2_dst/data/simmc2_special_tokens.json" \
    --train_data_pred_file="./gpt2_dst/data/simmc2_dials_dstc10_total_train_predict.txt" \
    --train_data_target_file="./gpt2_dst/data/simmc2_dials_dstc10_total_train_target.txt" \
    --do_eval\
    --eval_data_pred_file="./gpt2_dst/data/simmc2_dials_dstc10_dev_predict.txt" \
    --eval_data_target_file="./gpt2_dst/data/simmc2_dials_dstc10_dev_target.txt" \
    --num_train_epochs=10 \
    --overwrite_output_dir \
    --per_gpu_train_batch_size=4\
    --per_gpu_eval_batch_size=4\
    --save_steps=50000 \
    --logging_steps=50 \
```

---
## Generation (only sub task 2-4)
```sh
python -m gpt2_dst.scripts.run_generation_BART \
    --model_type=facebook/bart-large \
    --model_name_or_path="./gpt2_dst/save/model/" \
    --length=100 \
    --num_return_sequences=1 \
    --seed=1\
    --stop_token='<end>' \
    --prompts_from_file="./gpt2_dst/data/simmc2_dials_dstc10_teststd_public_predict.txt" \
    --path_output="./gpt2_dst/results/simmc2_dials_dstc10_teststd_public_predicted.txt"\

```

---
## Final result (we submitted)
The final prediction files that we submitted are located in the path below each branch.
**These are the final prediction results intended to be compared to those of other models**
### sub task 1
```sh
branch : final
# Ensemble model result
/model/disambiguate/RoBERTa/save_model/results/entry1/dstc10-simmc-teststd-pred-subtask-1.json

# Single model result
/model/disambiguate/RoBERTa/save_model/results/entry2/dstc10-simmc-teststd-pred-subtask-1.json
```

### sub task 2-4 (line-by-line)
```sh
branch : final
/model/mm_dst/gpt2_dst/results/dstc10-simmc-teststd-pred-subtask-3.txt

branch : final
/model/mm_dst/gpt2_dst/results/dstc10-simmc-teststd-pred-subtask-4.txt
```
