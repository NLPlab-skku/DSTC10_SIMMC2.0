CUDA_VISIBLE_DEVICES=1 python main.py \
    --mode='fine'\
    --train_file="../data/simmc2_disambiguate_dstc10_total_train.json" \
    --dev_file="../data/simmc2_disambiguate_dstc10_dev.json" \
    --devtest_file="../data/simmc2_disambiguate_dstc10_teststd_public.json" \
    --model_name_or_path="./save_model/20211007_FULL_POST(30)_FULL_FINE(5)_Seed3_Dev99.749_Test99.159"\
    --model="roberta-large"\
    --submission\
    --do_eval\
   