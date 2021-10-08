CUDA_VISIBLE_DEVICES=1 python main.py \
    --mode='fine'\
    --train_file="../data/simmc2_disambiguate_dstc10_total_train.json" \
    --dev_file="../data/simmc2_disambiguate_dstc10_dev.json" \
    --devtest_file="../data/simmc2_disambiguate_dstc10_devtest.json" \
    --model_name_or_path="./save_model/post/20211007_FULL_DATA(TRAIN_DEV_DEVTEST)_30epoch_Seed1"\
    --model="roberta-large"\
    --seed=6\
    --epochs=5\
    --batch_size=6\
    --learning_rate=2e-5\
    --do_train\
    --do_eval\

CUDA_VISIBLE_DEVICES=1 python main.py \
    --mode='fine'\
    --train_file="../data/simmc2_disambiguate_dstc10_total_train.json" \
    --dev_file="../data/simmc2_disambiguate_dstc10_dev.json" \
    --devtest_file="../data/simmc2_disambiguate_dstc10_devtest.json" \
    --model_name_or_path="./save_model/post/20211007_FULL_DATA(TRAIN_DEV_DEVTEST)_30epoch_Seed1"\
    --model="roberta-large"\
    --seed=7\
    --epochs=5\
    --batch_size=6\
    --learning_rate=2e-5\
    --do_train\
    --do_eval\


CUDA_VISIBLE_DEVICES=1 python main.py \
    --mode='fine'\
    --train_file="../data/simmc2_disambiguate_dstc10_total_train.json" \
    --dev_file="../data/simmc2_disambiguate_dstc10_dev.json" \
    --devtest_file="../data/simmc2_disambiguate_dstc10_devtest.json" \
    --model_name_or_path="./save_model/post/20211007_FULL_DATA(TRAIN_DEV_DEVTEST)_30epoch_Seed1"\
    --model="roberta-large"\
    --seed=8\
    --epochs=5\
    --batch_size=6\
    --learning_rate=2e-5\
    --do_train\
    --do_eval\

CUDA_VISIBLE_DEVICES=1 python main.py \
    --mode='fine'\
    --train_file="../data/simmc2_disambiguate_dstc10_total_train.json" \
    --dev_file="../data/simmc2_disambiguate_dstc10_dev.json" \
    --devtest_file="../data/simmc2_disambiguate_dstc10_devtest.json" \
    --model_name_or_path="./save_model/post/20211007_FULL_DATA(TRAIN_DEV_DEVTEST)_30epoch_Seed1"\
    --model="roberta-large"\
    --seed=9\
    --epochs=5\
    --batch_size=6\
    --learning_rate=2e-5\
    --do_train\
    --do_eval\
    
CUDA_VISIBLE_DEVICES=1 python main.py \
    --mode='fine'\
    --train_file="../data/simmc2_disambiguate_dstc10_total_train.json" \
    --dev_file="../data/simmc2_disambiguate_dstc10_dev.json" \
    --devtest_file="../data/simmc2_disambiguate_dstc10_devtest.json" \
    --model_name_or_path="./save_model/post/20211007_FULL_DATA(TRAIN_DEV_DEVTEST)_30epoch_Seed1"\
    --model="roberta-large"\
    --seed=10\
    --epochs=5\
    --batch_size=6\
    --learning_rate=2e-5\
    --do_train\
    --do_eval\