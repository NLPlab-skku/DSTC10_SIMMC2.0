CUDA_VISIBLE_DEVICES=1 python main.py \
    --mode='post'\
    --train_file="../data/simmc2_disambiguate_dstc10_total_train.json" \
    --dev_file="../data/simmc2_disambiguate_dstc10_dev.json" \
    --devtest_file="../data/simmc2_disambiguate_dstc10_devtest.json" \
    --model="roberta-large"\
    --seed=1\
    --epochs=30\
    --batch_size=6\
    --learning_rate=2e-5\
    --do_train\
    --do_eval\