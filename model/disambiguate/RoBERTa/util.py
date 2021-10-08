import torch
import torch.nn as nn
import numpy as np
import random
import json
import os
import argparse
import time
import copy
import datetime
from glob import glob
from tqdm import tqdm,trange
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score,classification_report, confusion_matrix
from transformers import RobertaTokenizerFast, AutoConfig, AdamW, get_linear_schedule_with_warmup
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification,RobertaForMaskedLM,RobertaClassificationHead
from torch.utils.data import DataLoader,TensorDataset
from masker import BertMasker



class RoBERTa_MLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.classifier = RobertaClassificationHead(config)
        print("\nRoBERTa_MLM")
        self.init_weights()

    def forward(self,\
        args=None,\
        input_ids=None,\
        attention_mask=None,\
        masked_lm_labels=None,\
        labels = None,
        ):

        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        prediction_scores = self.lm_head(sequence_output)
        logits = self.classifier(sequence_output)

        total_loss = 0
 
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            total_loss += loss

        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            total_loss += args.mlm_loss * masked_lm_loss

        return (total_loss,logits)

DOMAIN = {'fashion':0,'furniture':1}

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_device(args):
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        args.device = torch.device('cpu')
        print('Device name: cpu')
def tokenize(args, data_path, tokenizer, augmented=False):
    text_labels = list()
    text_inputs = list()
    domains = list()
    total_data = list()
    dialog_ids = list()
    turn_ids = list()
    if augmented:
        with open(data_path, "r") as inp:
            data = inp.readlines()
        for ex in data:
            label, text = ex.split("\t")
            text_inputs.append(text)
            text_labels.append(int(label))
    else:    
        with open(data_path, "r") as file_id:
            _raw_data = json.load(file_id)
            num_utterances = 2 * args.max_turns + 1
            num_instances = len(_raw_data)            
            for data in tqdm(_raw_data, desc="Tokenize"):
                dialog_datum = data
                dialog = data['input_text'].copy()
                domains.append(DOMAIN[dialog_datum['domain']])

                for turn_id, turn in enumerate(dialog):
                    if turn_id % 2 == 0:
                        dialog[turn_id] = "<USER> " + turn
                    else:
                        dialog[turn_id] = "<SYS> " + turn
                text = " ".join(dialog[-num_utterances :])
                text_inputs.append(text)
                if args.submission and dialog_datum["disambiguation_label_gt"] is None:
                    text_labels.append(0)                    
                else:
                    text_labels.append(dialog_datum["disambiguation_label_gt"])
                dialog_ids.append(dialog_datum["dialog_id"])
                turn_ids.append(dialog_datum["turn_id"])

            with open(data_path.replace(".json", ".txt"), mode='w') as out:
                for text, label in zip(text_inputs,text_labels):
                    out.write("{}\t{}\n".format(label,text))

    encoded_inputs = tokenizer(
        text_inputs, add_special_tokens=True, return_tensors="pt", padding=True, truncation=True,
    )
    args.total_tokens = torch.nonzero(encoded_inputs['attention_mask']==1)

    print("text labels : {} {}".format(len(text_labels), text_labels[:10]))
    print("domains : {} {}".format(len(domains), domains[:10]))

    return encoded_inputs,torch.tensor(text_labels, dtype=torch.long), torch.tensor(domains,dtype=torch.long),\
            torch.tensor(dialog_ids, dtype=torch.long), torch.tensor(turn_ids, dtype=torch.long)


def tokenize_without_additional_tokens(args, data_path, tokenizer):
    text_labels = []
    text_inputs = []
    dialog_ids = []
    turn_ids = []
    total_data = list()

    with open(data_path, "r") as file_id:
        _raw_data = json.load(file_id)

        num_utterances = 2 * args.max_turns + 1
        num_instances = len(_raw_data)
        for data in tqdm(_raw_data, desc="Tokenize"):
            dialog_datum = data
            dialog = data['input_text'].copy()
            for turn_id, turn in enumerate(dialog):
                # if turn_id % 2 == 0:
                #     dialog[turn_id] = "<USER> " + turn
                # else:
                #     dialog[turn_id] = "<SYS> " + turn
                if turn_id == 0:    dialog[turn_id] = turn
                else:               dialog[turn_id] = "</s><s>" + turn                

            dialog = dialog[-num_utterances :]
            dialog[0] = dialog[0].replace("</s><s>", "")
            text = "".join(dialog)
            text_inputs.append(text)
            text_labels.append(dialog_datum["disambiguation_label_gt"])
            dialog_ids.append(dialog_datum["dialog_id"])
            turn_ids.append(dialog_datum["turn_id"])
        encoded_inputs = tokenizer(
            text_inputs, add_special_tokens=True, return_tensors="pt", padding=True, truncation=True,
        )
        args.total_tokens = torch.nonzero(encoded_inputs['attention_mask']==1)
    return encoded_inputs,torch.tensor(text_labels, dtype=torch.long)


def tokenize_split_history_and_current(args, data_path, tokenizer):
    text_labels = []
    text_inputs = []
    dialog_ids = []
    turn_ids = []
    total_data = list()

    with open(data_path, "r") as file_id:
        _raw_data = json.load(file_id)
        num_utterances = 2 * args.max_turns + 1
        num_instances = len(_raw_data)
        for data in tqdm(_raw_data, desc="Tokenize"):
            dialog_datum = data
            dialog = data['input_text'].copy()
            delimiter = ""
            for turn_id, turn in enumerate(dialog):
                if turn_id+1 ==len(dialog)-1:   # 마지막 발화 턴이라면
                    delimiter = "</s><s> "

                if turn_id % 2 == 0:
                    dialog[turn_id] = delimiter + "<USER> " + turn
                else:
                    dialog[turn_id] = "<SYS> " + turn

            dialog = dialog[-num_utterances :]
            dialog[0] = dialog[0].replace("</s><s>", "")
            text = " ".join(dialog)
            text_inputs.append(text)
            text_labels.append(dialog_datum["disambiguation_label_gt"])
            dialog_ids.append(dialog_datum["dialog_id"])
            turn_ids.append(dialog_datum["turn_id"])

        encoded_inputs = tokenizer(
            text_inputs, add_special_tokens=True, return_tensors="pt", padding=True, truncation=True,
        )
        args.total_tokens = torch.nonzero(encoded_inputs['attention_mask']==1)
    return encoded_inputs,torch.tensor(text_labels, dtype=torch.long)


MODEL_CLASSES = {
    'fine' : RobertaForSequenceClassification,
    'post' : RobertaForMaskedLM,
    'multi' : RoBERTa_MLM,
}
def get_model(args, model):
    # args.model_name_or_path = args.model_name_or_path.replace("/best_model.bin","")
    tokenizer = RobertaTokenizerFast.from_pretrained(args.model)
    special_tokens_dict = {'additional_special_tokens': ['<USER>','<SYS>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    config = AutoConfig.from_pretrained(args.model)
    config.num_labels = 2

    if not args.do_train and\
        args.voting_models is not None:
        model = list()
        for path in args.voting_models:                     model.append(MODEL_CLASSES['fine'].from_pretrained(pretrained_model_name_or_path=args.model, config=config))
        for num in range(len(args.voting_models)):          model[num].resize_token_embeddings(len(tokenizer))
        for num,path in enumerate(args.voting_models):      model[num] = model[num].from_pretrained(pretrained_model_name_or_path=path + "/best_model.bin", config=config)
        for num in range(len(args.voting_models)):          model[num].to(args.device)

        return model.copy()

    elif not args.do_train and\
        args.fashion_model_name_or_path is not None and\
        args.furniture_model_name_or_path is not None:

        fashion_model = MODEL_CLASSES['fine'].from_pretrained(pretrained_model_name_or_path=args.model, config=config)
        furniture_model = MODEL_CLASSES['fine'].from_pretrained(pretrained_model_name_or_path=args.model, config=config)
        fashion_model.resize_token_embeddings(len(tokenizer))
        furniture_model.resize_token_embeddings(len(tokenizer))
        fashion_model = fashion_model.from_pretrained(pretrained_model_name_or_path=args.fashion_model_name_or_path + "/best_model.bin", config=config)
        furniture_model = furniture_model.from_pretrained(pretrained_model_name_or_path=args.furniture_model_name_or_path + "/best_model.bin", config=config)

        model = {0:fashion_model.to(args.device),
                  1:furniture_model.to(args.device)}

    elif not (args.mode == 'post' or args.mode == 'fine')\
        and args.model_name_or_path is not None\
        and os.path.isfile(os.path.join(args.model_name_or_path, "best_model.bin")):
        print("\n===== [ Re: ] =====")
        # print("\n{}개 신규 토큰 추가".format(num_added_toks))

        model = MODEL_CLASSES['fine'].from_pretrained(pretrained_model_name_or_path=args.model, config=config)
        model.resize_token_embeddings(len(tokenizer))

        saved_model = args.model_name_or_path + "/best_model.bin"
        model = model.from_pretrained(pretrained_model_name_or_path=saved_model, config=config)
        model.to(args.device)
    else:
        return model
    return model

def get_masker(args, tokenizer):
    if args.mode == 'fine': return None
    elif not args.mode == 'fine' : return BertMasker(tokenizer)

def idx_to_tensor(args, data, labels, bsz=1, domain=None, dialog_ids=None, turn_ids=None, train=False):
    print("Domain : {}".format(domain))
    if dialog_ids is not None and turn_ids is not None: dataset  = TensorDataset(data.input_ids, data.attention_mask, labels, domain, dialog_ids, turn_ids)
    elif domain is not None:                            dataset  = TensorDataset(data.input_ids, data.attention_mask, labels, domain)        
    else:                                               dataset  = TensorDataset(data.input_ids, data.attention_mask, labels)


    dataloader    = DataLoader(dataset, batch_size=bsz, shuffle=train)

    return dataloader

def save_checkpoint(args, best_model):
    today = str(datetime.datetime.today().strftime("%Y%m%d"))

    if not args.mode == 'post': output_dir = "./save_model/{}_mode{}_seed{}_lr{}_bs{}_ac{}_len{}_epoch{}".format(today, args.mode, args.seed,args.learning_rate, args.batch_size, args.gradient_accumulation_steps, args.max_seq_length,args.epochs)
    if args.mode == 'post': output_dir = "./save_model/post/{}_mode{}_seed{}_lr{}_bs{}_ac{}_len{}_epoch{}".format(today, args.mode, args.seed,args.learning_rate, args.batch_size, args.gradient_accumulation_steps, args.max_seq_length,args.epochs)

    args.model_name_or_path = output_dir
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)                  
    model = best_model.module if hasattr(best_model, "module") else best_model    

    torch.save(model.state_dict(), os.path.join(output_dir, "best_model.bin"))

def convert_batch_to_inputs(args, batch, masker):

    if args.mode == 'fine':
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {
                "input_ids" : batch[0], 
                "attention_mask" : batch[1], 
                "labels" : batch[2]}        
    elif args.mode == 'post':
        batch[0], masked_labels = masker.mask_tokens(batch[0], args.mlm_probability)        
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {
                "input_ids" : batch[0], 
                "attention_mask" : batch[1], 
                "labels" : masked_labels.to(args.device)}        

    elif args.mode == 'multi':
        batch[0], masked_labels = masker.mask_tokens(batch[0], args.mlm_probability)        
        batch = tuple(t.to(args.device) for t in batch)

        inputs = {
                "args" : args,
                "input_ids" : batch[0], 
                "attention_mask" : batch[1], 
                "labels" : batch[2],
                "masked_lm_labels" : masked_labels.to(args.device)}        

    return inputs

def default_parser(parser):
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mode", type=str, default='fine')
    parser.add_argument("--model", type=str, default='roberta-base')
    parser.add_argument("--model_name_or_path", type=str, default=None)

    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=512)

    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--mlm_probability", type=float, default=0.15)

    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--adam_epsilon", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--warmup_proportion", type=float, default=0.06)

    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--mlm_loss", type=float, default=0.2)

    parser.add_argument("--train_file", type=str, default="./data/")
    parser.add_argument("--dev_file", type=str, default="./data/")
    parser.add_argument("--devtest_file", type=str, default="./data/")

    parser.add_argument("--fashion_model_name_or_path", type=str, default=None)
    parser.add_argument("--furniture_model_name_or_path", type=str, default=None)


    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--augmented", action='store_true')
    parser.add_argument("--submission", action='store_true')

    parser.add_argument("--max_turns", type=int, default=5)

    parser.add_argument("--total_tokens", type=int, default=0)
    parser.add_argument("--voting_models", action="append")
    parser.add_argument("--voting_fashion_models", action="append")
    parser.add_argument("--voting_furniture_models", action="append")
    return parser