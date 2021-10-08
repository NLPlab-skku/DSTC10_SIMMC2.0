import json
import gzip
import copy
import pickle
from util import *
import itertools
import warnings
warnings.filterwarnings('ignore')
today = str(datetime.datetime.today().strftime("%Y%m%d"))




def _post_step(outputs):
    pass

def train(args = None, model = None, train_dataloader = None, dev_dataloader = None, train_masker = None, eval_masker = None):

    t_total = len(train_dataloader)//args.gradient_accumulation_steps * args.epochs
    no_decay = {"bias","LayerNorm.weight"}
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(params = optimizer_grouped_parameters, lr = args.learning_rate, eps = args.adam_epsilon)    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = int(t_total * args.warmup_proportion), num_training_steps = t_total)    

    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^14} | {'Elapsed':^9}")
    print("-"*60)

    import os

    if not args.mode == 'post': output_dir = "./save_model/{}_mode{}_seed{}_lr{}_bs{}_ac{}_len{}_epoch{}".format(today, args.mode, args.seed,args.learning_rate, args.batch_size, args.gradient_accumulation_steps, args.max_seq_length,args.epochs)
    if args.mode == 'post': output_dir = "./save_model/post/{}_mode{}_seed{}_lr{}_bs{}_ac{}_len{}_epoch{}".format(today, args.mode, args.seed,args.learning_rate, args.batch_size, args.gradient_accumulation_steps, args.max_seq_length,args.epochs)

    os.makedirs(output_dir, exist_ok=True)
    output_dir += "/dev.txt"
    save_ = open(output_dir, mode="w", encoding="UTF-8-sig", errors="ignore")
    save_.write("Epoch\t\tTrain_loss\t\tVal_loss\t\tACC\n")
    save_.close()

    model.zero_grad()            
    best_loss = float('inf')
    best_acc = 0 if not args.mode == 'post' else float('inf')
    best_model = None
    dev_acc = list()
    dev_loss = list()
    print("\nEpochs 시작")
    for epoch_i in trange(args.epochs, leave=False):
        t0_epoch = time.time()
        model.train()            

        total_loss = 0


        for step, batch in enumerate(tqdm(train_dataloader, leave=False)):        
            inputs = convert_batch_to_inputs(args=args, batch=batch, masker=train_masker)
            outputs = model(**inputs)

            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            total_loss += loss.detach().cpu().numpy().item()        
            loss.backward()
            del loss
            if not args.mode == 'fine': _post_step(outputs)

            # if args.mode == 'post' or args.mode== 'fine' and args.mlm: _post_step(outputs)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()                            
                optimizer.zero_grad()                    
                torch.cuda.empty_cache()

        train_loss = total_loss / len(train_dataloader)         

        if dev_dataloader is not None: 

            time_elapsed = time.time() - t0_epoch

            loss, acc =  evaluate(args=args, model=model, dev_dataloader=dev_dataloader, eval_masker=eval_masker)

            if (args.mode == 'post' and best_acc > acc) or\
                (not args.mode == 'post' and best_acc < acc) or\
                (best_acc == acc and best_loss > loss):                
                best_epoch = epoch_i + 1
                best_model = copy.deepcopy(model)
                best_acc = acc
                best_loss = loss
                save_checkpoint(args, best_model)

            print(f"{epoch_i + 1:^7} | {train_loss:^12.6f} | {loss:^10.6f} | {acc:^14.5f} | {time_elapsed:^9.2f}")

            if not args.mode == 'post': output_dir = "./save_model/{}_mode{}_seed{}_lr{}_bs{}_ac{}_len{}_epoch{}/dev.txt".format(today, args.mode, args.seed,args.learning_rate, args.batch_size, args.gradient_accumulation_steps, args.max_seq_length,args.epochs)
            if args.mode == 'post': output_dir = "./save_model/post/{}_mode{}_seed{}_lr{}_bs{}_ac{}_len{}_epoch{}/dev.txt".format(today, args.mode, args.seed,args.learning_rate, args.batch_size, args.gradient_accumulation_steps, args.max_seq_length,args.epochs)

            save_ = open(output_dir, mode="a", encoding="UTF-8-sig", errors="ignore")
            save_.write("{0}\t\t{1}\t\t{2}\t\t{3}\n".format(epoch_i,round(train_loss,5),loss,acc))
            dev_acc.append(str(acc))
            dev_loss.append(str(loss))
            save_.close()

    print("\nTraining complete!")
    print(f"Best model epoch: {best_epoch}") 

    if not args.mode == 'post': output_dir = "./save_model/{}_mode{}_seed{}_lr{}_bs{}_ac{}_len{}_epoch{}/dev.txt".format(today, args.mode, args.seed,args.learning_rate, args.batch_size, args.gradient_accumulation_steps, args.max_seq_length,args.epochs)
    if args.mode == 'post': output_dir = "./save_model/post/{}_mode{}_seed{}_lr{}_bs{}_ac{}_len{}_epoch{}/dev.txt".format(today, args.mode, args.seed,args.learning_rate, args.batch_size, args.gradient_accumulation_steps, args.max_seq_length,args.epochs)

    save_ = open(output_dir, mode="a", encoding="UTF-8-sig", errors="ignore")
    acc = ("\t").join(dev_acc) + "\n"
    loss = ("\t").join(dev_loss) + "\n"
    save_.write("Dev Acc : {}".format(acc))
    save_.write("Dev Loss : {}".format(loss))
    save_.close()

    return best_model

def evaluate(args=None, model=None, dev_dataloader=None, eval_masker=None, test=False):
    model.eval()      
    preds_list = list()
    labels_list = list()    
    dialog_ids =list()
    turn_ids = list()

    val_accuracy = 0    
    accuracy = 0
    eval_loss = 0       
    n_true = 0
    n_pred = 0
    n_correct = 0

    model = get_model(args, model)
    for step, batch in enumerate(tqdm(dev_dataloader, desc="Evaluate : ", leave=False)):
        inputs = convert_batch_to_inputs(args=args, batch=batch, masker=eval_masker)
        if not (args.mode == 'post' or args.mode=='fine'):    
            del inputs['args']
            del inputs['masked_lm_labels']
            
        with torch.no_grad():
            if args.voting_models is not None:
                outputs = list()
                for num in range(len(args.voting_models)):
                    outputs.append(model[num](**inputs)[1])

            elif args.fashion_model_name_or_path is not None and\
                args.furniture_model_name_or_path is not None:
                domain = batch[3].tolist()[0]
                outputs = model[domain](**inputs)
            else:
                outputs = model(**inputs)


        if args.mode == 'post': eval_loss += outputs[0].mean().item()
        elif not args.submission:
            if args.voting_models is not None:
                outputs = torch.stack(outputs, dim=1)
                preds = torch.argmax(torch.tensor(outputs[0]), dim=1).flatten()
                preds = preds.detach().cpu().numpy().tolist()
                zero = preds.count(0)
                one  = preds.count(1)
                if not one == len(args.voting_models) and not zero == len(args.voting_models):
                    print("0 : {} / 1 : {} / 다름 : {}".format(zero, one, preds))
                if one > zero:      preds_list.append(1)            
                elif one < zero:    preds_list.append(0)            
                elif one == zero:    print("예외 발생")            

            else:
                preds = torch.argmax(outputs[1], dim=1).flatten()
                preds_list.extend(p.item() for p in preds)            
            labels_list.extend(l.item() for l in inputs['labels'])

        elif args.submission:
                preds = torch.argmax(outputs[1], dim=1).flatten()
                dialog_id = batch[4].detach().cpu().numpy().tolist()[0]
                turn_id = batch[5].detach().cpu().numpy().tolist()[0]
                preds_list.extend(p.item() for p in preds)            
                dialog_ids.append(dialog_id)
                turn_ids.append(turn_id)
        # if step > 10: break


    if not args.submission:
        accuracy = accuracy_score(labels_list, preds_list) * 100        


    if args.mode == 'post': 
        output_dir = "./save_model/post/{}_mode{}_seed{}_lr{}_bs{}_ac{}_len{}_epoch{}".format(today, args.mode, args.seed,args.learning_rate, args.batch_size, args.gradient_accumulation_steps, args.max_seq_length,args.epochs)
        eval_loss = eval_loss/(step+1)
        perplexity = torch.exp(torch.tensor(eval_loss))  
        accuracy = perplexity

    else:        
        output_dir = "./save_model/{}_mode{}_seed{}_lr{}_bs{}_ac{}_len{}_epoch{}".format(today, args.mode, args.seed,args.learning_rate, args.batch_size, args.gradient_accumulation_steps, args.max_seq_length,args.epochs)

    final_result_dir = output_dir + "/final_preds.json"

    if test and not args.submission:
        os.makedirs(output_dir, exist_ok=True)
        output_dir += "/eval.txt"

        if not args.mode == 'post' and not args.submission:

            print("\nVAL TEST ACC : {}".format(accuracy))
            print("\naccuracy_score \t:", accuracy)
            print("Precision \t:", precision_score(labels_list, preds_list, average='macro') * 100)
            print("Recall \t:", recall_score(labels_list, preds_list, average='macro') * 100)
            print("F1 \t:", f1_score(labels_list, preds_list, average='macro') * 100, "\n")

            correct = 0
            for l,p in zip(labels_list,preds_list):
                if l==p: correct+=1
            print("맞춘 갯수 \t: {}/{}".format(correct, len(preds_list)))
            save_ = open(output_dir, mode="w", encoding="UTF-8-sig", errors="ignore")
            save_.write("Accuracy\t:\t" + str(accuracy_score(labels_list, preds_list) * 100) + "\n")
            save_.write("Precision\t:\t" + str(precision_score(labels_list, preds_list, average='macro') * 100) + "\n")
            save_.write("Recall\t:\t" + str(recall_score(labels_list, preds_list, average='macro') * 100) + "\n")            
            save_.write("F1 Score\t:\t" + str(f1_score(labels_list, preds_list, average='macro') * 100) + "\n")                
            save_.write("맞춘 갯수\t:{}({})\t".format(str(correct), str(len(preds_list))))                
            save_.close()

        elif args.mode == 'post' and not submission:
            print("\nPerplexity: {}".format(perplexity))
            save_ = open(output_dir, mode="w", encoding="UTF-8-sig", errors="ignore")
            save_.write("Perplexity\t:\t" + str(perplexity) + "\n")
            save_.close()

    elif args.submission:
        final_output = list()
        prev_d_id = None
        print("preds_list : {}".format(preds_list))
        for pred, d_id, t_id in zip(preds_list, dialog_ids, turn_ids):

            if prev_d_id is None:
                prev_d_id = d_id
                d = {"dialog_id":d_id, "predictions":[]}

            elif not prev_d_id == d_id:
                final_output.append(copy.deepcopy(d))
                d = {"dialog_id":d_id, "predictions":[]}
                prev_d_id = d_id

            d['predictions'].append({'turn_id':t_id, 'disambiguation_label':pred})

        # final_output.append(copy.deepcopy(d))        
        with open(final_result_dir, mode='w') as out:
            json.dump(final_output, out, indent=2)
            print("저장 완료")

    return eval_loss, accuracy


if __name__ == "__main__":
    parser = default_parser(argparse.ArgumentParser())
    args = parser.parse_args()

    seed_everything(args.seed)
    set_device(args)


    tokenizer = RobertaTokenizerFast.from_pretrained(args.model)
    special_tokens_dict = {'additional_special_tokens': ['<USER>','<SYS>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    # print("\n{}개 신규 토큰 추가".format(num_added_toks))
    config = AutoConfig.from_pretrained(args.model)
    config.num_labels = 2


    # 데이터 불러오기
    tokenized_train, label_train, domains_train, _, _                           = tokenize(args, args.train_file, tokenizer, augmented=args.augmented)
    tokenized_dev, label_dev, domains_dev, _, _                                 = tokenize(args, args.dev_file, tokenizer)
    tokenized_test, label_test, domains_test, dialog_ids_test, turn_ids_test    = tokenize(args, args.devtest_file, tokenizer)

    print("\n========== [ Dataset ] ==========")
    print("Train : {}".format(len(tokenized_train['input_ids'])))
    print("Dev : {}".format(len(tokenized_dev['input_ids'])))
    print("Test : {}".format(len(tokenized_test['input_ids'])))

    train_dataloader = idx_to_tensor(args, tokenized_train, label_train, bsz=args.batch_size, train=True)
    dev_dataloader = idx_to_tensor(args, tokenized_dev, label_dev, bsz=args.batch_size, train=False)
    if args.submission:     test_dataloader = idx_to_tensor(args, tokenized_test, label_test, bsz=1, domain=domains_test, dialog_ids=dialog_ids_test, turn_ids=turn_ids_test, train=False)
    else:                   test_dataloader = idx_to_tensor(args, tokenized_test, label_test, bsz=1, domain=domains_test, train=False)

    model = MODEL_CLASSES[args.mode].from_pretrained(pretrained_model_name_or_path=args.model, config=config)
    model.resize_token_embeddings(len(tokenizer))
    if args.model_name_or_path:
        print("Load trained Model : {}".format(args.model_name_or_path))
        model = MODEL_CLASSES[args.mode].from_pretrained(pretrained_model_name_or_path="{}/best_model.bin".format(args.model_name_or_path), config=config)                
    model.to(args.device)        

    train_masker = get_masker(args, tokenizer)
    eval_masker = get_masker(args, tokenizer)

    if args.do_train and not args.submission:
        model = train(args, model, train_dataloader, dev_dataloader, train_masker, eval_masker)

    if args.do_eval:
        print("\n\n******** Test *********")
        evaluate(args=args, model=model, dev_dataloader=test_dataloader, eval_masker=eval_masker, test=True)