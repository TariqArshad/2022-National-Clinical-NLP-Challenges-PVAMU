import pandas as pd
import transformers
from transformers import AutoTokenizer
from transformers import AutoModel
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn import preprocessing
from sklearn import metrics
import numpy as np
import pickle
import os
import argparse

import CMED_preprocessing
import CMED_Modeling_PT
from CMED_Modeling_PT import CMED_Model
import CMED_postprocessing
from eval_script import Corpora
from eval_script import evaluate

def Parse_Args():
    parser = argparse.ArgumentParser();
    parser.add_argument("--encoder_link", type=str, default = "emilyalsentzer/Bio_ClinicalBERT");
    parser.add_argument("--task", type=int,choices = [1, 2] , default = 1);
    parser.add_argument("--generate_data",choices = ["True", "False"] , default = "True");
    parser.add_argument("--train_on_eval", choices = ["True", "False"], default = "False");
    parser.add_argument("--train", choices = ["True", "False"], default = "True");
    parser.add_argument("--eval", choices = ["True", "False"], default = "True");
    parser.add_argument("--test", choices = ["True", "False"], default = "True");
    parser.add_argument("--continue_training", choices = ["True", "False"], default = "False");
    parser.add_argument("--train_debug", choices=["True", "False"], default = "False");
    parser.add_argument("--data_dir", type=str, default = "CMED/");
    parser.add_argument("--max_seq_length", type=int, default = 512);
    parser.add_argument("--num_train_epochs", type=int, default = 10);
    parser.add_argument("--batch_sz", type=int, default = 32);
    parser.add_argument("--lowercase", choices = ["True", "False"], default = "True");
    args = parser.parse_args();
    return args;

def main(args):
    """
    Encoder Link Examples:
    Transformer_Encoder_Link = "bert-base-cased"
    Transformer_Encoder_Link = "dmis-lab/biobert-bert-cased-v1.2"
    Transformer_Encoder_Link = "emilyalsentzer/Bio_ClinicalBERT"
    Transformer_Encoder_Link = "emilyalsentzer/Bio_Discharge_Summary_BERT"
    """
    Transformer_Encoder_Link = args.encoder_link;
    #Flag using for now to select which task to do from n2c2 Track 1(Task 1: Drug Detection, Task 2: Drug Event CLassification)
    #1 = Task1 , 2 = Task2
    TASK_FLAG = args.task

    if args.generate_data == "True": GENERATE_DATA_FLAG = True;
    else: GENERATE_DATA_FLAG = False;

    if args.train == "True": TRAIN_FLAG = True;
    else: TRAIN_FLAG = False;

    if args.train_on_eval == "True": TRAIN_PLUS_EVAL_FLAG = True;
    else: TRAIN_PLUS_EVAL_FLAG = False;

    if args.continue_training == "True": TRAIN_Continue = True;
    else: TRAIN_Continue = False;
    
    if args.train_debug == "True":TRAIN_Debug = True;
    else: TRAIN_Debug = False;
    
    if args.eval == "True":EVAL_FLAG = True;
    else: EVAL_FLAG = False;
    
    if args.test == "True":TEST_FLAG = True
    else: TEST_FLAG = False;

    if args.lowercase == "True": lowercase = True;
    else: lowercase = False;
    #PREDICT_FLAG = args.predict;

    current_dir = os.getcwd();

    current_dir = current_dir.replace("\\", "/");
    Data_dir = args.data_dir;
    Training_data_dir = Data_dir + "train/";
    Eval_data_dir = Data_dir + "eval/";
    Test_data_dir = Data_dir + "test/";
    Trainpluseval_data_dir = Data_dir + "train_plus_eval/";

    if TASK_FLAG == 1:
        train_tsv_filepath = "CMED_Train_T1.tsv";
        eval_tsv_filepath = "CMED_Eval_T1.tsv";
        test_tsv_filepath = "CMED_Test_T1.tsv";
        trainpluseval_tsv_filepath = "CMED_trainplusEval_T1.tsv";

        train_torch_file = "CMED_train_dict_T1.pt";
        eval_torch_file = "CMED_eval_dict_T1.pt";
        test_torch_file = "CMED_test_dict_T1.pt";
        trainpluseval_torch_file = "CMED_trainpluseval_dict_T1.pt";

        Model_Path = "CMED_Model_T1.pt"
        Train_History_Path = "Model_Train_History_T1.pt"

        Tags_dict = {"Drug": ("B-Drug"), "IDrug": ("I-Drug"), "Disposition": ("B-Drug"), "IDisposition": ("I-Drug"),
                "NoDisposition": ("B-Drug"),
                "INoDisposition": ("I-Drug"), "Undetermined": ("B-Drug"), "IUndetermined": ("I-Drug"),
                "Outside": ("O")};
        
        tags = ["B-Drug", "I-Drug", "O"];

    elif TASK_FLAG == 2:
        train_tsv_filepath = "CMED_Train_T2.tsv";
        eval_tsv_filepath = "CMED_Eval_T2.tsv";
        test_tsv_filepath = "CMED_Test_T2.tsv";
        trainpluseval_tsv_filepath = "CMED_trainplusEval_T2.tsv";

        train_torch_file = "CMED_train_dict_T2.pt";
        eval_torch_file = "CMED_eval_dict_T2.pt";
        test_torch_file = "CMED_test_dict_T2.pt";
        trainpluseval_torch_file = "CMED_trainpluseval_dict_T2.pt";

        Model_Path = "CMED_Model_T2.pt"
        Train_History_Path = "Model_Train_History_T2.pt"

        Tags_dict = {"Drug": ("B-Drug"), "IDrug": ("I-Drug"), "Disposition": ("B-Disposition"),
                "IDisposition": ("I-Disposition"), "NoDisposition": ("B-NoDisposition"),
                "INoDisposition": ("I-NoDisposition"), "Undetermined": ("B-Undetermined"),
                "IUndetermined": ("I-Undetermined"), "Outside": ("O")};
        
        tags = ["B-Disposition", "I-Disposition","B-NoDisposition" ,"I-NoDisposition" ,"B-Undetermined","I-Undetermined" ,  "O"]


    Max_Embedding_length = args.max_seq_length;
    Train_Epochs = args.num_train_epochs;
    batch_sz = args.batch_sz
    bert_tokenizer = transformers.AutoTokenizer.from_pretrained(Transformer_Encoder_Link);
    bert_model = transformers.AutoModel.from_pretrained(Transformer_Encoder_Link);

    if torch.cuda.is_available():
        device = torch.device("cuda:0");
    else:
        device = torch.cpu;

    if GENERATE_DATA_FLAG:
        train_data_dict = CMED_preprocessing.Generate_cmed_data(Data_dir=Training_data_dir, wordpiece_tokenizer=bert_tokenizer, tsv_filepath=train_tsv_filepath, max_token_length = Max_Embedding_length, Tags = Tags_dict);
        print("Train Data Generated!!!");

        eval_data_dict = CMED_preprocessing.Generate_cmed_data(Data_dir=Eval_data_dir, wordpiece_tokenizer=bert_tokenizer, tsv_filepath=eval_tsv_filepath, max_token_length = Max_Embedding_length, Tags = Tags_dict);
        print("Eval Data Generated!!!");

        test_data_dict = CMED_preprocessing.Generate_cmed_data(Data_dir=Test_data_dir, wordpiece_tokenizer = bert_tokenizer, tsv_filepath=test_tsv_filepath, max_token_length = Max_Embedding_length, Tags = Tags_dict);
        print("Test Data Generated!!!");

        if TRAIN_PLUS_EVAL_FLAG:
            trainpluseval_data_dict = CMED_preprocessing.Generate_cmed_data(Data_dir=Trainpluseval_data_dir, wordpiece_tokenizer=bert_tokenizer, tsv_filepath=trainpluseval_tsv_filepath, max_token_length = Max_Embedding_length, Tags = Tags_dict);
            torch.save(trainpluseval_data_dict, trainpluseval_torch_file)

        torch.save(train_data_dict, train_torch_file);
        torch.save(eval_data_dict, eval_torch_file);
        torch.save(test_data_dict, test_torch_file);


    if TRAIN_FLAG:
        if TRAIN_PLUS_EVAL_FLAG:
            Train_CMED_dict = torch.load(trainpluseval_torch_file);
        else:
            Train_CMED_dict = torch.load(train_torch_file);


        train_dataloader = CMED_Modeling_PT.CMED_PT_Dataset(torch.tensor(Train_CMED_dict["input_ids"]), Train_CMED_dict["Wordpiece Tags"], batch_sz=batch_sz);
        Model = CMED_Model(bert_model=bert_model, Task = TASK_FLAG);
        Model = Model.to(device);
        if TRAIN_Continue: 
            Model = torch.load(Model_Path);
        Model, Train_History = CMED_Modeling_PT.Train_Model(train_dataloader=train_dataloader, model=Model ,Num_Epochs=Train_Epochs,Device=device, Model_Path = Model_Path, debug=TRAIN_Debug, tags = tags, bert_tokenizer = bert_tokenizer);

    if EVAL_FLAG:
        Eval_CMED_dict = torch.load(eval_torch_file);
        eval_dataloader = CMED_Modeling_PT.CMED_PT_Dataset(torch.tensor(Eval_CMED_dict["input_ids"]), Eval_CMED_dict["Wordpiece Tags"], batch_sz=batch_sz);
        Model = CMED_Model(bert_model=bert_model, Task = TASK_FLAG);
        Model = Model.to(device);
        Model = torch.load(Model_Path);
        CMED_Modeling_PT.Eval_Model(eval_dataloader=eval_dataloader,model=Model, Device=device, tags = tags, bert_tokenizer = bert_tokenizer);

    if TEST_FLAG:
        Test_CMED_dict = torch.load(test_torch_file);
        #test_dataloader = CMED_Modeling_PT.CMED_PT_Dataset(torch.tensor(Test_CMED_dict["input_ids"]), Test_CMED_dict["Wordpiece Tags"], batch_sz=32);
        Model = CMED_Model(bert_model=bert_model, Task = TASK_FLAG);
        Model = Model.to(device);
        Model = torch.load(Model_Path);

        True_wordpiece_tags = " ".join(Test_CMED_dict["Wordpiece Tags"]);
        True_wordpiece_tags = True_wordpiece_tags.split(" ");

        wordpieces, ner_tags = CMED_Modeling_PT.Test_Model(test_xdata=torch.tensor(Test_CMED_dict["input_ids"]), test_xdata_wordpieces = Test_CMED_dict["Wordpiece Tokens"],model=Model,tags = tags ,Device=device, bert_tokenizer = bert_tokenizer);

        predict_df_wp = pd.DataFrame([wordpieces, ner_tags, True_wordpiece_tags]);

        if TASK_FLAG: predict_df_wp.T.to_csv("Test_predict_wp_T1.tsv", sep="\t", index=None, header=None);
        else: predict_df_wp.T.to_csv("Test_predict_wp_T2.tsv", sep="\t", index=None, header=None);

        words, true_ner_tags = CMED_postprocessing.wordpiece_detokenize(wordpieces,True_wordpiece_tags);
        words, ner_tags = CMED_postprocessing.wordpiece_detokenize(wordpieces, ner_tags)


        predict_df = pd.DataFrame([words, ner_tags, true_ner_tags]);

        if TASK_FLAG: predict_df.T.to_csv("Test_predict_T1.tsv", sep= "\t", index=None, header=None);
        else: predict_df.T.to_csv("Test_predict_T2.tsv", sep= "\t", index=None, header=None);

        dir_list = os.listdir();
        if TASK_FLAG:
            output_dir = "ann_predict_T1/";
            if output_dir.replace("/", "") not in dir_list:
                os.mkdir(output_dir.replace("/", ""));
        else:
            output_dir = "ann_predict_T2/";
            if output_dir.replace("/", "") not in dir_list:
                os.mkdir(output_dir.replace("/", ""));

        labels = tags.remove("O");
        CMED_postprocessing.Create_predict_annotation(Test_data_dir, words, ner_tags, output_dir, lowercase = lowercase, Labels = labels, task = TASK_FLAG);

        corpora = Corpora(Test_data_dir, output_dir);
        if corpora.docs:
            evaluate(corpora, verbose=False);

    return 0;


if __name__ == '__main__':
    args = Parse_Args();
    main(args);

