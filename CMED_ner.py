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

import CMED_preprocessing
import CMED_Modeling_PT
from CMED_Modeling_PT import CMED_Model
import CMED_postprocessing
from eval_script import Corpora
from eval_script import evaluate


def main():
    #Model encoder links
    Transformer_Encoder_Link = "bert-base-cased"
    #Transformer_Encoder_Link = "dmis-lab/biobert-bert-cased-v1.2";
    #Transformer_Encoder_Link = "emilyalsentzer/Bio_ClinicalBERT";
    #Transformer_Encoder_Link = "emilyalsentzer/Bio_Discharge_Summary_BERT";
    #Transformer_Encoder_Link = "Charagan/MedBERT";


    #Flag using for now to select which task to do from n2c2 Track 1(Task 1: Drug Detection, Task 2: Drug Event CLassification)
    #True = Task1 , False = Task2
    TASK_FLAG = True;

    GENERATE_DATA_FLAG = False;
    TRAIN_FLAG = False;
    TRAIN_PLUS_EVAL_FLAG = True;
    TRAIN_Continue = False;
    TRAIN_Debug = False;
    EVAL_FLAG = False;
    TEST_FLAG = True;
    PREDICT_FLAG = False;

    current_dir = os.getcwd();

    current_dir = current_dir.replace("\\", "/");
    Data_dir = current_dir + "/CMED/";
    Training_data_dir = Data_dir + "train/";
    Eval_data_dir = Data_dir + "eval/";
    Test_data_dir = Data_dir + "test/";
    Trainpluseval_data_dir = Data_dir + "train_plus_eval/";

    if TASK_FLAG:
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
        
        Tags = ["B-Drug", "I-Drug", "O"];

    else:
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
        
        Tags = ["B-Disposition", "I-Disposition","B-NoDisposition" ,"I-NoDisposition" ,"B-Undetermined","I-Undetermined" ,  "O"]


    Max_Embedding_length = 512;
    Train_Epochs = 10;
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


        train_dataloader = CMED_Modeling_PT.CMED_PT_Dataset(torch.tensor(Train_CMED_dict["input_ids"]), Train_CMED_dict["Wordpiece Tags"], batch_sz=32);
        Model = CMED_Model(bert_model=bert_model, Task = TASK_FLAG);
        Model = torch.nn.DataParallel(Model, device_ids=[0, 1, 2, 3]);
        Model = Model.to(device);
        if TRAIN_Continue: 
            Model = torch.load(Model_Path);
        Model, Train_History = CMED_Modeling_PT.Train_Model(train_dataloader=train_dataloader, model=Model ,Num_Epochs=Train_Epochs,Device=device, Model_Path = Model_Path,History_Path = Train_History_Path , debug=TRAIN_Debug, Tags = Tags);

    if EVAL_FLAG:
        Eval_CMED_dict = torch.load(eval_torch_file);
        eval_dataloader = CMED_Modeling_PT.CMED_PT_Dataset(torch.tensor(Eval_CMED_dict["input_ids"]), Eval_CMED_dict["Wordpiece Tags"], batch_sz=32);
        Model = CMED_Model(bert_model=bert_model, Task = TASK_FLAG);
        Model = torch.nn.DataParallel(Model, device_ids=[0, 1, 2, 3]);
        Model = Model.to(device);
        Model = torch.load(Model_Path);
        CMED_Modeling_PT.Eval_Model(eval_dataloader=eval_dataloader,model=Model, Device=device, Tags = Tags);

    if TEST_FLAG:
        Test_CMED_dict = torch.load(test_torch_file);
        #test_dataloader = CMED_Modeling_PT.CMED_PT_Dataset(torch.tensor(Test_CMED_dict["input_ids"]), Test_CMED_dict["Wordpiece Tags"], batch_sz=32);
        Model = CMED_Model(bert_model=bert_model, Task = TASK_FLAG);
        Model = torch.nn.DataParallel(Model, device_ids=[0, 1, 2, 3]);
        Model = Model.to(device);
        Model = torch.load(Model_Path);

        True_wordpiece_tags = " ".join(Test_CMED_dict["Wordpiece Tags"]);
        True_wordpiece_tags = True_wordpiece_tags.split(" ");

        wordpieces, ner_tags = CMED_Modeling_PT.Test_Model(test_xdata=torch.tensor(Test_CMED_dict["input_ids"]), test_xdata_wordpieces = Test_CMED_dict["Wordpiece Tokens"],model=Model,tags = Tags ,Device=device);

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

        labels = Tags.remove("O");
        CMED_postprocessing.Create_predict_annotation(Test_data_dir, words, ner_tags, output_dir, lowercase = True, Labels = labels, task = TASK_FLAG);

        corpora = Corpora(Test_data_dir, output_dir);
        if corpora.docs:
            evaluate(corpora, verbose=False);
        

    if PREDICT_FLAG:
        predict_input = "The patient is on Codine for medication";
        bert_model = transformers.AutoModel.from_pretrained(Transformer_Encoder_Link);
        Model = CMED_Model(bert_model=bert_model, Task = TASK_FLAG);
        Model = torch.load(Model_Path);
        Model = Model.to(device);

        words, tags = CMED_Modeling_PT.Model_predict(input=predict_input, bert_tokenizer=bert_tokenizer, model=Model,
                                                     Device=device);
        predict_df = pd.DataFrame([words, tags]);
        predict_df = predict_df.transpose();
        predict_df.to_csv("CMED_wordpiece_predict.tsv", header=None, index=False, sep="\t")

        words, tags = CMED_postprocessing.wordpiece_detokenize(words, tags);
        predict_df = pd.DataFrame([words, tags]);
        predict_df = predict_df.transpose();
        predict_df.to_csv("CMED_predict.tsv", header=None, index=False, sep="\t");

    return 0;


if __name__ == '__main__':
    main();

