import pickle
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import os
import numpy as np
import string
import torch
import transformers
from transformers import AutoTokenizer
import pandas as pd
from nltk.corpus import stopwords
from torch.utils.data import DataLoader


def Parse_Annotation(filepath = ""):
    Tags = ["Disposition", "NoDisposition", "Undetermined", "Drug"];

    Annotation_file = open(filepath, "r");
    Annotation = Annotation_file.readlines();
    Annotation_file.close();

    Ann_List = [];
    remove_list = list(string.punctuation);

    start_index_list = [];
    for Ann in Annotation:
        Ann = nltk.word_tokenize(Ann);

        if (Ann[0][0] == "T") and (Ann[1] in Tags):
            # For multiple of the same annotation
            if Ann[2] in start_index_list:
                continue;
            else:
                start_index_list.append(Ann[2]);

            if len(Ann) > 5 and Ann.count(";")>0 and str.isnumeric(Ann[3]) and str.isnumeric(Ann[5]):
                #filter annotation
                for word in Ann[7:]:
                    if word in remove_list:
                        Ann[Ann.index(word)] = "_";
                Ann = ((int(Ann[2]), int(Ann[6])) , Ann[1] , "_".join(Ann[7:]));
                Ann_List.append(Ann);
            else:
                #filter annotation
                for word in Ann[4:]:
                    if word in remove_list:
                        Ann[Ann.index(word)] = "_";
                Ann = ((int(Ann[2]),int(Ann[3])), Ann[1],  '_'.join(Ann[4:]));
                Ann_List.append(Ann);

    #sort ann_list
    dtype = [('char_index', tuple), ("tag", list), ("drug", list)];
    Ann_List = np.array(Ann_List, dtype=dtype);
    Ann_List.sort(order = "char_index");
    #reverse order of annotation
    Ann_List = Ann_List[::-1];

    return Ann_List;

def Tokenize_EHR(file_path = "", Ann=[]):

    EHR_Tokenized = [];
    #Add token ID for use in post-processing stage
    EHR_ID = file_path[(len(file_path) - 10) :].replace(".txt", "")
    EHR_ID_Token = "RecordID" + EHR_ID.replace("-", "");
    EHR_Tokenized.append([EHR_ID_Token])
    EHR_txt_file = open(file_path, 'r');
    EHR_txt = EHR_txt_file.read();
    EHR_txt_file.close();
    #Add tags from annotation into EHR text with [TAG] flag seperating drug and tag
    if len(Ann) != 0:
        for ann in Ann:
            EHR_txt = EHR_txt[:ann[0][0]] + " " + ann[1] + "_TAG_" + ann[2] + " " + EHR_txt[ann[0][1]:];
    punctuation = [];
    EHR_txt = EHR_txt.replace("\n \n", "\n\n\n");

    EHR_Sections = EHR_txt.split("\n\n\n");

    #split section by sentances(sentace indicators: ., ?, !)
    EHR_total = 0;
    for section in EHR_Sections:
        if(section == ""): continue;
        else:
            sentances = nltk.sent_tokenize(section);
            # split sentances into tokens
            for sent in sentances:
                sent_tokens = nltk.word_tokenize(sent);
                EHR_Tokenized.append(sent_tokens);
    return EHR_Tokenized;
def Wordpiece_Tagger(wordpieces = [], Word_tag = ""):
    tags_list = [];
    tags_list.append(Word_tag);
    if Word_tag == "O":
        for wordpiece in wordpieces[1:]:
            tags_list.append("X");
    else:
        for wordpiece in wordpieces[1:]:
            #tags_list.append(Word_tag);
            tags_list.append("X");
    return tags_list;
def EHR_Embedding(EHR = [],  wordpiece_tokenizer=AutoTokenizer.from_pretrained("bert-base-cased"), tsv_filepath = "", max_token_length = 512, Tags = None):

    if Tags == None:
        Tags = {"Drug": ("B-Drug"), "IDrug": ("I-Drug"), "Disposition": ("B-Disposition"),
                "IDisposition": ("I-Disposition"), "NoDisposition": ("B-NoDisposition"),
                "INoDisposition": ("I-NoDisposition"), "Undetermined": ("B-Undetermined"),
                "IUndetermined": ("I-Undetermined"), "Outside": {"O"}};

    Tag_flag = "_TAG_";
    EHR_dict = {"EHR Tokens":EHR, "Wordpiece Tokens":[], "Wordpiece Tags":[],"input_ids":[]}

    CLS_TAGID = wordpiece_tokenizer.convert_tokens_to_ids("[CLS]")
    SEP_TAGID = wordpiece_tokenizer.convert_tokens_to_ids("[SEP]")
    PAD_TAGID = wordpiece_tokenizer.convert_tokens_to_ids("[PAD]")


    if(tsv_filepath != ""):
        EHR_taggedfile = open(tsv_filepath, "a");

    for sect in EHR:
        EHR_temp_tag_list = [];
        EHR_temp_wordpiece_list = []
        EHR_temp_wordpiece_tag_list = [];
        EHR_temp_input_ids = []

        #filter data here
        punctuation = list(string.punctuation);
        punctuation.remove("_");
        sect = " ".join(sect);
        for punct in punctuation:
            sect = sect.replace(punct, "_");
        sect = sect.split(" ");

        #Add cls token to beginning of each list
        EHR_temp_wordpiece_tag_list.extend(["[CLS]"]);
        EHR_temp_tag_list.extend(["[CLS]"]);
        EHR_temp_wordpiece_list.extend(["[CLS]"]);
        EHR_temp_input_ids.extend([CLS_TAGID]);
        for words in sect:
            if words == "" or words == "_":continue;
            if "___"in words and Tag_flag not in words:
                continue;
            elif "__" in  words:
                words = words.replace("__", "_");

            #Loop for tag flag in tokens
            if(words.find("_TAG_") != -1):
                words = words.split("_TAG_");
                Drug = words[1].split("_");
                EHR_temp_tag_list.append(Tags[words[0]]);
                wordpieces = wordpiece_tokenizer.tokenize(Drug[0]);
                input_ids = wordpiece_tokenizer.encode(Drug[0], add_special_tokens = False);
                EHR_temp_wordpiece_list.extend(wordpieces);
                EHR_temp_wordpiece_tag_list.extend(Wordpiece_Tagger(wordpieces,Tags[words[0]]));
                EHR_temp_input_ids.extend(input_ids);



                if(tsv_filepath != ""): EHR_taggedfile.write(Drug[0] + " "+Tags[words[0]]+ "\n");

                if len(Drug) > 1:
                    words[0] = "I" + words[0];
                    for elements in Drug[1:]:
                        if elements == "": continue;
                        EHR_temp_tag_list.append(Tags[words[0]]);
                        wordpieces = wordpiece_tokenizer.tokenize(elements);
                        input_ids = wordpiece_tokenizer.encode(elements, add_special_tokens = False);
                        EHR_temp_wordpiece_list.extend(wordpieces);
                        EHR_temp_wordpiece_tag_list.extend(Wordpiece_Tagger(wordpieces, Tags[words[0]]));
                        EHR_temp_input_ids.extend(input_ids);

                        if(tsv_filepath != ""):EHR_taggedfile.write(elements + " " + Tags[words[0]] + "\n");
            else:
                if words == "": continue;
                if("_" in words):
                    words = words.split("_");
                    for word in words:
                        if word == "":continue;
                        else:
                            EHR_temp_tag_list.append('O');
                            wordpieces = wordpiece_tokenizer.tokenize(word);
                            input_ids = wordpiece_tokenizer.encode(word, add_special_tokens = False)
                            EHR_temp_wordpiece_list.extend(wordpieces);
                            EHR_temp_wordpiece_tag_list.extend(Wordpiece_Tagger(wordpieces, 'O'));
                            EHR_temp_input_ids.extend(input_ids);

                            if (tsv_filepath != ""): EHR_taggedfile.write(word + " " + 'O' + "\n");
                else:
                    EHR_temp_tag_list.append('O');
                    wordpieces = wordpiece_tokenizer.tokenize(words);
                    input_ids = wordpiece_tokenizer.encode(words, add_special_tokens = False);
                    EHR_temp_wordpiece_list.extend(wordpieces);
                    EHR_temp_wordpiece_tag_list.extend(Wordpiece_Tagger(wordpieces, 'O'));
                    EHR_temp_input_ids.extend(input_ids);

                    if(tsv_filepath != ""):EHR_taggedfile.write(words + " " + 'O'+ "\n");

        if(tsv_filepath != ""):EHR_taggedfile.write("\n");

        #Divide sequneces that are longer than max sequence length
        if len(EHR_temp_wordpiece_list) > (max_token_length-1):
            while(len(EHR_temp_wordpiece_list) > (max_token_length-1)):
                EHR_temp_wordpiece_list_2 = EHR_temp_wordpiece_list[:max_token_length-1];
                EHR_temp_wordpiece_tag_list_2 = EHR_temp_wordpiece_tag_list[:max_token_length-1];
                EHR_temp_input_ids_2 = EHR_temp_input_ids[:max_token_length-1];
                #Add [SEP] token to end of each sequence
                EHR_temp_wordpiece_tag_list_2.extend(["[SEP]"]);
                EHR_temp_wordpiece_list_2.extend(["[SEP]"]);
                EHR_temp_input_ids_2.extend([SEP_TAGID]);
                EHR_dict["Wordpiece Tokens"].append(" ".join(EHR_temp_wordpiece_list_2));
                EHR_dict["Wordpiece Tags"].append(" ".join(EHR_temp_wordpiece_tag_list_2));
                #add padding to input_ids
                Padding_length = max_token_length - len(EHR_temp_input_ids_2);
                EHR_temp_input_ids_2.extend(list(np.zeros(Padding_length)));
                EHR_dict["input_ids"].append(EHR_temp_input_ids_2);
                #
                EHR_temp_wordpiece_list = EHR_temp_wordpiece_list[max_token_length-1:];
                EHR_temp_wordpiece_tag_list = EHR_temp_wordpiece_tag_list[max_token_length-1:];
                EHR_temp_input_ids = EHR_temp_input_ids[max_token_length-1:];
                EHR_temp_wordpiece_list.insert(0, "[CLS]");
                EHR_temp_wordpiece_tag_list.insert(0, "[CLS]");
                EHR_temp_input_ids.insert(0, CLS_TAGID);

        #Add [SEP] tag to end of each sequence
        EHR_temp_wordpiece_tag_list.extend(["[SEP]"]);
        EHR_temp_wordpiece_list.extend(["[SEP]"]);
        EHR_temp_input_ids.extend([SEP_TAGID]);
        EHR_dict["Wordpiece Tokens"].append(" ".join(EHR_temp_wordpiece_list));
        EHR_dict["Wordpiece Tags"].append(" ".join(EHR_temp_wordpiece_tag_list));
        #add padding to input ids
        Padding_length = max_token_length -  len(EHR_temp_input_ids);

        EHR_temp_input_ids.extend(list(np.zeros(Padding_length)));
        EHR_dict["input_ids"].append(EHR_temp_input_ids);

    if(tsv_filepath != ""): EHR_taggedfile.close();

    return EHR_dict;
def Generate_cmed_data(Data_dir = str, Output_dir = "", tag = True,wordpiece_tokenizer =AutoTokenizer.from_pretrained("bert-base-cased")  , removal_list = [], tsv_filepath = "", max_token_length = 256, Tags = None):
    files_list = os.listdir(Data_dir);
    EHR_taggedfile = open(tsv_filepath, "w");
    EHR_taggedfile.close();

    cmed_data_dict = {"Wordpiece Tokens": [], "Wordpiece Tags": [], "input_ids": []};
    Data_files_dict = {};
    for files in files_list:
       Data_files_dict[files[:6]] = None;
    for IDs in Data_files_dict:
        if tag:
            Ann_path = Data_dir + IDs + ".ann";
            Ann = Parse_Annotation(Ann_path);
        else: Ann = [];
        EHR_path = Data_dir + IDs + ".txt";
        Tokens = Tokenize_EHR(EHR_path, Ann);
        Embedding_dict = EHR_Embedding(EHR=Tokens, wordpiece_tokenizer=wordpiece_tokenizer, tsv_filepath=tsv_filepath, max_token_length=max_token_length, Tags=Tags);

        for key in cmed_data_dict.keys():
                cmed_data_dict[key].extend(Embedding_dict[key]);
    return cmed_data_dict;


