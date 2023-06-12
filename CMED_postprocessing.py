import math
from nltk.tag import StanfordTagger
import pandas as pd
import numpy as np
import argparse


def wordpiece_detokenize(wordpiece_tokens=[], labels = [] ):
    word_list = [];
    label_list = [];
    index = 0;
    for wordpiece, label in zip(wordpiece_tokens, labels):
        if(wordpiece == "[CLS]"): continue;
        elif(wordpiece == "[SEP]"):
            word_list.append(" ");
            label_list.append(" ");
            index +=1;
        elif wordpiece[:2] == "##":
            word_list[index-1] = word_list[index-1] + wordpiece[2:];
        else:
            word_list.append(wordpiece);
            label_list.append(label);
            index +=1;

    return word_list, label_list;

def Create_predict_annotation(data_doc_dir, tokens, tag_pred, output_dir, lowercase = False, Labels =["B-Drug", "I-Drug", "0"], task = True):

    Record_ID_Flag = "RecordID";
    if lowercase: Record_ID_Flag = Record_ID_Flag.lower();
    num_predictions = len(tokens);
    entity = "";
    ENTITY_FLAG = False;

    for token_index in range(num_predictions):
        #Check to see which record predictions are from
        if  type(tokens[token_index]) == str and  Record_ID_Flag in tokens[token_index]:
            ID_token = tokens[token_index];
            Record_ID = ID_token[8:11] + "-" + ID_token[11:];
            Record_path = data_doc_dir + Record_ID + ".txt";
            Record = open(Record_path, 'r').read();
            if lowercase:Record = Record.lower();

            Output_path = output_dir + Record_ID + ".ann";
            Output_ann = open(Output_path, 'w');

            Term_index = 1;
            token_record_start_index = 0;
            token_record_end_index = 0;
            continue;

        if "B-" in tag_pred[token_index]:
            entity = tokens[token_index];
            token_record_start_index = Record.index(tokens[token_index], token_record_end_index);
            token_record_end_index = token_record_start_index + len(tokens[token_index]);
            ENTITY_FLAG = True;

            if "I-" in tag_pred[token_index + 1]:
                continue;

        elif ("I-" in tag_pred[token_index]) and (ENTITY_FLAG==True):
            entity = entity + " " + tokens[token_index];
            token_record_end_index = token_record_end_index +  len(tokens[token_index]) + 1;
            if "I-" in tag_pred[token_index + 1]:
                continue;
        else: continue;

        Annotation = "T" + str(Term_index) + "\t" + "Drug " + str(token_record_start_index) + " " + str(token_record_end_index) + "\t" + entity + "\n";
        Output_ann.write(Annotation);
        if task == False:
            Annotation = "E" + str(Term_index) + "\t" + tag_pred[token_index][2:] + ":T" + str(Term_index) + "\n";
            Output_ann.write(Annotation);

        Term_index += 1;
        ENTITY_FLAG = False;

    return 0;

def CMED_Error_Detection(NER_result_colln_path, Output_dir = ""):


    results = open(NER_result_colln_path).readlines();
    Error_flag = False;
    Temp_list = [];
    Error_list = [];
    for line in results:
        line = line.replace("\n", "");
        line = line.replace("-MISC", "");
        line = line.split();
        if len(line) != 0 and ("RecordID" in line[0] or "recordid" in line[0]) :
            Record_ID = line;

        if len(line) == 0:
            if not Error_flag:
                Temp_list = [];
                continue;
            else:
                Error_flag = False;
                Error_list.append(Record_ID);
                Error_list.extend(Temp_list);
                Error_list.append([])
                Temp_list = [];
        else:
            if line[1] != line[2]:
                line.append("ERROR");
                Error_flag = True;
            Temp_list.append(line);
    Error_DF = pd.DataFrame(Error_list);
    Error_DF.to_csv(Output_dir + "Error_List.txt", sep= " ", index=False, header=False)
    return 0

def main():
    import os
    from eval_script import Corpora, evaluate
    TASK_FLAG = False;
    dir_list = os.listdir();
    Test_data_dir = "C:/Users/Tariq/OneDrive - Prairie View A&M University/PVAMU Research/PVAMU_Thesis_Project/PVAMU_Thesis/CMED/test/";
    if TASK_FLAG:
        Tags = ["B-Drug", "I-Drug", "O"];
        output_dir = "ann_predict_T1/";
        if output_dir.replace("/", "") not in dir_list:
            os.mkdir(output_dir.replace("/", ""));
    else:
        Tags = ["B-Disposition", "I-Disposition", "B-NoDisposition", "I-NoDisposition", "B-Undetermined", "I-Undetermined", "O"];
        output_dir = "ann_predict_T2/";
        if output_dir.replace("/", "") not in dir_list:
            os.mkdir(output_dir.replace("/", ""));

    Data = pd.read_csv("Results/ClinicalbertDisharge_results/Test_predict_T2.tsv",sep = "\t",header=None).to_numpy().T;
    words = Data[0]
    ner_tags = Data[1];
    labels = Tags.remove("O");
    Create_predict_annotation(Test_data_dir, words, ner_tags, output_dir, lowercase=True,
                                                  Labels=labels, task=TASK_FLAG);

    corpora = Corpora(Test_data_dir, output_dir);
    if corpora.docs:
        evaluate(corpora, verbose=False);



    return 0;



if __name__ == '__main__':
    main()