# Electronic Health Record Modeling For Medication Extraction

## How to run
###Requirements
####Pre & Post Processing
- Python 3
####Modeling
- Python 2
- Tensorflow 1
###Pre-Processing
- To Generate Training & Testing Data
    ```
    CMED_preprocessing.py
        --max_seq_length 128
        --Data_dir ~/n2c2_2022_project/CMED/
        --Output_dir ~/n2c2_2022_project/Outputs/
   ```
- CMED_preprocessing.py all parameters:
  - max_seq_length: Max number of words in a sequence
  - Data_dir: Directory with train & test clinical notes & annotated notes
  - Remove_stopwords: Flag for if to remove stopwords from training & test data based on training data word distribution
  - Remove_numbers: Flag for if to remove numbers from training & test data
  - Ouput_dir: Directory to place processed training & test data  
###Modeling
- To Run Training & Testing
  ```
  run_ner_PVAMU.py 
      --output_dir ~/n2c2_2022_project/Outputs
      --data_dir ~/n2c2_2022_project/Outputs
      --do_train True
      --max_seq_length 425
      --num_train_epochs 10
      --bert_config_file ~/n2c2_2022_project/Modeling/BioBert/bert_config.json
      --vocab_file ~/n2c2_2022_project/Modeling/BioBert/vocab.txt
      --checkpoint ~/n2c2_2022_project/Modeling/BioBert/model.ckpt-1000000
      --do_lower_case True    
  ```
- run_ner_PVAMU.py all parameters:
  - output_dir: Directory to place all model output 
  - data_dir: Directory with processed training & test data
  - do_train: Flag for model to do training, do not add argument if testing 
  - do_eval: Flag for model to do evaluation, output predictions not given
  - do_test: Flag for model to do testing and give word piece prediction output, do not add argument if training
  - max_seq_length: max word piece length
  - num_train_epochs: number of epochs for training
  - bert_config_file: json file with Bert model parameters
  - vocab_file: text file with model word piece vocabulary
  - checkpoint: file containing pre-trained models checkpoint
  - do_lower_case: flag for if input data to model should be lowercase
###Post-Processing
- To get prediction outputs in annotated format
  ```
  CMED_postprocessing.py 
      --prediction_dir ~/n2c2_2022_project/Outputs/ 
      --Test_data_dir ~/n2c2_2022_project/CMED/dev/ 
  ```
- CMED_postprocessing.py all parameters:
  - prediction_dir: Directory with word piece predictions
  - Test_data_dir: Directory with test data clinical notes
  - lower_case: Flag for if word piece predictions are lower case
  - Error_detection: Flag for if error detection is done, must be used with NER_result_coll.txt file
###Evaluation
- To evaluate annotated outputs
  ```
  eval_script.py <annotated answers folder> <annoated predictions folder>
  ```
###Error Detection
- To get mis-tagged errors file & predictions as txt file
  ```
  ner_detokenize.py --output_dir ~/n2c2_2022_project/Outputs/ --answer_path ~/n2c2_2022_project/Outputs/test.tsv
  perl colleval.pl < ~/n2c2_2022_project/Outputs/NER_result_conll.txt
  CMED_postprocessing.py --prediction_dir ~/n2c2_2022_project/Outputs/ --Test_data_dir ~/n2c2_2022_project/CMED/dev/ --Error_detection True
  ```
- ner_detokenize.py all parameters:
  - output_dir: Directory with word piece outputs from test data
  - answer_path: Directory with processed test data file

*Shell scrips for linux based systems included for running scripts

##Abstract
The area of machine learning, Natural Language Processing(NLP), has been applied by researchers for mining clinical notes
to extract & identify important information. Machine learning models can learn representations of the language from text & apply
the model to task such as information extraction or question answering. 

The 2022 National Clinical NLP Challenges(n2c2) has task researchers with developing solutions for extracting information
regarding medication & medication related events. For this task the Contextualized Medication Event Dataset(CMED) has been given.
CMED is a dataset of clinical notes & annotated versions of the notes, containing task relevent information about the notes.

In this work models with the Bert architecture, BioBert & two variations of Bio+Clinical Bert are applied to learn representations
of CMED & used to extract mentions of medications from clinical notes. The Bert architecture is used for language modeling 
with the applied versions pre-trained on general domain text, biomedical text & clinical text. System performance is evaluated 
on an evaluation portion of CMED with metrics recall, precision, & F1-Score. 

##Modeling:BERT(Bi-Directional Encoder Representations From Transformers) 
For modeling the language in the clinical notes the BERT architecture is fine-tuned with CMED & applied for Named Entity Recognition. 
The versions of Bert applied are BioBert and two version of Clinical Bert, and are pretrained on the following corpus combinations:
- Biobert Pre-Training Corpus: English Wiki + Books Corpus + PubMed Abstracts
- Bio+Clinical Bert Discharge: English Wiki + Books Corpus + PubMed Abstracts + PMC Articles + MIMIC III Discharge Summaries
- Bio+Clinical Bert All Notes: English Wiki + Books Corpus + PubMed Abstracts + PMC Articles + All MIMIC III notes 
####BioBert & Clinical Bert Models Checkpoint file links
- Biobert Version 1.1: https://github.com/naver/biobert-pretrained/releases
- Bio+Clinical Bert All MIMIC III Notes: https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT/tree/main
- Bio+Clinical Bert MIMIC III Discharge Summaries: https://huggingface.co/emilyalsentzer/Bio_Discharge_Summary_BERT/tree/main

*Biobert Version 1.1 files included in repository
##Testing Results
For the training and testing done in this work pipelines are compared with uncased and lower-cased pre-processed data, without word or number removal.
The max word length used is 128 & the max work piece length used is 425. Each model is trained for 10 epochs with a leaning rate of 5e-5. 
For testing the metrics used are precision, recall, & F-score. Each of these scores are evaluated using lenient and strict scores meaning that 
if a medication is partially detected then it counts in lenient scores where with strict the entire medication will need to be detected.

![img.png](img.png)![img_1.png](img_1.png)![img_2.png](img_2.png)![img_3.png](img_3.png)

The testing results show the best performance from the version of Bio+Clinical Bert pre-trained on all MIMIC III clinical notes 
with lower case input tokens, with the highest lenient & strict F1-scores of 0.9549 & 0.9222. Bio+Clinical Bert pre-trained on
all MIMIC III clinical notes also achieves the lowest lenient & strict F1-scores when uncased with scores of 0.9464 & 0.9086.
Bio+Clinical Bert pre-trained on MIMIC III Discharge summaries when uncased achieves the highest lenient precision of 0.9617
& BioBert when lower-cased achieves the highest strict precision of 0.9329. 
