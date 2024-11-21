# Electronic Health Record Modeling With Large Language Models For Medication Event Extraction

## How to run
    
    CMED_ner.py
        --encoder_link bert-base-uncased
        --task 1
        --generate_data True
        --data_dir ./CMED/
        --num_train_epochs 10
        --batch_sz 32
        --lowercase True

    *If punkt & stopwords not downloaded from nltk, uncomment lines 2 & 3 in CMED_preprocessing
   
## Abstract
Attention-based models have become the leading
approach in modeling medical language for Natural Language
Processing (NLP) in clinical notes. These models outperform
traditional techniques by effectively capturing contextual representations of language.

In this research a comparative analysis is done amongst pretrained attention based models namely Bert Base, BioBert, two
variations of Bio+Clinical Bert, RoBerta, and Clinical Longformer on task related to Electronic Health Record (EHR)
information extraction. The tasks from Track 1 of Harvard
Medical School’s 2022 National Clinical NLP Challenges (n2c2)
are considered for this comparison, with the Contextualized
Medication Event Dataset (CMED) given for these task. CMED
is a dataset of unstructured EHR’s and annotated notes that
contain task relevant information about the EHR’s. The goal
of the challenge is to develop effective solutions for extracting
contextual information related to patient medication events from
EHR’s using data driven methods.

Each pre-trained model is fine-tuned and applied on CMED
to perform medication extraction, medical event detection, and
multi-dimensional medication event context classification. Processing methods are also detailed for breaking down EHR’s
for compatibility with the applied models. Performance analysis
has been carried out using a script based on constructing
medical terms from the evaluation portion of CMED with metrics
including recall, precision, and F1-Score. The results demonstrate
that models pre-trained on clinical data are more effective in
detecting medication and medication events, but Bert Base, pretrained on general domain data showed to be the most effective
for classifying the context of events related to medications. 

#### Bert variants Huggingface links
    - "bert-base-cased"
    - "dmis-lab/biobert-bert-cased-v1.2"
    - "emilyalsentzer/Bio_ClinicalBERT"
    - "emilyalsentzer/Bio_Discharge_Summary_BERT"


#### Contextualized Medication Event Dataset File Structure
    CMED/ 
        train,
        eval,
        train_plus_eval,
        test

## Task 1 Test Results
<p align="center">
  <img src="figures/Task 1 Results.png" />
</p>

## Task 2 Test Results
<p align="center">
  <img src="figures/Task 2 Results.png" />
</p>

*NOTE: Contextulaized Medication Event Dataset not publicly available and must be requested from https://n2c2.dbmi.hms.harvard.edu/

-TODO: add in inference code
