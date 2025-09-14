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

#### Model links for importing directly from HuggingFace
    - "bert-base-cased"
    - "dmis-lab/biobert-bert-cased-v1.2"
    - "emilyalsentzer/Bio_ClinicalBERT"
    - "emilyalsentzer/Bio_Discharge_Summary_BERT"


*NOTE: Contextulaized Medication Event Dataset not publicly available and must be requested from https://n2c2.dbmi.hms.harvard.edu/

#### Citation
    @mastersthesis{quddoos2023performance,
      author       = {Tariq Abdul Quddoos},
      title        = {Performance Analysis Of Attention Based Deep Learning Models On Named Entity Recognition In Electronic Health Records},
      school       = {Prairie View A\&M University},
      year         = {2023},
      address      = {Prairie View, TX},
      url          = {https://digitalcommons.pvamu.edu/pvamu-theses/1533/}
    }
