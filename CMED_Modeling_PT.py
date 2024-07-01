import pandas as pd
import transformers
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn import preprocessing
from sklearn import metrics
import numpy as np
import pickle

import CMED_postprocessing
from eval_script import Corpora
from eval_script import evaluate

class CMED_Model(torch.nn.Module):
    def __init__(self, bert_model,Task = 1):
        super(CMED_Model, self).__init__()
        self.encoder = bert_model;
        self.feedforward_L1 = torch.nn.Linear(768, 256);
        if Task == 1: self.output_layer_last_ff = torch.nn.Linear(256, 3);
        elif Task == 2: self.output_layer_last_ff = torch.nn.Linear(256, 7);
    def forward(self, input_ids):
        bert_output = self.encoder(input_ids = input_ids);
        logits = self.feedforward_L1(bert_output["last_hidden_state"]);
        logits = self.output_layer_last_ff(logits);
        return logits;

class CMED_Dataset(Dataset):
    def __init__(self, tokens, labels):
        self.text = tokens;
        self.labels = labels;

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):

        return self.text[idx], self.labels[idx];

def CMED_PT_Dataset(x_data = np.array([]), y_data = np.array([]), batch_sz = 32, shuffle = False):

    CMED_Dataset_ = CMED_Dataset(tokens=x_data, labels=y_data)
    CMED_Dataloader = DataLoader(CMED_Dataset_,batch_size=batch_sz, shuffle= shuffle);

    return CMED_Dataloader;

def Train_Model(train_dataloader, bert_tokenizer, model = torch.nn.Module(),Num_Epochs = 10 ,Device = None, debug=False, embedding_length = 128, Model_Path =  "CMED_Model.pt", tags=["B-Drug", "I-Drug", "O"]):
    History_Path = Model_Path.replace(".pt", "Train_History.pt");
    loss_function = torch.nn.CrossEntropyLoss();
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.00005);
    #optimizer = torch.optim.SGD(model.parameters(), lr = 0.00005);
    model.train();
    Special_Token_dict = {"[CLS]":101 , "[SEP]":102 , "[PAD]":0};
    Num_batches = len(train_dataloader);

    training = True;
    epoch = 0;
    batch_sz = 32;
    history = {};

    while training:
        epoch += 1;
        history[epoch] = {"Batch Loss": [],  "Avg Batch Loss": []};
        print("Epoch", str(epoch), "-------------------------------------------------------");
        Avg_loss_temp_list = [];
        epoch_step = 0;

        for index, (x,y) in enumerate(train_dataloader):
            epoch_step += 1;

            y = " ".join(y);
            y = np.array(y.split(" "));
            NOTCLS_INDEX  = np.where(y != "[CLS]");
            NOTSEP_INDEX = np.where(y != "[SEP]");
            NOTX_INDEX = np.where(y != "X");

            #Only keep ground truth that is being predicted with
            y_prediction_index = np.intersect1d(NOTSEP_INDEX, NOTCLS_INDEX);
            y_prediction_index = np.intersect1d(y_prediction_index, NOTX_INDEX);
            y = y[y_prediction_index];
            #encode labels in Y to numerical values
            label_encoder = preprocessing.LabelEncoder();
            label_encoder.fit(tags);
            encoded_labels = label_encoder.transform(y);
            encoded_labels = torch.tensor(encoded_labels.reshape((-1, 1)), device=Device);
            encoded_labels = encoded_labels.reshape((-1,));

            #Get index for x that are going to be used for training
            x_copy = x.flatten();
            NOTPAD_INDEX = np.where(x_copy != Special_Token_dict["[PAD]"]);
            x_copy = x_copy[NOTPAD_INDEX];
            NOTCLS_INDEX = np.where(x_copy != Special_Token_dict["[CLS]"]);
            NOTSEP_INDEX = np.where(x_copy != Special_Token_dict["[SEP]"]);
            x_prediction_index = np.intersect1d(NOTX_INDEX, NOTCLS_INDEX);
            x_prediction_index = np.intersect1d(x_prediction_index, NOTSEP_INDEX);

            optimizer.zero_grad();

            model_input =x.long();
            model_input = model_input.to(Device);
            model_output = model(model_input);
            #reshape to (batch_sz*num_wordpieces, embedding length)
            model_output = model_output.reshape((model_output.shape[0]*model_output.shape[1], model_output.shape[2]));
            model_output = model_output[NOTPAD_INDEX];
            model_output = model_output[x_prediction_index];

            loss = loss_function(model_output, encoded_labels);
            loss.backward();
            optimizer.step();

            history[epoch]["Batch Loss"].append(loss.item());
            Avg_Batch_Loss = np.average(history[epoch]["Batch Loss"]);
            history[epoch]["Avg Batch Loss"].append(Avg_Batch_Loss);

            print("Epoch Percentage: ", str(np.round((epoch_step/Num_batches)*100, decimals = 2)));
            print("Avg Loss: ", str(np.round(Avg_Batch_Loss, decimals=8)));

        if debug:
            Eval_Model(train_dataloader, bert_tokenizer, model = model, Device=Device, embedding_length=embedding_length)
            train_input = input("Continue Training(Y/N)");
            if train_input.lower() == "y": training = True;
            else: training = False
        elif epoch == Num_Epochs:
            break;

    torch.save(model,Model_Path);
    torch.save(history, History_Path);
    print("Training Finished!!!");

    return model, history;

def Eval_Model(eval_dataloader, bert_tokenizer, model = torch.nn.Module(), Device = None, embedding_length = 128, tags=["B-Drug", "I-Drug", "O"]):

    correct_list = [];
    pred_list = [];

    Num_batches = len(eval_dataloader);
    history = {"Epoch": [],  "Avg loss": [], "mae": []};
    Special_Token_dict = {"[CLS]": 101, "[SEP]": 102, "[PAD]": 0};
    model.eval()
    
    for index, (x,y) in enumerate(eval_dataloader):
        y = " ".join(y);
        y = np.array(y.split(" "));
        NOTCLS_INDEX = np.where(y != "[CLS]");
        NOTSEP_INDEX = np.where(y != "[SEP]");
        NOTX_INDEX = np.where(y != "X");
        y_prediction_index = np.intersect1d(NOTSEP_INDEX, NOTCLS_INDEX);
        y_prediction_index = np.intersect1d(y_prediction_index, NOTX_INDEX);
        y = y[y_prediction_index];
        label_encoder = preprocessing.LabelEncoder();
        label_encoder.fit(tags);
        encoded_labels = label_encoder.transform(y);
        encoded_labels = torch.tensor(encoded_labels.reshape((-1, 1)), device=Device, dtype=torch.int);
        encoded_labels = encoded_labels.reshape((-1,));


        x_copy = x.flatten();
        NOTPAD_INDEX = np.where(x_copy != Special_Token_dict["[PAD]"]);
        x_copy = x_copy[NOTPAD_INDEX];
        NOTCLS_INDEX = np.where(x_copy != Special_Token_dict["[CLS]"]);
        NOTSEP_INDEX = np.where(x_copy != Special_Token_dict["[SEP]"]);
        x_prediction_index = np.intersect1d(NOTX_INDEX, NOTCLS_INDEX);
        x_prediction_index = np.intersect1d(x_prediction_index, NOTSEP_INDEX);

        model_input = x.long();
        model_input = model_input.to(Device);
        model_output = model(model_input);
        model_output = model_output.reshape((model_output.shape[0] * model_output.shape[1], -1));
        model_output = model_output[NOTPAD_INDEX];
        model_output = model_output[x_prediction_index];
        softmax = torch.nn.LogSoftmax();
        softmax = torch.nn.Softmax();
        model_output = softmax(model_output);
        model_output = torch.argmax(model_output, dim = 1);

        correct_list.extend(encoded_labels.tolist());
        pred_list.extend(model_output.tolist());

    print(metrics.classification_report(correct_list, pred_list));

    return correct_list, pred_list;

def Test_Model(test_xdata,test_xdata_wordpieces ,model = torch.nn.Module() ,bert_tokenizer = transformers.AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2"),embedding_length = 128 ,tags = ["B-Drug", "I-Drug", "O"], save_filepath = "", Device = None):
    model.eval();
    label_decoder = preprocessing.LabelEncoder();
    label_decoder.fit(tags);

    test_xdata = torch.tensor(test_xdata);
    test_dataloader = CMED_PT_Dataset(x_data = test_xdata, y_data = test_xdata_wordpieces ,batch_sz=32)

    Special_Token_dict = {"[CLS]": 101, "[SEP]": 102, "[PAD]": 0};
    predictions_list = [];
    words_list = [];
    for index, (input_ids, input_wordpieces) in enumerate(test_dataloader):
        input_wordpieces = " ".join(input_wordpieces);
        input_wordpieces = np.array(input_wordpieces.split(" "));
        input_copy = input_ids.flatten();
        PAD_index = np.where(input_copy == Special_Token_dict["[PAD]"]);
        NOTPAD_index = np.where(input_copy != Special_Token_dict["[PAD]"]);
        SEP_index = np.where(input_copy == Special_Token_dict["[SEP]"]);
        CLS_index = np.where(input_copy == Special_Token_dict["[CLS]"]);

        model_input = input_ids.long();
        model_input = model_input.to(Device);
        logits = model(model_input);
        logits = logits.reshape((logits.shape[0] * logits.shape[1], -1));

        softmax = torch.nn.LogSoftmax(dim=0);
        probs = softmax(logits);

        predictions = torch.argmax(probs, dim=1);
        predictions = predictions.cpu();
        predictions = label_decoder.inverse_transform(predictions);
        predictions[SEP_index] = "[SEP]";
        predictions[CLS_index] = "[CLS]";
        predictions =predictions[NOTPAD_index];

        predictions_list.extend(list(predictions));
        words_list.extend(list(input_wordpieces));

    return words_list, predictions_list;

def Model_predict(input = [],model = torch.nn.Module() ,bert_tokenizer = transformers.AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2"), tags = ["B-Drug", "I-Drug", "O"], save_filepath = "", Device = None):
    model.eval();
    label_decoder = preprocessing.LabelEncoder();
    label_decoder.fit(tags);

    y_null = np.zeros(len(input));
    input_dataloder = CMED_PT_Dataset(x_data = input, y_data = y_null ,batch_sz=32)

    predictions_list = [];
    words_list = [];

    for index, (x, y) in enumerate(input_dataloder):
        model_input = bert_tokenizer.batch_encode_plus(list(x), max_length=embedding_length, truncation=True,
                                                       pad_to_max_length=True, return_tensors='pt');
        model_input = model_input.to(Device);
        logits = model(model_input["input_ids"]);
        softmax = torch.nn.LogSoftmax();
        probs = softmax(logits);

        predictions = torch.argmax(probs, dim=2);
        predictions = predictions.cpu();
        predictions = np.reshape(predictions, (predictions.shape[0] * predictions.shape[1],))
        predictions = label_decoder.inverse_transform(predictions);

        words = " ".join(input_batch);
        words = np.array(words.split(" "));

        PAD_index = np.where(words == "[PAD]");
        SEP_index = np.where(words == "[SEP]");

        predictions[SEP_index] = "[SEP]";
        predictions = np.delete(predictions, PAD_index);
        words = np.delete(words, PAD_index);

        predictions_list.extend(list(predictions));
        words_list.extend(list(words));

    return words_list, predictions_list;
