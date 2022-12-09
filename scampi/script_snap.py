#!/usr/bin/env python3.9

import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import os
import gc
from modules import IntegrativeModel, TransformerProt
from utils import ProtPairsDatasetSnap as ProtPairsDataset
from sklearn.metrics import roc_auc_score
import json

#load the global environment variables set by main script
os.environ["NCCL_P2P_DISABLE"] = str(1)
PERMUTE = eval(os.environ["SCAMPI_NO_TRANS"])
USE_CUDA = eval(os.environ["SCAMPI_CUDA_IS_ENABLED"])
TRAINING_FRACTION = float(os.environ["SCAMPI_FRACTION"])
PATH = os.environ["SCAMPI_PATH"]

def add_negative_sample(df, reference_df, ratio=1):
        """
        Adds a negative sample to df.
        reference df is a dataframe used as reference for the positive samples.
        If a pair randomly sampled is contained, it's dropped.
        ratio is the initial number of samples, as a fraction of the length of
        the positive sample (length of df).

        """
        size = int(df.shape[0] * ratio)
        mets = df["chemical"].to_numpy()
        prots = df["Entry"].to_numpy()
        met_sample = np.random.choice(mets, size=size, replace=False)
        prot_sample = np.random.choice(prots, size=size, replace=False)
        # random sample, with scores init to 0
        random_sample = pd.DataFrame(
            {"chemical": met_sample, "Entry": prot_sample, "score": np.zeros([size])})
        # outer join keeping track of the origin.
        random_merged = reference_df.merge(
            random_sample, how='outer', on=["chemical", "Entry"], indicator=True)
        # The random pairs found in both dataframes are discarded.
        random_merged = random_merged[random_merged["_merge"] == "right_only"].loc[:, ["chemical", "Entry", "score_y"]].reset_index(drop=True)
        random_merged.columns = ["chemical", "Entry", "score"]
        # return both dataframes concatenated
        output = pd.concat([df, random_merged], axis=0).reset_index(drop=True)
        return output

def generate_interactions(X_train, merged_data, uniprot):
    """
    Generates interactions by adding a negative sample to the positives found in the dataset
    """
    out = (add_negative_sample(X_train, merged_data)
           .merge(uniprot, how="inner", left_on="Entry", right_on="Entry")
           .drop("Entry", axis=1).reset_index(drop=True).sample(frac=1, replace=False)
           .reset_index(drop=True))
    valid_seq = out["Sequence"].apply(len) > 12 # filters sequences that are too short
    return out[valid_seq].fillna(1)

def train_model(RUN):
    merged_data = pd.read_csv(os.path.join(PATH, "data/biosnap/interactions_BIOSNAP.csv"), index_col=0).drop("Sequence", axis=1)
    if PERMUTE: # creates a random permutation of one set
        merged_data["Entry"] = np.random.permutation(merged_data["Entry"])
    mets = pd.read_csv(os.path.join(PATH, "data/biosnap/processed_biosnap.csv"), index_col=0) # reads the metabolite set
    uniprot = pd.read_csv(os.path.join(PATH, "data/biosnap/SNAP_seqs.csv"), index_col=0) # reads the sequences for the proteins
    X_train, X_test = train_test_split(merged_data, train_size = 0.9, random_state=3558) # creates the train/test split
    merged_data = merged_data.assign(score=1)
    def train(model, device, dataloader, loss_fn, optimizer, batch_acc):
        optimizer.zero_grad()
        model.train()
        losses = []
        rocs = []
        for x, batch in enumerate(dataloader):
            data_prot = batch[0].float().to(device)
            data_met = batch[1].float().to(device)
            target = batch[2].float().squeeze().to(device)
            loss = loss_fn()
            logits = model(data_met, data_prot).squeeze()
            y_pred = torch.sigmoid(logits) # creates prediction
            output = loss(logits, target)
            output.backward()
            if (x+1)%batch_acc == 0: # uses gradient accumulation for making (effectively) bigger batches
                nn.utils.clip_grad_norm_(model.parameters(), 2)
                optimizer.step()
                optimizer.zero_grad()
            try:
                roc = roc_auc_score(target.cpu().numpy(), y_pred.detach().cpu().numpy()) # adds computation to the roc
                rocs.append(roc)
            except ValueError:
                pass
            loss_instance = output.data.cpu().numpy()
            losses.append(loss_instance)
        try:
            del data_prot, data_met, target, logits
        except UnboundLocalError:
            pass

        gc.collect()
        return np.mean(losses), np.mean(np.array(rocs))

    def test(model, device, dataloader, loss_fn): # tests the performance of the model without training
        model.eval()
        losses = []
        rocs = []
        accs_0 = []
        accs_1 = []
        with torch.no_grad():
            for x, batch in enumerate(dataloader):
                data_prot = batch[0].float().to(device)
                data_met = batch[1].float().to(device)
                target = batch[2].float().squeeze().to(device)
                loss = loss_fn()
                logits = model(data_met, data_prot).squeeze()
                y_pred = torch.sigmoid(logits)
                output = loss(logits, target)
                try:
                    roc = roc_auc_score(target.cpu().numpy(), y_pred.detach().cpu().numpy())
                    rocs.append(roc)
                except ValueError:
                    pass
                accs_0.append(((y_pred[target==0] < 0.5).sum()/(target==0).sum()).cpu().numpy())
                accs_1.append(((y_pred[target==1] > 0.5).sum()/(target==1).sum()).cpu().numpy())
                loss_instance = output.data.cpu().numpy()
                losses.append(loss_instance)
            try:
                del data_prot, data_met, target, logits
            except UnboundLocalError:
                pass
            gc.collect()
        return np.mean(losses), np.mean(np.array(rocs)),  np.mean(accs_0), np.mean(accs_1)
    loss_fn = nn.BCEWithLogitsLoss
    model = IntegrativeModel(hidden_dim=2048,
                 init_dim_met=630,
                 dim_prot=128,
                 n_attn_blocks=32,
                 n_res_blocks=2,
                 p_dropout= 0.4)
    # Check if the GPU is available
    path = "../models/" + RUN + "_last.pth"
    if USE_CUDA:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")
    best_loss = 5
    if os.path.exists(path): # if the model exists, resumes training
        checkpoint = torch.load("../models/"+RUN + "_last.pth")
        start = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"])
        model= nn.DataParallel(model)
        optim = torch.optim.Adam(model.parameters(), lr=0.000031, weight_decay= 0.0000072)
        optim.load_state_dict(checkpoint["optimizer"])
        with open("../models/"+RUN+'.json', 'r') as f:
            train_log = json.load(f)
        for epoch in range(0, start):
            best_loss = min(float(train_log[str(epoch)]["test_roc"]), best_loss)
        print(f"Resuming training from epoch {start}")
    else: # if not starts training from scratch
        prot_transformer = TransformerProt()
        prot_transformer.load_state_dict(torch.load("2022-02-18 09:21:36.083931.pth", map_location=device))
        model.prot_transformer = prot_transformer.transformer
        model.set_cold()
        model= nn.DataParallel(model)
        optim = torch.optim.Adam(model.parameters(), lr=0.000031, weight_decay= 0.0000072)
        start = 0
        train_log = {}
        with open("../models/"+RUN+'.json', 'w') as f:
            json.dump(train_log, f)
    batch_acc = 4*((start//5)+1)
    model.to(device)
    print(f'Selected device: {device}')
    test_dataset = ProtPairsDataset(generate_interactions(X_test, merged_data, uniprot), mets, length=1024)
    test_dataloader = torch.utils.data.dataloader.DataLoader(test_dataset, num_workers=16, drop_last=True, shuffle=False, batch_size = 256)
    num_epochs = 40
    for epoch in range(start, num_epochs):
        train_log[epoch] = {}
        if epoch == 1:
            model.module.set_warm()
        elif epoch in [10, 15, 20, 25, 30, 35]: # doubles the effective training batch size, decreasing the learning rate
            batch_acc *= 2
            for par in optim.param_groups:
                par["weight_decay"] = par["weight_decay"]/2 # reduces the weight decay for effective decrease of the learning rate
        train_dataloader = torch.utils.data.dataloader.DataLoader(ProtPairsDataset(generate_interactions(X_train.sample(frac=TRAINING_FRACTION), merged_data, uniprot), mets, length=1024),
                                                                  num_workers=16, drop_last=True, shuffle=False, batch_size = 128)
        print("starting epoch...")
        train_loss = train(
            model, device, train_dataloader, loss_fn, optim, batch_acc)
        test_loss = test(model, device, test_dataloader, loss_fn)
        print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch
                                                                                + 1, num_epochs, train_loss, test_loss))
        metrics = {}
        metrics["train_loss"] = str(train_loss[0])
        metrics["train_roc"] = str(train_loss[1])
        metrics["test_loss"] = str(test_loss[0])
        metrics["test_roc"] = str(test_loss[1])
        train_log[epoch] = metrics
        with open("../models/"+RUN+'.json', 'w') as f:
            json.dump(train_log, f) # adds epoch to the logger
        if test_loss[1] > best_loss:
            best_loss = test_loss[0]
            torch.save({"epoch":epoch,
                    "optimizer": optim.state_dict(),
                    "model":model.module.state_dict()},
                    "{}.pth".format("../models/"+RUN+"_best"))
        best_loss = test_loss[0]
        torch.save({"epoch":epoch,
                    "optimizer": optim.state_dict(),
                    "model":model.module.state_dict()},
                    "{}.pth".format("../models/"+RUN+"_last"))
        gc.collect()
