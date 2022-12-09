#!/usr/bin/env python3.9

import pandas as pd
import numpy as np
import torch
from torch import nn
import os
import gc
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
import json
from modules import IntegrativeModel, TransformerProt
from utils import ProtPairsDataset

# load the global environment variables set by main script
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
    prots = df["protein"].to_numpy()
    met_sample = np.random.choice(mets, size=size, replace=False)
    prot_sample = np.random.choice(prots, size=size, replace=False)
    # random sample, with scores init to 0
    random_sample = pd.DataFrame(
        {"chemical": met_sample, "protein": prot_sample, "interaction_score": np.zeros([size])})
    # outer join keeping track of the origin.
    random_merged = reference_df.merge(
        random_sample, how='outer', on=["chemical", "protein"], indicator=True)
    # The random pairs found in both dataframes are discarded.
    random_merged = random_merged[random_merged["_merge"] == "right_only"].loc[:, ["chemical", "protein", "interaction_score_y"]].reset_index(drop=True)
    random_merged.columns = ["chemical", "protein", "interaction_score"]
    # return both dataframes concatenated
    output = pd.concat([df, random_merged], axis=0).reset_index(drop=True)
    return output


def generate_interactions(X_train, merged_data, uniprot):
    """
    Generates the interaction dataset by adding negative samples.
    """
    out = (add_negative_sample(X_train, merged_data)
           .merge(uniprot, how="inner", left_on="protein", right_on="Cross-reference (STRING)")
           .drop("Cross-reference (STRING)", axis=1).reset_index(drop=True).sample(frac=1, replace=False)
           .reset_index(drop=True))
    valid_seq = out["Sequence"].apply(len) > 12
    return out[valid_seq]



def train_model(RUN):
    # Loads the different partitions used for testing and training
    X_test3 = pd.read_csv("../data/stitch/data_db2.csv", index_col=0)
    X_test2 = pd.read_csv("../data/stitch/data_cb2.csv", index_col=0)
    X_test1 = pd.read_csv("../data/stitch/data_pb2.csv", index_col=0)
    X_train = pd.read_csv("../data/stitch/data_tr2.csv", index_col=0)
    if PERMUTE: # randomization
        X_train["protein"] = np.random.permutation(X_train["protein"])
    mets = pd.read_csv("../data/stitch/processed_mets.csv", index_col=0)
    uniprot = pd.read_csv("../data/stitch/prot.csv")
    uniprot = uniprot[uniprot["Sequence"].apply(len) <= 1022] # discards the proteins that are longer than 1022 amino acids for avoiding truncation
    merged_data = pd.concat([X_train, X_test1, X_test2, X_test3], axis=0, ignore_index=True)
    protein_counts = merged_data["protein"].value_counts()
    protein_array = protein_counts.to_numpy()
    min_scale = 0.2
    emms = MinMaxScaler([min_scale, 1.5])
    scaled = emms.fit_transform(np.log10(1/(protein_array/protein_array.sum()))[:,None]).astype(np.float32).squeeze()
    protein_weights = pd.DataFrame([protein_counts.index, scaled]).transpose()
    protein_weights.columns = ["Cross-reference (STRING)", "weight"]
    uniprot = uniprot.merge(protein_weights, on="Cross-reference (STRING)")

    def train(model, device, dataloader, loss_fn, optimizer, batch_acc):
        optimizer.zero_grad()
        model.train()
        losses = []
        rocs = []
        for x, batch in enumerate(dataloader):
            data_prot = batch[0].float().to(device)
            data_met = batch[1].float().to(device)
            weights = batch[2].float().squeeze().to(device) # weights associated to reweighting (based on stitch score)
            target = batch[3].float().squeeze().to(device)
            loss = loss_fn(weight=weights, reduction="mean").to(device)
            logits = model(data_met, data_prot).squeeze()
            y_pred = torch.sigmoid(logits)
            output = loss(logits, target)
            output.backward()
            if (x+1) % batch_acc == 0: # gradient accumulation, effectively increading batch size
                nn.utils.clip_grad_norm_(model.parameters(), 2)
                optimizer.step()
                optimizer.zero_grad()
            try:
                roc = roc_auc_score(target.cpu().numpy(),
                                    y_pred.detach().cpu().numpy())
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

    def test(model, device, dataloader, loss_fn):
        model.eval()
        losses = []
        rocs = []
        accs_0 = []
        accs_1 = []
        with torch.no_grad():
            for x, batch in enumerate(dataloader):
                data_prot = batch[0].float().to(device)
                data_met = batch[1].float().to(device)
                weights = batch[2].float().squeeze().to(device)
                target = batch[3].float().squeeze().to(device)
                loss = loss_fn(weight=weights, reduction="mean")
                logits = model(data_met, data_prot).squeeze()
                y_pred = torch.sigmoid(logits)
                output = loss(logits, target)
                try:
                    roc = roc_auc_score(target.cpu().numpy(),
                                        y_pred.detach().cpu().numpy())
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
                 init_dim_met=622,
                 dim_prot=128,
                 n_attn_blocks=32,
                 n_res_blocks=5,
                 p_dropout= 0.33)
    # Check if the GPU is available
    path = "../models/" + RUN + "_last.pth"
    if USE_CUDA:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")
    if os.path.exists(path):
        checkpoint = torch.load("../models/" + RUN + "_last.pth",
                                map_location=device)
        start = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"])
        model = nn.DataParallel(model)
        model.to(device)
        optim = torch.optim.Adam(model.parameters(),
                                 lr=0.000031,
                                 weight_decay=0.0000072)
        optim.load_state_dict(checkpoint["optimizer"])
        with open("../models/"+RUN+'.json', 'r') as f:
            train_log = json.load(f)
    else:
        prot_transformer = TransformerProt()
        prot_transformer.load_state_dict(torch.load(
            "2022-02-18 09:21:36.083931.pth",
            map_location = device))
        model.prot_transformer = prot_transformer.transformer
        model.set_cold()
        model = nn.DataParallel(model)
        model.to(device)
        optim = torch.optim.Adam(model.parameters(), lr=0.000031,
                                 weight_decay= 0.0000072)
        start = 0
        train_log = {}
        with open("../models/"+RUN+'.json', 'w') as f:
            json.dump(train_log, f)
    batch_acc = 4*((start//5)+1)
    print(f'Selected device: {device}')
    num_epochs = 60
    for epoch in range(start, num_epochs):
        train_log[epoch] = {}
        if epoch == 1:
            model.module.set_warm()
        elif epoch in [10, 15, 20, 25, 30, 35, 40]:
            batch_acc *= 2
            for par in optim.param_groups:
                par["weight_decay"] = par["weight_decay"]/2
        train_dataloader = torch.utils.data.dataloader.DataLoader(
            ProtPairsDataset(generate_interactions(
                            X_train.sample(frac=0.3).iloc[0:50],
                            merged_data, uniprot), mets, length=1024),
            num_workers=16, drop_last=True, shuffle=False, batch_size = 128)
        train_loss = train(
            model, device, train_dataloader, loss_fn, optim, batch_acc)
        metrics = {}
        metrics["train_loss"] = str(train_loss[0])
        metrics["train_roc"] = str(train_loss[1])
        train_log[epoch] = metrics
        with open("../models/" + RUN+'.json', 'w') as f: # save log
            json.dump(train_log, f)
        # Save model
        torch.save({"epoch": epoch,
                    "optimizer": optim.state_dict(),
                    "model": model.module.state_dict()},
                    "{}.pth".format("../models/" + RUN + "_last"))
        gc.collect()


if __name__ == "__main__":
    RUN = "train_alldata_30"
    train_model(RUN)
