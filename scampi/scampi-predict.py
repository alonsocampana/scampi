import argparse
import os
import pickle
import pandas as pd
from utils import MetaboliteProcessor, MetaboliteDownloader, PredictionDataset
from modules import IntegrativeModel
import torch
import numpy as np
import gc
import warnings


def pred(model, device, dataloader):
    """
    Predicts the instance in dataloader using model, and returns the prediction concatenated as a numpy array.
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for x, batch in enumerate(dataloader):
            data_prot = batch[0].float().to(device)
            data_met = batch[1].float().to(device)
            logits = model(data_met, data_prot).squeeze()
            y_pred = torch.sigmoid(logits) # maps to (0, 1) range
            preds.append(y_pred.detach().cpu().numpy())
        del data_prot, data_met, logits
        gc.collect()
    return np.concatenate(preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction script for scampi")
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="Enables GPU training")

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enables warnings")

    parser.add_argument(
        "--data",
         type = str,
         required=True,
         help="CSV file to be predicted, containing the protein sequences on\
          the first column and the pubchem cids on the second")

    parser.add_argument(
        "--batch_size",
         type = int,
         default=2,
         required=False,
         help="Batch size for predictions. Increases speed and memory usage.")

    args, _ = parser.parse_known_args()
    if args.verbose:
        pass
    else:
        warnings.filterwarnings("ignore") #Ignores warnings
    os.environ["SCAMPI_CUDA_IS_ENABLED"] = str(args.cuda)
    os.environ["SCAMPI_PATH"] = os.path.join(os.getcwd(), os.pardir) #sets global variables with path of the experiment
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda') # uses gpu training
    else:
        device = torch.device('cpu')
    prediction_df = pd.read_csv(args.data) # Reads the cids present in a provided CSV file
    unique_cids = prediction_df.iloc[:, 1].unique()
    md = MetaboliteDownloader() #downloads the metabolite data
    mp = MetaboliteProcessor()  # applies preprocessing to the metabolite data
    with open("fitted_metproc.pkl", "rb") as f:
        mp = pickle.load(f) # loads the preprocessor for the metabolite features
    met_data = md.get(unique_cids[0:100])
    md.exit()
    mets = mp.transform(met_data)
    model = IntegrativeModel(hidden_dim=2048,
                 init_dim_met=622,
                 dim_prot=128,
                 n_attn_blocks=32,
                 n_res_blocks=5,
                 p_dropout= 0.33) # creates the model with the selected hyperparameters
    model.load_state_dict(torch.load("../models/trained_scampi.pth",
                                     map_location=device)["model"]) # loads the pretraining weights
    model.to(device)
    ppd = PredictionDataset(prediction_df, mets, length=1024) # Creates a pytorch dataset from the provided data
    pred_dataloader = torch.utils.data.dataloader.DataLoader(ppd, # creates a dataloader
                                                             num_workers=16,
                                                             drop_last=False,
                                                             shuffle=False,
                                                             batch_size=args.batch_size) 
    predictions = pred(model, device, pred_dataloader) 
    new_file = os.path.splitext(args.data)[0] + "_preds.csv" 
    prediction_df.assign(predicted_interaction=predictions).to_csv(new_file, # saves the predictions to the csv file
                                                                   index=False) 
