import pandas as pd
import numpy as np
import pickle
import torch
from torch import nn
import requests
import os
import base64
from sklearn.preprocessing import MinMaxScaler
import uuid
import warnings
import shutil


class ProtPairsDataset(torch.utils.data.Dataset):
    """
    Dataset containing a set of interactions used for the stitch dataset.

    Args:
    interactions: A DataFrame containing the interactions
    length: The length of the sequences.
    All sequences will be truncated or padded to this length.
    """

    def __init__(self,
                 interactions,
                 mets,
                 length=512,
                 transform=None,
                 target_transform=None):
        self.interactions = interactions
        with open("AAdict.pkl", "rb") as f: # open dictionary containing the embedding
            self.AA_dict = pickle.load(f)
        self.length = length
        # Create entries for infrequent symbols
        self.AA_dict["U"] = self.AA_dict["X"]
        self.AA_dict["B"] = self.AA_dict["X"]
        self.AA_dict["Z"] = self.AA_dict["X"]
        self.AA_int = {key: i+1 for i, key in enumerate((self.AA_dict.keys()))}
        # create positional encoding
        self.positional_encoding = self.get_positional_encoding(
            self.length, 128)
        self.mets = mets.astype({"CID": np.int64}).set_index("CID")

    def __len__(self):
        return self.interactions.shape[0]

    def vec_translate(self, a):
        """
        Utility function used for translating sequences on a vectorized way.
        """
        return np.vectorize(self.AA_int.__getitem__)(a)

    def embed_sequence(self, seq):
        """
        From a sequence as a string creates an array representation
        """
        max_length = self.length
        seq = list(seq)
        posrr = np.array([self.AA_dict[aa] for aa in seq]).T
        length = posrr.shape[1]
        if length > max_length-2:
            posrr = posrr[:, :max_length-2]
            mask = np.concatenate(
                [np.array([24]), np.ones(max_length-2), np.array([25])])
        else:
            mask = np.concatenate([np.array([21]), np.ones(
                length), np.array([24]), np.zeros(max_length-2-length)])
        posrr = np.concatenate(
            [np.zeros([128, 1]), posrr, np.zeros([128, 1])], axis=1)
        prot_arr = np.zeros([128, max_length])
        prot_arr[:posrr.shape[0], :posrr.shape[1]] = posrr
        final_embedding = (prot_arr + self.positional_encoding).T
        return final_embedding, mask, length

    def get_positional_encoding(self, seq_length, n_features):
        """
        Creates the synusoid embedding of dimensions seq_length x n_features
        """
        positional_encoding_1 = np.sin(
            np.tile(np.arange(0, seq_length), [n_features//2, 1])
            / (np.tile(1000**(np.arange(0, (n_features//2))/(n_features//4))[:, None], [1, seq_length])))
        positional_encoding_2 = np.cos(
            np.tile(np.arange(0, seq_length), [n_features//2, 1])
            / (np.tile(1000**(np.arange(0, (n_features//2))/(n_features//4))[:, None], [1, seq_length])))
        positional_encoding = np.concatenate(
            [positional_encoding_1, positional_encoding_2], axis=0)
        return positional_encoding

    def __getitem__(self, idx):
        entry = self.interactions.iloc[idx]
        seq = entry["Sequence"]
        met = entry["chemical"]
        weight = entry["weight"]
        target = entry["interaction_score"]
        if target == 0:
            target = np.zeros(1)
            weight = 0.85
        else:
            weight = target/1000
            target = np.ones(1)
        met_data = self.mets.loc[int(met)]
        embedding, mask, l = self.embed_sequence(seq)
        return torch.Tensor(embedding), torch.Tensor(met_data.to_numpy()), torch.Tensor(np.array([weight])), torch.Tensor(target)


class ProtPairsDatasetSnap(torch.utils.data.Dataset):
    """
    Dataset containing a set of interactions used for the snap dataset.

    Args:
    interactions: A DataFrame containing the interactions
    length: The length of the sequences.
    All sequences will be truncated or padded to this length.
    """

    def __init__(self,
                 interactions,
                 mets,
                 length=512,
                 transform=None,
                 target_transform=None):
        self.interactions = interactions
        with open("AAdict.pkl", "rb") as f:
            self.AA_dict = pickle.load(f)
        self.length = length
        self.AA_dict["U"] = self.AA_dict["X"]
        self.AA_dict["B"] = self.AA_dict["X"]
        self.AA_dict["Z"] = self.AA_dict["X"]
        self.AA_int = {key: i+1 for i, key in enumerate((self.AA_dict.keys()))}
        self.positional_encoding = self.get_positional_encoding(
            self.length, 128)
        self.mets = mets.astype({"CID":np.int64}).set_index("CID")

    def __len__(self):
        return self.interactions.shape[0]

    def vec_translate(self, a):
        return np.vectorize(self.AA_int.__getitem__)(a)

    def embed_sequence(self, seq):
        max_length = self.length
        seq = list(seq)
        posrr = np.array([self.AA_dict[aa] for aa in seq]).T
        length = posrr.shape[1]
        if length > max_length-2:
            posrr = posrr[:, :max_length-2]
            mask = np.concatenate(
                [np.array([24]), np.ones(max_length-2), np.array([25])])
        else:
            mask = np.concatenate([np.array([21]), np.ones(
                length), np.array([24]), np.zeros(max_length-2-length)])
        posrr = np.concatenate(
            [np.zeros([128, 1]), posrr, np.zeros([128, 1])], axis=1)
        prot_arr = np.zeros([128, max_length])
        prot_arr[:posrr.shape[0], :posrr.shape[1]] = posrr
        final_embedding = (prot_arr + self.positional_encoding).T
        return final_embedding, mask, length

    def get_positional_encoding(self, seq_length, n_features):
        positional_encoding_1 = np.sin(
            np.tile(np.arange(0, seq_length), [n_features//2, 1])
            / (np.tile(1000**(np.arange(0, (n_features//2))/(n_features//4))[:, None], [1, seq_length])))
        positional_encoding_2 = np.cos(
            np.tile(np.arange(0, seq_length), [n_features//2, 1])
            / (np.tile(1000**(np.arange(0, (n_features//2))/(n_features//4))[:, None], [1, seq_length])))
        positional_encoding = np.concatenate(
            [positional_encoding_1, positional_encoding_2], axis=0)
        return positional_encoding

    def __getitem__(self, idx):
        entry = self.interactions.iloc[idx]
        seq = entry["Sequence"]
        met = entry["chemical"]
        target = entry["score"]
        met_data = self.mets.loc[int(met)]
        embedding, mask, l = self.embed_sequence(seq)
        return torch.Tensor(embedding), torch.Tensor(met_data.to_numpy()), torch.Tensor([target])


class PredictionDataset(torch.utils.data.Dataset):
    """
    Dataset containing a set of pairs for prediction

    Args:
    interactions: A DataFrame containing the interactions
    length: The length of the sequences.
    All sequences will be truncated or padded to this length.
    """

    def __init__(self,
                 interactions,
                 mets,
                 length=512,
                 transform=None,
                 target_transform=None):
        self.interactions = interactions
        with open("AAdict.pkl", "rb") as f:
            self.AA_dict = pickle.load(f)
        self.length = length
        self.AA_dict["U"] = self.AA_dict["X"]
        self.AA_dict["B"] = self.AA_dict["X"]
        self.AA_dict["Z"] = self.AA_dict["X"]
        self.AA_int = {key: i+1 for i, key in enumerate((self.AA_dict.keys()))}
        self.positional_encoding = self.get_positional_encoding(
            self.length, 128)
        self.mets = mets.astype({"CID":np.int64}).set_index("CID")

    def __len__(self):
        return self.interactions.shape[0]

    def vec_translate(self, a):
        return np.vectorize(self.AA_int.__getitem__)(a)

    def embed_sequence(self, seq):
        max_length = self.length
        seq = list(seq)
        posrr = np.array([self.AA_dict[aa] for aa in seq]).T
        length = posrr.shape[1]
        if length > max_length-2:
            posrr = posrr[:, :max_length-2]
            mask = np.concatenate(
                [np.array([24]), np.ones(max_length-2), np.array([25])])
        else:
            mask = np.concatenate([np.array([21]), np.ones(
                length), np.array([24]), np.zeros(max_length-2-length)])
        posrr = np.concatenate(
            [np.zeros([128, 1]), posrr, np.zeros([128, 1])], axis=1)
        prot_arr = np.zeros([128, max_length])
        prot_arr[:posrr.shape[0], :posrr.shape[1]] = posrr
        final_embedding = (prot_arr + self.positional_encoding).T
        return final_embedding, mask, length

    def get_positional_encoding(self, seq_length, n_features):
        positional_encoding_1 = np.sin(
            np.tile(np.arange(0, seq_length), [n_features//2, 1])
            / (np.tile(1000**(np.arange(0, (n_features//2))/(n_features//4))[:, None], [1, seq_length])))
        positional_encoding_2 = np.cos(
            np.tile(np.arange(0, seq_length), [n_features//2, 1])
            / (np.tile(1000**(np.arange(0, (n_features//2))/(n_features//4))[:, None], [1, seq_length])))
        positional_encoding = np.concatenate(
            [positional_encoding_1, positional_encoding_2], axis=0)
        return positional_encoding

    def __getitem__(self, idx):
        entry = self.interactions.iloc[idx]
        seq = entry.iloc[0]
        met = entry.iloc[1]
        met_data = self.mets.loc[int(met)]
        embedding, mask, l = self.embed_sequence(seq)
        return torch.Tensor(embedding), torch.Tensor(met_data.to_numpy())


def init_weights(m):
    """
    Helper function for initializing Pytorch layers.
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)


def download_file_from_google_drive(id, destination):
    """
    source: https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    """
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    """
    source: https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    """
    Adapted from source:
    https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    """
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for i, chunk in enumerate(response.iter_content(CHUNK_SIZE)):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
        print("")


def select_first(*args):
    return args[0]


def progress_bar(current, total, length=50):
    progress = "█"
    fill = "_"
    frac = int(current/total*length)
    bar = progress * frac + (length-frac) * fill + "| " + '{0:.1f}'.format(frac/length*100) + "%"
    return bar


class MetaboliteProcessor():
    """
    Helper class for processing the metabolites and storing the objects for
    future processing.

    Args:
    scale: The min max scaler will be fitted to [-range, range]
    nnif_factor: Factor for discarding features if they are all/almost all equal.
    Each feature requires at least number of observations/nnif_factor different features not to be discarded.
    """

    def __init__(self, scale=5, nnif_factor = 500):
        self.mms = MinMaxScaler([-scale, scale])
        self.nnif_factor = nnif_factor
        self.fitted = False

    def fit(self, metabolite_properties):
        """
        Fits the processor to a DataFrame of metabolite features with the correct format.
        """
        met_props = metabolite_properties.copy()
        self.mms.fit(metabolite_properties.iloc[:, 1:23])
        met_props.iloc[:, 1:23] = self.mms.transform(
            metabolite_properties.iloc[:, 1:23])
        fingerprint_features = (met_props["Fingerprint2D"]
                                .apply(self.to_bits_decode))
        fingerprint_features = pd.DataFrame(np.stack(fingerprint_features.values))
        length = metabolite_properties.shape[0]
        self.threshold = length//self.nnif_factor
        is_fp_always_zero = fingerprint_features.sum() < self.threshold
        is_fp_always_one = fingerprint_features.sum() > length - self.threshold
        self.is_informative = ~is_fp_always_zero & ~is_fp_always_one
        met_props = (pd.concat([met_props.drop("Fingerprint2D", axis=1).reset_index(drop=True),
                                           fingerprint_features.iloc[:,self.is_informative.to_numpy()]], axis=1))
        self.colmeans = met_props.mean(axis=0)
        self.cols_to_dummy = met_props.loc[:,met_props.isna().sum() >0].columns.to_numpy()
        self.fitted = True

    def transform(self, metabolite_properties):
        """
        Transforms a dataframe of metabolite properties.
        """
        assert self.fitted, "You are trying to transform data using a non-fitted processor"
        met_props = metabolite_properties.copy()
        met_props.iloc[:, 1:23] = self.mms.transform(metabolite_properties.iloc[:, 1:23])
        fingerprint_features = (met_props["Fingerprint2D"]
                                .apply(self.to_bits_decode))
        fingerprint_features = pd.DataFrame(np.stack(fingerprint_features.values))
        met_props = pd.concat([met_props.drop("Fingerprint2D", axis=1)
                                        .reset_index(drop=True),
                              fingerprint_features
                                        .iloc[:,self.is_informative.to_numpy()]], axis=1)
        for col in self.cols_to_dummy:
            colmean = self.colmeans[col]
            is_col_na = met_props[col].isna().astype(int)
            met_props[col+"is_na"] = is_col_na
        met_props = met_props.replace(np.nan, colmean)
        met_props = met_props.groupby("CID").min().reset_index()
        return met_props

    def to_bits_decode(self, seq):
        """
        Decodes the fingerprint as an array of numerical features.
        """
        decoded = base64.decodebytes(seq.encode('utf-8'))
        bits = []
        for x in decoded:
            bits += [i for i in str("{:08b}".format(x))]
        return(np.array(bits).astype(int))

    def save_fitted(self, saving_path="fitted_metproc.pkl"):
        """
        Saves the fitted preprocessor for later reuse
        """
        with open(saving_path, "wb") as f:
            pickle.dump(self, f)

class MetaboliteDownloader():
    """
    Helper class for downloading metabolite properties

    Args:
    use_cache: If true caches the successful requests and retries after failure.
    retries: Maximum number of failures before error.
    """

    def __init__(self, use_cache=True, retries=10):
        self.props = "MolecularWeight,XLogP,TPSA,Complexity,Charge,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,HeavyAtomCount,AtomStereoCount,BondStereoCount,CovalentUnitCount,Volume3D,XStericQuadrupole3D,YStericQuadrupole3D,ZStericQuadrupole3D,FeatureAcceptorCount3D,FeatureDonorCount3D,FeatureAnionCount3D,FeatureCationCount3D,FeatureRingCount3D,FeatureHydrophobeCount3D,Fingerprint2D"
        self.use_cache = use_cache
        if use_cache:
            random_suffix = str(uuid.uuid4())
            self.path = "./cache_" + random_suffix
            os.makedirs("./cache_" + random_suffix)
            self.errors = 0
        self.retries = retries

    def get(self, cids_, batch=64):
        """
        Downloads the cids from the pubchem API.
        """
        cids = cids_.astype(str)
        dfs = []
        for cid_i in np.arange(0, len(cids), batch):
            cid_subset = cids[cid_i:cid_i+batch]
            cid_string = ",".join(list(cid_subset))
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_string}/property/{self.props}/CSV"
            mets = pd.read_csv(url)
            path_csv = f"{self.path}/{cid_i}_cache_cids.csv"
            if self.use_cache & os.path.exists(path_csv):
                mets = pd.read_csv(path_csv, index_col=0)
                dfs.append(mets)
            else:
                successful = False
                while self.errors < self.retries and not successful:
                    try:
                        mets = pd.read_csv(url)
                        mets.to_csv(path_csv)
                        successful = True
                    except:
                        self.errors += 1
                        successful = False
                if not successful:
                    warnings.warn("Warning, not all cid could be fetched, this could lead to downstream errors")
                else:
                    dfs.append(mets)
        return pd.concat(dfs, axis=0, ignore_index=True)

    def exit(self):
        """
        Deletes the cache
        """
        shutil.rmtree(self.path)


def get_affix(dataset, fraction, permute, no_attn, no_trans):
    """
    Creates the model name in function of the configuration set
    """
    if permute:
        affix_p = "_permuted_prots"
    else:
        affix_p = ""
    if no_attn:
        affix_attn = "_noattn"
    else:
        affix_attn = ""

    if no_trans:
        affix_trans = "_notrans"
    else:
        affix_trans = ""
    RUN = "train_" + dataset + str(fraction * 100)+ affix_attn + affix_trans + affix_p
    return RUN
