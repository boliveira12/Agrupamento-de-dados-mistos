import numpy as np
import pandas as pd
from elmoformanylangs import Embedder
from sklearn.metrics import pairwise_distances, silhouette_score, silhouette_samples
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel
import torch
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids

# Coocorrencia necessário para conseguir fazer o Ahmad


def cooccur(df):
    n = df.shape[1]
    res = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            res[i, j] = (df.iloc[:, i] == 1) & (df.iloc[:, j] == 1)
            res[j, i] = res[i, j]
    return res
# Codigo Ahmad que calcula a distancia de numerico+categorico
# Esse código é uma adaptação para Python da biblioteca Distmix do R

def distmix(data, method="ahmad", idnum=None, idbin=None, idcat=None):
    if data.isna().any().any():
        raise ValueError("Cannot handle missing values!")

    if not isinstance(data, (pd.DataFrame, pd.Series, np.ndarray)):
        raise ValueError(
            "The data must be a DataFrame, Series, or NumPy array!")

    if idnum is None and idbin is None and idcat is None:
        raise ValueError(
            "There is no distance computation, specify the numerical, binary, categorical variables!")

    if (idbin is None and idcat is None) or (idnum is None and idcat is None) or (idnum is None and idbin is None):
        raise ValueError("There is no mixed variables!")

    dist_num4 = ["gower", "wishart", "podani", "huang", "harikumar", "ahmad"]
    method = method if method in dist_num4 else "ahmad"

    if (idbin is not None and idcat is not None and len(idbin) + len(idcat) == 1) and method == "ahmad":
        raise ValueError(
            "Ahmad-Dey distance cannot be calculated because the combined binary and categorical variable is only 1 variable!")

    if idnum is None:
        num = 0
        msd = 0
        dist_numeric = 0
    else:
        num = len(idnum)
        msd = np.mean(data[idnum].std())
        x = data[idnum].to_numpy()
        dist_numeric = {
            "gower": pairwise_distances(x, x, metric="manhattan"),
            "wishart": pairwise_distances(x, x, metric="euclidean"),
            "podani": pairwise_distances(x, x, metric="euclidean"),
            "huang": pairwise_distances(x, x, metric="euclidean"),
            "harikumar": pairwise_distances(x, metric="manhattan"),
            "ahmad": pairwise_distances(x, x, metric="euclidean")
        }[method]

    if idbin is None:
        bin = 0
        dist_binary = 0
    else:
        bin = len(idbin)
        dist_matchbin = pd.DataFrame(index=data.index, columns=data.index)
        for col in idbin:
            dist_matchbin[col] = (data[col] != data[idbin]).sum(axis=1)
        if method == "ahmad":
            dist_binary = cooccur(
                pd.concat([data[idbin], data[idcat]], axis=1))
        else:
            if method in ["huang", "harikumar"]:
                dist_binary = dist_matchbin * bin
            else:
                dist_binary = dist_matchbin

    if idcat is None:
        cat = 0
        dist_cat = 0
    else:
        cat = len(idcat)
        dist_matchcat = pd.DataFrame(index=data.index, columns=data.index)
        for col in idcat:
            if method == "huang":
                dist_cat = dist_matchcat * cat
            else:
                if method == "ahmad":
                    dist_cat = dist_binary
                else:
                    dist_cat = dist_matchcat

    nvar = num + bin + cat
    dist_mix = {
        "gower": dist_numeric * 1/nvar + dist_binary * bin/nvar + dist_cat * cat/nvar,
        "wishart": (dist_numeric * 1/nvar + dist_binary * bin/nvar + dist_cat * cat/nvar)**0.5,
        "podani": (dist_numeric + dist_binary * bin + dist_cat * cat)**0.5,
        "huang": dist_numeric + dist_binary * msd + dist_cat * msd,
        "harikumar": dist_numeric + dist_binary + dist_cat,
        "ahmad": (dist_numeric + dist_binary)**2
    }[method]

    return dist_mix

# Codigo do Bert pretreinado 'bert-base-portuguese-cased'
def bert_txt(df, txt_column=None):
    tokenizer = BertTokenizer.from_pretrained(
        "neuralmind/bert-base-portuguese-cased")
    model = BertModel.from_pretrained("neuralmind/bert-base-portuguese-cased")

    tokenized_texts = tokenizer(df[txt_column].astype(str).tolist(),
                                add_special_tokens=True,
                                truncation=True,
                                max_length=128,
                                padding=True,
                                return_tensors='pt')

    model.eval()

    embeddings = []
    with torch.no_grad():
        outputs = model(**tokenized_texts)
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()

    return embeddings

#Gera um dos embeddings
def gera_embbeding(data, modelo, coluna):
    if modelo == 'elmo':
        e = Embedder('elmoformanylangs/168/')
        sents = [data[coluna]]
        txt_embedding_elmo = e.sents2elmo(sents)
        return txt_embedding_elmo[0]
    else:
        txt_embeddings_bert = bert_txt(df=data, txt_column=coluna)
        return txt_embeddings_bert

# Cria coluna PCA
def criando_coluna_PCA(df, modelo, col, result_embbe, pca_k):
    i = 0
    pca = PCA(n_components=pca_k)
    PCA_result = pca.fit_transform(result_embbe)
    PCA_result = pd.DataFrame(PCA_result)
    for i in range(pca_k):
        df[f'{modelo}{col}PC{i+1}'] = PCA_result.iloc[0:, i]
        i += 1
    return df
