from sklearn.neighbors import LocalOutlierFactor
from joblib import dump, load
import numpy as np

paras = {}
paras["n_time_windows"] = 100
paras["k"] = 20
paras["vd"] = 100
paras["ws"] = 5
paras["epoch"] = 100
paras["alpha"] = 0.025
paras["n_neighbors"] = 20
paras["contamination"] = 0.04
paras["mpl"] = 10

def main(cfg):
    train_pv_path = cfg.featurization.embed_edges._edge_embeds_dir + "trainembeddings.npy"
    train_pv = np.load(train_pv_path)
    clf = LocalOutlierFactor(novelty=True, n_neighbors=paras["n_neighbors"], contamination=paras["contamination"]).fit(
    train_pv)
    dump(clf, cfg.detection.gnn_training._trained_models_dir+"provd_model.pkl")
