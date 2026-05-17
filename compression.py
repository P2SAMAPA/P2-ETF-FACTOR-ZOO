import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def double_lasso_compress(X, y, alpha1=0.01, alpha2=0.005):
    """
    Double Lasso: first Lasso on all factors, then second Lasso on selected.
    Returns selected factor names and the compressed model.
    """
    # First Lasso
    lasso1 = Lasso(alpha=alpha1, max_iter=1000)
    lasso1.fit(X, y)
    selected = np.where(lasso1.coef_ != 0)[0]
    if len(selected) == 0:
        # fallback: keep all
        selected = np.arange(X.shape[1])
    X_selected = X[:, selected]
    # Second Lasso on selected factors
    lasso2 = Lasso(alpha=alpha2, max_iter=1000)
    lasso2.fit(X_selected, y)
    final_coef = np.zeros(X.shape[1])
    final_coef[selected] = lasso2.coef_
    # Model: y_pred = X @ final_coef
    return final_coef, selected

def ppca_compress(X, y, n_components=20):
    """
    Probabilistic PCA: compress factors to lower dimension, then linear regression.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    lr = LinearRegression()
    lr.fit(X_pca, y)
    # Return a function that predicts new X
    def predict(new_X):
        new_X_scaled = scaler.transform(new_X)
        new_X_pca = pca.transform(new_X_scaled)
        return lr.predict(new_X_pca)
    return predict, pca, scaler, lr
