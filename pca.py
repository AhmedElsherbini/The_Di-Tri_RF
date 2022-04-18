#import lib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utils import *


def PCA_model(f_name):
    data, x, y1, y2, _, t1, t2, _ = load_train_data(f_name)
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    p_Components = pca.fit_transform(x)
    pc_df = pd.DataFrame(p_Components, columns = ['pc1', 'pc2'])
    y = pd.DataFrame(np.asarray(y1), columns= ['class'])
    finalDf = pd.concat([pc_df, y[['class']]], axis = 1,)
    plot_2d_pca(pca, finalDf, t1, f_name= 'Clade')

    y = pd.DataFrame(np.asarray(y2), columns= ['class'])
    finalDf = pd.concat([pc_df, y[['class']]], axis = 1,)
    plot_2d_pca(pca, finalDf, t2, f_name='Cont')
