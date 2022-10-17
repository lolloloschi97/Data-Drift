from tqdm import tqdm
from config import *
from torch import nn
from typing import Union, Dict, List
from ahead.utils import OutputColumnsHandler
from ahead.pred import Predictor
from ahead.nn import Net1Conv
import copy
import numpy as np

def enable_MCD(model: Union[Net1Conv,nn.Module],
               p:float = 0.3):
    '''
    :param model: input model
    :param p: dropout probability
    :return: model with enabled dropout
    '''
    if isinstance(model,Net1Conv):
        mod = [model.features_conv, model.classifier0, model.classifier]
        for m in mod:
            for layer in m.modules():
                if layer.__class__.__name__ == 'Dropout':
                    layer.p = p
                    layer.train()
        return model
    else:
        mod = copy.deepcopy(model)
        for layer in mod.modules():
            if layer.__class__.__name__ == 'Dropout':
              layer.p = 0.3
              layer.train()
        return mod

def get_monte_carlo_predictions(dataset,
                                config_path: str,
                                map_col: Union[Dict, OutputColumnsHandler],
                                n_classes: int = 14,
                                B: int = 100,
                                p: float = 0.8):
    """ Function to get the monte-carlo samples and uncertainty estimates
    through multiple forward passes

    Parameters
    ----------
    dataset: pd.Dataframe Dataset over wich compute dropout
    config_path: path to the pred configuartion
    map_col : dictionary with the mapping of the columns
    n_classes : int number of classes in the dataset
    B : int  number of iteration of Dropout
    p : float probability od Dropout
    """
    pred_fetures = list(OutputColumnsHandler(map_col).cols_type('pred').values())[0]
    predictor = Predictor(config=config_path, model_type='nn', task='test')
    n_samples = len(dataset)
    predictor.model = enable_MCD(predictor.model, p)
    dropout_pred = np.empty((B, n_samples, n_classes))
    for b in tqdm(range(B), desc ="Iteration: "):
        predictor.predict(dataset)
        dropout_pred[b,:,:] = predictor.df[pred_fetures].to_numpy()

    # Calculating variance across multiple MCD forward passes
    variance = np.var(dropout_pred, axis=0) # shape (n_samples, n_classes)
    return dropout_pred,variance
