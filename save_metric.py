import copy
from config import *
from typing import Union, Dict
from ahead.utils import OutputColumnsHandler, write_file
from ahead.configuration_handlers import  PredictionConfigurationHandler



def compute_percentile(df,
                       map_columns: Union[Dict, OutputColumnsHandler],
                       metrics_name: str = None,
                       percentili : list = [0.9,0.95,0.97],
                       save_out: bool = False,
                       config_path: Union[PredictionConfigurationHandler,dict,str] = ''):

    metric_col = [OutputColumnsHandler(map_columns).col('val_qta_kg',str(el),metrics_name) for el in range(14)]
    values =  []
    values_name = []
    for p in percentili:
        temp = copy.deepcopy(df[['id']+ metric_col].groupby('id').quantile(p))
        values.append(temp)
        values_name.append('percentile_'+str(p))
    if save_out:
        predictor_conf = PredictionConfigurationHandler(config_path,'nn')
        values_dict = [df.T.to_dict() for df in values]
        out_file = dict(zip(values_name,values_dict))
        out_path = predictor_conf.out_path('percentile_output.json')
        write_file(data = out_file,file_path = out_path)
    return dict(zip(values_name,values))

def compute_dict_metrics(df,
                           map_columns: Union[Dict, OutputColumnsHandler],
                           metrics_name: str = None,
                           percentili : list = [0.9,0.95,0.97],
                           save_out: bool = False,
                           config_path: Union[PredictionConfigurationHandler,dict,str] = ''):
    metric_col = [OutputColumnsHandler(map_columns).col('val_qta_kg',str(el),metrics_name) for el in range(14)]
    values =  []
    for p in percentili:
        temp = copy.deepcopy(df[['id']+ metric_col].groupby('id').quantile(p))
        temp['percentile'] = np.array(['percentile_'+str(p)for c in range(len(temp))])
        values.append(temp)
    values = pd.concat(values)
    values.set_index(pd.MultiIndex.from_arrays([list(values.index),values['percentile']],names = ['id','percentile']), inplace=True)
    diz = {}
    for id in values.index.get_level_values('id'):
        diz[id] = values.xs(id,level = 'id')[metric_col].T.to_dict()

    if save_out:
        predictor_conf = PredictionConfigurationHandler(config_path,'nn')
        out_path = predictor_conf.out_path('percentile_output.json')
        write_file(data = diz,file_path = out_path)
    return diz

def load_dict(dict_metrics:dict):
    df = []
    for key,val in list(dict_metrics.items()):
        temp = pd.DataFrame(val).T
        id = pd.MultiIndex.from_product([[key],list(temp.index)], names = ['id','percentile'])
        temp.set_index(id, inplace = True)
        df.append(temp)
    return pd.concat(df)

def compute_percentile_2(df,
                       map_columns: Union[Dict, OutputColumnsHandler],
                       metrics_name: str = None,
                       percentili : list = [0.9,0.95,0.97],
                       save_out: bool = False,
                       config_path: Union[PredictionConfigurationHandler,dict,str] = ''):

    metric_col = [OutputColumnsHandler(map_columns).col('val_qta_kg',str(el),metrics_name) for el in range(14)]
    values =  []
    for p in percentili:
        temp = copy.deepcopy(df[['id']+ metric_col].groupby('id').quantile(p))
        temp['percentile'] = np.array(['percentile_'+str(p)for c in range(len(temp))])
        values.append(temp)
    values = pd.concat(values)
    values.set_index(pd.MultiIndex.from_arrays([list(values.index),values['percentile']],names = ['id','percentile']), inplace=True)
    out_file = values.to_dict()
    if save_out:
        predictor_conf = PredictionConfigurationHandler(config_path,'nn')
        out_path = predictor_conf.out_path('percentile_output.json')
        write_file(data = out_file,file_path = out_path)
    return out_file

def compare_mae(df1,
                df2,
                w,
                map_columns: Union[Dict, OutputColumnsHandler],
                metrics_name: str = None,):
    datelist = df1['dat_trasporto'].unique()
    metric_col = [OutputColumnsHandler(map_columns).col('val_qta_kg', str(el), metrics_name) for el in range(14)]
    df_list = []
    for date in datelist:
        df1_temp = df1[df1['dat_trasporto'] == date]
        index = df1_temp.index
        df = pd.DataFrame(np.where(df2[df2.index.isin(index)][metric_col] < df1_temp[metric_col],1,0), index = index, columns=metric_col)
        df['w'] = w
        df['prob'] = df[metric_col].mean(axis = 1).to_numpy()
        df['w_prob'] = df['prob']*df['w']
        df.set_index(pd.MultiIndex.from_product([list(df1_temp.index),[date]],names = ['id','dat_trasporto']), inplace=True)
        df_list.append(df)
    df_list = pd.concat(df_list)
    return df_list