import nannyml as nml
from config import *
from ahead.metrics import evaluate_errors
from ahead.utils import OutputColumnsHandler


def graph_dataset(data,map_col,metric = METRIC, baseline = False):
  '''
  Given a dataset, extract metric column and melt the dataset for plot putposes.
  data - Dataframe
  map_col - Dict returned by ahead prediction_main
  metric_list - list containing the metric of interest -- mape,mae,me
  id_list - subset of id
  '''
  if baseline == False:
      prefix = 'pred'
      df, map_col_err = evaluate_errors(data, map_col, metrics_name = ['mape','mae'], y_true='real', y_pred='pred', prefix = prefix)
  else:
      prefix = 'baseline'
      df, map_col_err = evaluate_errors(data, map_col, metrics_name= ['mape','mae'], y_true='real', y_pred='baseline', prefix = prefix)

  metric_columns = [OutputColumnsHandler(map_col_err).col('val_qta_kg', str(el), prefix + '_' + metric) for el in
                    range(14)]
  df = df[['dat_trasporto','id'] + metric_columns]
  dict_metric = dict(zip(metric_columns,np.arange(1,15).tolist()))

  df = pd.melt(df,id_vars = ['dat_trasporto','id'],value_vars = metric_columns, var_name = 'steps')
  df['steps'] = df['steps'].map(dict_metric)
  df = df.sort_values(by = ['steps','value']) # sorted
  df.columns = df.columns.str.replace('value', metric)
  return df


def slice_dataset(data,date1 = None, date2 = None,id = None):
  '''
  Given a Dataset return a subset of specifc date and or ids
  data -> DataFrame
  data1,data2 = datetime.datetime()
  '''
  NoneType = type(None)
  if date1 != None and date2 != None:
      datalist = pd.date_range(start=date1, end=date2, freq='D').tolist()
      if not isinstance(id, NoneType):
          data = data[(data['id'].isin(id)) & (data['dat_trasporto'].isin(datalist))]
      else:
          data = data[data['dat_trasporto'].isin(datalist)]
  elif date1 != None:
      datalist = [date1]
      if not isinstance(id, NoneType):
          data = data[(data['id'].isin(id)) & (data['dat_trasporto'].isin(datalist))]
      else:
          data = data[data['dat_trasporto'].isin(datalist)]
  elif date2 != None:
      datalist = [date2]
      if not isinstance(id, NoneType):
          data = data[(data['id'].isin(id)) & (data['dat_trasporto'].isin(datalist))]
      else:
          data = data[data['dat_trasporto'].isin(datalist)]
  else:
      if not isinstance(id, NoneType):
          data = data[data['id'].isin(id)]
  return data

def last_year(data_now):
  ''' data_now: datatime '''
  if data_now == None:
    return None
  delta = pd.Timedelta(days=365)
  return data_now-delta


def data_preparation(curr_df,ref_df,map_col_curr,map_col_ref, date1,date2 = None, id_list = ID_LIST,metric = METRIC, baseline = BASELINE):
    current_melt = graph_dataset(curr_df, map_col_curr, metric = metric)
    reference_melt = graph_dataset(ref_df, map_col_ref, metric = metric, baseline = baseline)
    current_melt_id = slice_dataset(current_melt, date1,date2, id = id_list)
    reference_melt_id = slice_dataset(reference_melt, last_year(date1),last_year(date2),id = id_list)

    melt_id = pd.concat([reference_melt_id, current_melt_id])
    melt_id['period'] = np.append(np.ones(len(reference_melt_id)), np.zeros(len(current_melt_id)))
    melt_id['period'] = melt_id['period'].map({1: 'reference', 0: 'current_year'})
    return melt_id,current_melt_id, reference_melt_id

### NUNNYML

def compute_month_day(df, date_column):
    '''
    Given a Dataframe and its date column in datetime format, parse each date as MM-DD.
    es: 2018-09-02 --> 9-2
    '''

    m_d = df[date_column].apply(lambda x: str(x.month) + '-' + str(x.day)).to_numpy()
    return m_d

def metadata_generation(pred,feature_cols, info_cols = info_columns):
    info_cols = list(set(info_cols))
    df_prediction = pred[feature_cols + info_cols]
    metadata = nml.extract_metadata(df_prediction, model_type='classification_binary', exclude_columns=info_cols)
    metadata.target_column_name = 'y_true'
    metadata.prediction_column_name = 'y_pred'
    metadata.predicted_probability_column_name = 'predicted_probability'
    metadata.partition_column_name = 'partition'
    metadata.timestamp_column_name = 'dat_trasporto'
    return metadata


def data_drift_prep(curr_df,ref_df, date1 = None, date2 = None, id_list = None, feature_cols = None):
    current_id = slice_dataset(curr_df, date1, date2, id = id_list)
    reference_id = slice_dataset(ref_df,id = id_list)
    current_id['partition'] = np.array(['analysis' for c in range(len(current_id))])
    reference_id['partition'] = np.array(['reference' for c in range(len(reference_id))])

    prediction = pd.concat([current_id, reference_id]).reset_index()
    prediction = prediction.sort_values(by='dat_trasporto')
    prediction['partition'] = np.append(np.array(['reference' for c in range(len(reference_id))]),
                                    np.array(['analysis' for c in range(len(current_id))]))
    prediction['predicted_probability'] = np.random.random_sample((len(prediction),))
    prediction['y_pred'] = np.random.randint(0,1,len(prediction))
    prediction['y_true'] = np.random.randint(0,1,len(prediction))
    metadata = metadata_generation(prediction, feature_cols)
    return prediction, metadata


