import copy
from config import *
import nannyml as nml
from dataset_preparation import compute_month_day


def univariate_drift(pred,metadata, chunk_size = None,month_day = False):
    analysis = pred[pred['partition'] == 'analysis']
    reference = pred[pred['partition'] == 'reference']

    if month_day:
        reference['m_d'] = compute_month_day(reference, 'dat_trasporto')
        reference = reference[reference['m_d'].isin(compute_month_day(analysis, 'dat_trasporto'))].drop(columns='m_d')

    univariate_calculator = nml.UnivariateStatisticalDriftCalculator(model_metadata = metadata,  chunk_period = "M")
    univariate_calculator = univariate_calculator.fit(reference_data=reference)
    data = pd.concat([reference, analysis], ignore_index=True)
    univariate_results = univariate_calculator.calculate(data=data)
    ranker = nml.Ranker.by('alert_count')
    rank = ranker.rank(univariate_results, model_metadata=metadata, only_drifting=False)
    return rank, univariate_results

def multivariate_drift(pred,metadata, chunk_size = None, month_day = False):
    analysis = pred[pred['partition'] == 'analysis']
    reference = pred[pred['partition'] == 'reference']
    if month_day:
        reference['m_d'] = compute_month_day(reference, 'dat_trasporto')
        reference = reference[reference['m_d'].isin(compute_month_day(analysis, 'dat_trasporto'))].drop(columns='m_d')

    rcerror_calculator = nml.DataReconstructionDriftCalculator(model_metadata=metadata,  chunk_period = "M")
    rcerror_calculator = rcerror_calculator.fit(reference_data = reference)
    data = pd.concat([reference, analysis], ignore_index=True)
    rcerror_results = rcerror_calculator.calculate(data=data)
    return rcerror_results
