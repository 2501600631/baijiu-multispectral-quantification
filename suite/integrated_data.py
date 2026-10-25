import numpy as np
from .common import digest

def as_ethanol_target(data):
    d = dict(data)
    d['Y'] = np.asarray(data['ethanol'], float).reshape(-1, 1).copy()
    d['target_names'] = ['ethanol']
    d['target_labels'] = ['Measured alcohol strength (source scale; unit unconfirmed)']
    d['data_audit'] = dict(data['data_audit'])
    d['data_audit']['fingerprint'] = digest({'source': data['data_audit'].get('fingerprint'), 'ids': data['ids'], 'X': data['X'], 'Y': d['Y'], 'target': 'ethanol'})
    d['data_audit']['prediction_input'] = 'spectra_only; measured ethanol target never passed to model as feature'
    d['data_audit']['concentration_unit'] = 'original_author_workbook_scale_not_g_L'
    return d
