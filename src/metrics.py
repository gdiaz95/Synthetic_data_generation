
from mostlyai import qa
import os


'''
Resources:
- https://arxiv.org/html/2501.03941v1
- https://www.nature.com/articles/s41746-023-00771-5 uses median for NNDR
'''

def get_metrics(data_train, data_gen, data_holdout):

    # calculate metrics
    _, metrics = qa.report(
        syn_tgt_data=data_gen,
        trn_tgt_data=data_train,
        hol_tgt_data=data_holdout,
    )
    os.remove('model-report.html')
    return metrics