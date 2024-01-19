# from . import rwj_dataset, nih_dataset, sv_dataset, mar_dataset, ld_dataset
from . import lv_dataset


def get_datasets(opts):
    if opts.dataset == 'LV':
        trainset = lv_dataset.LVTrain(opts)
        valset = lv_dataset.LVVal(opts)
        testset = lv_dataset.LVTest(opts)

    else:
        raise NotImplementedError

    return trainset, valset, testset
