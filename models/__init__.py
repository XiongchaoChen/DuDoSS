from models import cas_model, casrec_model, cnn_model


def create_model(opts):
    if opts.model_type == 'model_casrec':
        model = casrec_model.CasRecModel(opts)

    elif opts.model_type == 'model_cas':
        model = cas_model.CasModel(opts)

    elif opts.model_type == 'model_cnn':
        model = cnn_model.CNNModel(opts)

    else:
        raise NotImplementedError

    return model
