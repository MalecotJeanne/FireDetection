from models.mae import MAE
from models.encoders import TransformerEncoder, CNNEncoder
from models.decoders import TransformerDecoder

models = {
    'mae': MAE,
}

encoders = {
    'transformer': TransformerEncoder,
    'cnn': CNNEncoder,
}

decoders = {
    'transformer': TransformerDecoder
}

def init_model(name, encoder_name, decoder_name, *args, **kwargs):
    if name not in models.keys():
        raise KeyError("Unknown models: {}".format(name)) 
    if encoder_name not in encoders.keys():
        raise KeyError("Unknown encoder: {}".format(encoder_name))
    if decoder_name not in decoders.keys():
        raise KeyError("Unknown decoder: {}".format(decoder_name))
    
    config = kwargs.pop('config', None)
    if config is None:
        raise ValueError("No configuration provided")
    config_model = config["models"][name] 
    config_encoder = config["encoders"][encoder_name]
    config_decoder = config["decoders"][decoder_name]

    config_model['encoder'] = config_encoder
    config_model['decoder'] = config_decoder
        
    kwargs['config'] = config_model

    return models[name](encoders[encoder_name], decoders[decoder_name], *args, **kwargs)
