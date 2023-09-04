import models


def get_models(config):
    if config["task"] == "force":
        if config["force"]["model"]["type"] == "MLP":
            cfg_model = config["force"]["model"]
            model = getattr(models, cfg_model["type"])(
                input_size=cfg_model["input_size"],
                hidden_size=cfg_model["hidden_size"],
                output_size=cfg_model["output_size"],
                dropout_rate=cfg_model["dropout_rate"],
            ).to(config["device"])
    elif config["task"] == "energy":
        if config["energy"]["model"]["type"] == "BiLSTM":
            cfg_model = config["energy"]["model"]
            model = getattr(models, cfg_model["type"])(
                input_size=cfg_model["input_size"],
                hidden_size=cfg_model["hidden_size"],
                output_size=cfg_model["output_size"],
                dropout_rate=cfg_model["dropout_rate"],
                num_layers=cfg_model["num_layers"],
            ).to(config["device"])
    return model
