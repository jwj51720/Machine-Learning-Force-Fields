import models


def get_models(config):
    if config["task"] == "force":
        if config["model"]["force"]["type"] == "MLP":
            cfg_model = config["model"]["force"]
            model = getattr(models, cfg_model["type"])(
                input_size=cfg_model["input_size"],
                hidden_size=cfg_model["hidden_size"],
                output_size=cfg_model["output_size"],
                dropout_rate=cfg_model["dropout_rate"],
            ).to(config["device"])
    elif config["task"] == "energy":
        if config["type"] == "BiLSTM":
            pass
    return model
