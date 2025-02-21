from loguru import logger

def load_config(config):
    """
    Load configuration file
    """
    # resize transform to imsize from mae
    if config["models"]["mae"]["image_dim"]:
        config["transforms"]["resize"] = (config["models"]["mae"]["image_dim"], config["models"]["mae"]["image_dim"])
    else:
        logger.warning("Image dimension not specified, using default value of 224")
        config["transforms"]["resize"] = (224, 224)

    # num_patches to num_patches from mae
    if config["models"]["mae"]["n_patches"]:
        config["pre-training"]["n_patches"] = config["models"]["mae"]["n_patches"]
    else:
        logger.warning("Number of patches not specified, using default value of 16")
        config["pre-training"]["n_patches"] = 16
        
    return config
