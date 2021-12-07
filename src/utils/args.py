import argparse


def get_args():
    """Retreive input arguments

    Input Arguments:
        --train_config: The path to the yaml config file used to initialise
            the train config object.
        --vit_model: The model to be used for. Available Models can be found in
            config/models.yaml

    """
    parser = argparse.ArgumentParser(description='Training Config')

    parser.add_argument(
        '--train_config',
        metavar='PATH',
        help='path to yaml config',
        default='train_config.yaml'
    )

    parser.add_argument(
        '--vit_model',
        type=str,
        metavar='MODEL',
        help='path to yaml config',
        default='vit-s_patch16',
    )

    return parser.parse_args()
