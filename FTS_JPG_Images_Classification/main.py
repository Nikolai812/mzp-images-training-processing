import argparse
from config_reader import ConfigReader
from config_reader import Framework


def build_framework_manager(framework_name, config_reader):
    if framework_name == Framework.PYTORCH.name:
        from pytorch_manager import PyTorchManager

        config = config_reader.get_framework_config(Framework.PYTORCH)
        manager = PyTorchManager(config)

    elif framework_name == Framework.TENSORFLOW.name:
        from tensorflow_manager import TensorFlowManager

        config = config_reader.get_framework_config(Framework.TENSORFLOW)
        manager = TensorFlowManager(config)

    else:
        raise ValueError(
            f"Unsupported framework: {framework_name}"
        )

    return config, manager


def main():
    parser = argparse.ArgumentParser(
        description="Train or load a model for image color classification."
    )

    parser.add_argument(
        "-f",
        "--framework",
        type=str,
        choices=["PYTORCH", "TENSORFLOW"],
        required=True,
        help="Choose the framework: pytorch or tensorflow."
    )

    parser.add_argument(
        "-t",
        "--train",
        action="store_true",
        help="Train the model."
    )

    parser.add_argument(
        "-p",
        "--predict",
        action="store_true",
        help="Load a pre-trained model and make predictions."
    )

    args = parser.parse_args()

    config_reader = ConfigReader()
    config_reader.ensure_directories_exist()

    #
    # Framework-specific initialization only
    #

    config, manager = build_framework_manager(args.framework, config_reader)

    if args.train:
        manager.pre_train()
        manager.train()

    elif args.predict:
        prediction_model = config['predict_from_model_file']
        manager.load_model(prediction_model)
        manager.predict()

    else:
        raise ValueError(
            f"Specify either --train or --predict for {args.framework}."
        )


if __name__ == "__main__":
    main()