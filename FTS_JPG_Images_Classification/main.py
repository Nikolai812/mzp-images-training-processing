import argparse
from config_reader import ConfigReader
from config_reader import Framework


def build_framework_manager(framework_value: str, config_reader):
    # several configs nay use pytorch (for jpg, fts, diff categories)
    if "PYTORCH" in framework_value:
        from pytorch_manager import PyTorchManager
        print(f"Going to load PYTORCH for configuration: {framework_value}")
        config = config_reader.get_framework_config(framework_value)
        manager = PyTorchManager(config)

    # several configs nay use tensorflow (for jpg, fts, diff categories)
    elif "TENSORFLOW" in framework_value:
        from tensorflow_manager import TensorFlowManager
        print(f"Going to load TENSORFLOW for configuration: {framework_value}")
        config = config_reader.get_framework_config(framework_value)
        manager = TensorFlowManager(config)

    else:
        raise ValueError(
            f"Unsupported framework: {framework_value}"
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
        # Choice type may be removed in future to add configurations without hardcoding the choice options
        #choices=["PYTORCH", "TENSORFLOW", "PYTORCH_FTS"],
        choices=[f.value for f in Framework],
        required=True,
        help="Choose the framework-related configuratiom: pytorch or tensorflow and jpg/fts images. Configuration includes paths for training images and for images for predictions"
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