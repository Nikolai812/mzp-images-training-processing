import argparse
from config_reader import ConfigReader


def main():
    parser = argparse.ArgumentParser(
        description="Train or load a model for image color classification."
    )

    parser.add_argument(
        "-f",
        "--framework",
        type=str,
        choices=["pytorch", "tensorflow"],
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

    if args.framework == "pytorch":
        from pytorch_manager import PyTorchManager

        config = config_reader.get_pytorch_config()
        manager = PyTorchManager(config)

        model_file = config["model_file"]
        epochs = int(config["epochs"])

        predict_model_file = config.get(
            "predict_from_model_file",
            ""
        ).strip()

        prediction_model = (
            predict_model_file
            if predict_model_file
            else model_file
        )

        if args.train:
            manager.load_dataset()
            manager.initialize_model()
            manager.train()

        elif args.predict:
            manager.load_model(prediction_model)
            manager.predict()

        else:
            raise ValueError(
                "Specify either --train or --predict for PyTorch."
            )

    elif args.framework == "tensorflow":
        from tensorflow_manager import TensorFlowManager

        config = config_reader.get_tensorflow_config()
        manager = TensorFlowManager(config)

        model_file = config["model_file"]
        epochs = int(config["epochs"])

        predict_model_file = config.get(
            "predict_from_model_file",
            ""
        ).strip()

        prediction_model = (
            predict_model_file
            if predict_model_file
            else model_file
        )

        if args.train:
            manager.define_data_generators()
            manager.define_and_compile_model()

            manager.train()

        elif args.predict:
            manager.load_model(prediction_model)
            manager.predict()

        else:
            raise ValueError(
                "Specify either --train or --predict for TensorFlow."
            )


if __name__ == "__main__":
    main()