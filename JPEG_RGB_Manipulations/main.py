import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Train or load a model for image color classification.")
    parser.add_argument("-f", "--framework", type=str, choices=["pytorch", "tensorflow"], required=True, help="Choose the framework: pytorch or tensorflow.")
    parser.add_argument("-t", "--train", action="store_true", help="Train the model.")
    parser.add_argument("-p", "--predict", action="store_true", help="Load a pre-trained model and make predictions.")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("-m", "--model_path", type=str, default="model.pth", help="Path to save/load the model.")
    args = parser.parse_args()

    models_dir = "./MODELS/"
    os.makedirs(models_dir, exist_ok=True)

    if args.framework == "pytorch":
        import torch
        from PyTorchImageQualityTrainer import PyTorchImageQualityTrainer

        trainer = PyTorchImageQualityTrainer(root_dir='.')
        pytorch_subdir = 'PYTORCH'
        pytorch_models_path = os.path.join(models_dir, pytorch_subdir)
        os.makedirs(pytorch_models_path, exist_ok=True)

        if args.train:
            trainer.load_dataset()
            trainer.initialize_model()
            trainer.train(epochs=args.epochs)
            model_filename = os.path.join(pytorch_models_path, args.model_path)
            torch.save(trainer.model.state_dict(), model_filename)
            print(f"PyTorch model saved to {model_filename}")

        elif args.predict:
            trainer.model = PyTorchImageQualityTrainer.ColorCNN()
            model_filename = os.path.join(pytorch_models_path, args.model_path)
            trainer.model.load_state_dict(torch.load(model_filename))
            trainer.model.eval()
            print(f"PyTorch model loaded from {model_filename}")

            # Add prediction logic here (e.g., call a prediction function)
            # Example: predict_and_save(trainer.model, input_dir="./RAWINPUT", output_file="output_dict.json")

        else:
            raise ValueError("Specify either --train or --predict for PyTorch.")

    elif args.framework == "tensorflow":
        from tensorflow.keras import models
        from TensorFlowImageQualityTrainer import TensorFlowImageQualityTrainer

        trainer = TensorFlowImageQualityTrainer(root_dir='.')

        if args.train:
            trainer.define_data_generators()
            trainer.define_and_compile_model()
            trainer.train(epochs=args.epochs)
            trainer.model.save(args.model_path)
            print(f"TensorFlow model saved to {args.model_path}")

        elif args.predict:
            trainer.model = models.load_model(args.model_path)
            print(f"TensorFlow model loaded from {args.model_path}")

            # Add prediction logic here (e.g., call a prediction function)

        else:
            raise ValueError("Specify either --train or --predict for TensorFlow.")

if __name__ == "__main__":
    main()
