import argparse
from PyTorchImageQualityTrainer import PyTorchImageQualityTrainer
from TensorFlowImageQualityTrainer import TensorFlowImageQualityTrainer

def main():
    parser = argparse.ArgumentParser(description="Train or load a model for image color classification.")
    parser.add_argument("--framework", type=str, choices=["pytorch", "tensorflow"], required=True, help="Choose the framework: pytorch or tensorflow.")
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--load", action="store_true", help="Load a pre-trained model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--model_path", type=str, default="model.pth", help="Path to save/load the model.")
    args = parser.parse_args()

    if args.framework == "pytorch":
        trainer = PyTorchImageQualityTrainer(root_dir='.')
        if args.train:
            trainer.load_dataset()
            trainer.initialize_model()
            trainer.train(epochs=args.epochs)
            torch.save(trainer.model.state_dict(), args.model_path)
            print(f"PyTorch model saved to {args.model_path}")
        elif args.load:
            trainer.model = PyTorchImageQualityTrainer.ColorCNN()
            trainer.model.load_state_dict(torch.load(args.model_path))
            trainer.model.eval()
            print(f"PyTorch model loaded from {args.model_path}")
        else:
            raise ValueError("Specify either --train or --load for PyTorch.")

    elif args.framework == "tensorflow":
        trainer = TensorFlowImageQualityTrainer(root_dir='.')
        if args.train:
            trainer.define_data_generators()
            trainer.define_and_compile_model()
            trainer.train(epochs=args.epochs)
            trainer.model.save(args.model_path)
            print(f"TensorFlow model saved to {args.model_path}")
        elif args.load:
            trainer.model = models.load_model(args.model_path)
            print(f"TensorFlow model loaded from {args.model_path}")
        else:
            raise ValueError("Specify either --train or --load for TensorFlow.")

if __name__ == "__main__":
    import torch
    from tensorflow.keras import models
    main()

'''
TO TRAIN THE MODEL:
python main.py --framework pytorch --train --epochs 10 --model_path pytorch_model.pth
python main.py --framework tensorflow --train --epochs 10 --model_path tensorflow_model.h5

TO LOAD THE PRE-TRAINED MODEL:
python main.py --framework pytorch --load --model_path pytorch_model.pth
python main.py --framework tensorflow --load --model_path tensorflow_model.h5

'''