import argparse
import os
from PyTorchImageQualityTrainer import PyTorchImageQualityTrainer
from TensorFlowImageQualityTrainer import TensorFlowImageQualityTrainer

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
        trainer = PyTorchImageQualityTrainer(root_dir='.')
        pytorch_subdir = 'PYTORCH'
        pytotch_models_path = os.path.join(models_dir, pytorch_subdir)
        os.makedirs(pytotch_models_path, exist_ok=True)
        if args.train:
            trainer.load_dataset()
            trainer.initialize_model()
            trainer.train(epochs=args.epochs)
            model_filename = os.path.join(models_dir, pytorch_subdir, args.model_path)
            torch.save(trainer.model.state_dict(), model_filename)
            print(f"PyTorch model saved to {args.model_path}")
        elif args.predict:
            trainer.model = PyTorchImageQualityTrainer.ColorCNN()
            model_filename = os.path.join(models_dir, pytorch_subdir, args.model_path)
            trainer.model.load_state_dict(torch.load(model_filename))
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
        elif args.predict:
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

TO LOAD THE PRE-TRAINED MODEL AND MAKE PREDICTIONS:
python main.py --framework pytorch --predict --model_path pytorch_model.pth
python main.py --framework tensorflow --predict --model_path tensorflow_model.h5

'''