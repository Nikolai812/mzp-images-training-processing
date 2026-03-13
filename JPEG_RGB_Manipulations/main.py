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
        from pytorch_manager import PyTorchManager
        manager = PyTorchManager(root_dir='./TRAINING_INPUT')

        if args.train:
            manager.load_dataset()
            manager.initialize_model()
            manager.train(epochs=args.epochs)
            pytorch_subdir = 'PYTORCH'
            pytorch_models_path = os.path.join(models_dir, pytorch_subdir)
            os.makedirs(pytorch_models_path, exist_ok=True)
            model_filename = os.path.join(pytorch_models_path, args.model_path)
            manager.save(model_filename)

        elif args.predict:
            model_filename = os.path.join(models_dir, 'PYTORCH', args.model_path)
            manager.load_model(model_filename)
            results = manager.predict(input_dir='./RAW_INPUT')
            manager.save_results(results, "output_dict.json")

        else:
            raise ValueError("Specify either --train or --predict for PyTorch.")

    elif args.framework == "tensorflow":
        from tensorflow_manager import TensorFlowManager
        manager = TensorFlowManager(root_dir='./TRAINING_INPUT')
        if args.train:
            manager.define_data_generators()
            manager.define_and_compile_model()
            tensorflow_subdir = 'TENSORFLOW'
            tensorflow_models_path = os.path.join(models_dir, tensorflow_subdir)
            os.makedirs(tensorflow_models_path, exist_ok=True)
            model_filename = os.path.join(tensorflow_models_path, args.model_path)
            manager.train(epochs=args.epochs, model_path=model_filename)
        elif args.predict:
            model_filename = os.path.join(models_dir, 'TENSORFLOW', args.model_path)
            manager.load_model(model_filename)
            results = manager.predict(input_dir='./RAW_INPUT', output_file="output_dict_tf.json")
        else:
            raise ValueError("Specify either --train or --predict for TensorFlow.")

if __name__ == "__main__":
    main()
