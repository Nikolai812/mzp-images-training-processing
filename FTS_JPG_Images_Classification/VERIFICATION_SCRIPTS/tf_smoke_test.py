from tensorflow_manager import TensorFlowManager


def smoke_test_manager(manager: TensorFlowManager):
    for path in [
        r"./RAW_INPUT/rgb_4_82_213_256x256.jpg",
        r"./RAW_INPUT_FTS/AA/polar_at-12_-0005bias.fit",
    ]:
        image = manager.load_image(path)
        print(
            path,
            "shape:", image.shape,
            "dtype:", image.dtype,
            "range:", (image.min(), image.max()),
        )

    pass