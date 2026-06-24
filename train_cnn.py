import os
import torch

from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_model
from helper_lib.utils import get_device


def main():
    device = get_device()

    print(f"Using device: {device}")

    train_loader = get_data_loader(
        data_dir="data",
        batch_size=64,
        train=True
    )

    test_loader = get_data_loader(
        data_dir="data",
        batch_size=64,
        train=False
    )

    model = get_model("CNN")

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=5,
        learning_rate=0.001,
        device=device
    )

    os.makedirs("models", exist_ok=True)

    model_path = "models/cifar10_cnn.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": [
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck"
            ],
            "history": history
        },
        model_path
    )

    print(f"Final model saved to: {model_path}")


if __name__ == "__main__":
    main()