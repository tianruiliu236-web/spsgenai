import os
import torch


def save_checkpoint(model, optimizer, epoch, loss, accuracy, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"cnn_epoch_{epoch + 1}.pth"
    )

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "accuracy": accuracy
        },
        checkpoint_path
    )

    return checkpoint_path


def load_checkpoint(model, optimizer, checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return (
        checkpoint["epoch"],
        checkpoint["loss"],
        checkpoint["accuracy"]
    )