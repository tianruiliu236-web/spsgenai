import torch
from tqdm import tqdm

from helper_lib.evaluator import evaluate_model
from helper_lib.checkpoints import save_checkpoint


def train_model(
    model,
    train_loader,
    test_loader,
    epochs=5,
    learning_rate=0.001,
    device="cpu"
):
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    history = {
        "train_loss": [],
        "test_loss": [],
        "test_accuracy": []
    }

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        total_samples = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}"
        )

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            progress_bar.set_postfix(loss=loss.item())

        train_loss = running_loss / total_samples

        test_loss, test_accuracy = evaluate_model(
            model,
            test_loader,
            criterion,
            device
        )

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)

        checkpoint_path = save_checkpoint(
            model,
            optimizer,
            epoch,
            test_loss,
            test_accuracy
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Accuracy: {test_accuracy:.4f}"
        )

        print(f"Checkpoint saved: {checkpoint_path}")

    return model, history