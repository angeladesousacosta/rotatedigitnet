import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import RotatedDataset
from model import RotateDigitNet
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms


def main():
    root = Path.cwd()
    current = root.joinpath('dataset')

    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_dataset = RotatedDataset(
        csv_file=current.joinpath('training.csv'),
        root_path=current.joinpath('training'),
        transform=transform
    )

    val_dataset = RotatedDataset(
        csv_file=current.joinpath('validation.csv'),
        root_path=current.joinpath('validation'),
        transform=transform
    )

    test_dataset = RotatedDataset(
        csv_file=current.joinpath('testing.csv'),
        root_path=current.joinpath('testing'),
        transform=transform
    )

    batch_size = 64

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = RotateDigitNet().to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    criterion_digit = nn.CrossEntropyLoss()
    criterion_angle = nn.SmoothL1Loss()

    epochs = 10

    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        for images, angles, digits in train_loader:
            images = images.to(device)
            angles = angles.to(device)
            digits = digits.to(device)

            digit_predictions, angle_predictions = model(images)

            digit_loss = criterion_digit(digit_predictions, digits)
            angle_loss = criterion_angle(angle_predictions.squeeze(-1), angles)

            loss = digit_loss + angle_loss
            train_loss += loss.item()

            _, predicted_digits = torch.max(digit_predictions, 1)
            correct_train += (predicted_digits == digits).sum().item()
            total_train += digits.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_accuracy = (correct_train / total_train) * 100
        train_losses.append(train_loss / len(train_loader))
        train_acc.append(train_accuracy)


        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, angles, digits in val_loader:
                images = images.to(device)
                angles = angles.to(device)
                digits = digits.to(device)

                digit_predictions, angle_predictions = model(images)

                digit_loss = criterion_digit(digit_predictions, digits)
                angle_loss = criterion_angle(angle_predictions.squeeze(-1), angles)

                loss = digit_loss + angle_loss
                val_loss += loss.item()

                _, predicted_digits = torch.max(digit_predictions, 1)
                correct_val += (predicted_digits == digits).sum().item()
                total_val += digits.size(0)

        val_accuracy = (correct_val / total_val) * 100
        val_losses.append(val_loss / len(val_loader))
        val_acc.append(val_accuracy)

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    print('Finished Training')

    torch.save(model.state_dict(), 'final_model.pt')

    model.to(device)
    model.eval()

    with torch.no_grad():
        total = 0
        correct_digit = 0
        angle_sum = 0.0

        for images, angles, digits in test_loader:
            images = images.to(device)
            angles = angles.to(device)
            digits = digits.to(device)

            digit_predictions, angle_predictions = model(images)

            _, predicted_digits = torch.max(digit_predictions, 1)
            correct_digit += (predicted_digits == digits).sum().item()

            angle_mae = torch.abs(angle_predictions.squeeze(-1) - angles).mean()
            angle_sum += angle_mae.item() * images.size(0)
            total += digits.size(0)

        digit_accuracy = (correct_digit / total) * 100
        angle_avergae = angle_sum / total

        print(f"Digit Accuracy: {digit_accuracy:.2f}%, Angle MAE: {angle_avergae:.2f} degrees")

    path = root.joinpath('training.pkl')

    if path.exists():
        path.unlink(missing_ok=True)

    training = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_acc': train_acc,
        'val_acc': val_acc
    }

    with open(path, 'wb') as handle:
        pickle.dump(training, handle)


if __name__ == '__main__':
    main()
