import matplotlib.pyplot as plt
import torch

from dataset import RotatedDataset
from model import RotateDigitNet
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms


def predictions(loader, model, device, samples):
    model.eval()
    visualize = min(samples, len(loader.dataset))

    fig, axis = plt.subplots(
        nrows=visualize,
        ncols=2,
        figsize=(10, 2 * visualize)
    )

    for index, (images, angles, digits) in enumerate(loader):
        if index >= visualize:
            break

        images = images.to(device)
        angles = angles.to(device)
        digits = digits.to(device)

        with torch.no_grad():
            digit_predictions, angle_predictions = model(images)
            _, predicted_digits = torch.max(digit_predictions, 1)

        image = images[0].cpu().squeeze().numpy()
        real_angle = angles.item()
        predicted_angle = angle_predictions.squeeze().item()

        image_axis = axis[index, 0]
        image_axis.imshow(image, cmap='gray')
        image_axis.set_title(f"Real Digit: {digits.item()}, Predicted: {predicted_digits.item()}")
        image_axis.axis('off')

        text_axis = axis[index, 1]
        angle_info = f"Real Angle: {real_angle:.2f}°\nPredicted: {predicted_angle:.2f}°"
        text_axis.text(0.5, 0.5, angle_info, ha='center', va='center', fontsize=12, transform=text_axis.transAxes)
        text_axis.axis('off')

    plt.tight_layout()
    plt.savefig('results.png', dpi=300, format='png')
    plt.show()
    plt.close()


def main():
    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    model = RotateDigitNet()
    model.load_state_dict(torch.load('final_model.pt'))
    model = model.to(device)

    root = Path.cwd()
    current = root.joinpath('dataset')

    transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,)),
    ])

    test_dataset = RotatedDataset(
        csv_file=current.joinpath('testing.csv'),
        root_path=current.joinpath('testing'),
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True
    )

    predictions(test_loader, model, device, samples=5)


if __name__ == '__main__':
    main()
