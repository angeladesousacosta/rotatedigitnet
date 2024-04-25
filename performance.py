import matplotlib.pyplot as plt
import pickle
import scienceplots

from pathlib import Path


def main():
    root = Path.cwd()
    path = root.joinpath('training.pkl')

    with open(path, 'rb') as handle:
        training = pickle.load(handle)

    train_losses = training.get('train_losses')
    val_losses = training.get('val_losses')
    train_acc = training.get('train_acc')
    val_acc = training.get('val_acc')

    plt.style.use('science')
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss.png', dpi=300, format='png')
    plt.show()

    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy.png', dpi=300, format='png')
    plt.show()

    plt.close()


if __name__ == '__main__':
    main()
