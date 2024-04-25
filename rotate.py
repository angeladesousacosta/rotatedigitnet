import pandas as pd
import random

from pathlib import Path
from PIL import Image


def rotate_images(source, target):
    images = []

    files = source.rglob('*.png')

    for file in files:
        metadata = {}

        img = Image.open(file)

        angle = random.uniform(-45, 45)
        rotated = img.rotate(-angle)

        relative = file.relative_to(source)

        metadata['path'] = relative
        metadata['angle'] = angle
        metadata['digit'] = relative.parent

        target_path = target / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        rotated.save(target_path)

        images.append(metadata)

    return images


def main():
    mnist_path = Path('mnist_dataset')

    dataset_path = Path('dataset')
    dataset_path.mkdir(parents=True, exist_ok=True)

    for folder in ['training', 'validation', 'testing']:
        source_path = mnist_path / folder
        target_path = dataset_path / folder
        metadata = rotate_images(source_path, target_path)

        path = dataset_path.joinpath(folder)
        path.mkdir(parents=True, exist_ok=True)

        path = dataset_path.joinpath(folder + '.csv')

        dataframe = pd.DataFrame(metadata)
        dataframe.to_csv(path, index=False)

    print('Dataset processing complete.')


if __name__ == '__main__':
    main()
