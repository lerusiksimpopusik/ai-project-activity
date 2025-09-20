"""
Модуль для построения тепловой карты распределения яркости изображения.

Функционал:
1. Загрузка изображения (встроенное или пользовательское).
2. Конвертация в градации серого.
3. Нормализация значений яркости к диапазону [0, 1].
4. Построение и сохранение тепловой карты с помощью matplotlib.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, data, io
from typing import Union, Optional


def load_image(image: Optional[Union[str, np.ndarray]] = None) -> np.ndarray:
    """Загрузка изображения (из файла или встроенное)."""
    if image is None:
        return data.camera()
    if isinstance(image, str):
        try:
            return io.imread(image)
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл '{image}' не найден.")
    if isinstance(image, np.ndarray):
        return image
    raise TypeError("Аргумент 'image' должен быть str, np.ndarray или None.")


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Преобразование в grayscale и нормализация к [0, 1]."""
    if img.ndim == 3:
        if img.shape[2] == 4:
            # Убираем альфа-канал
            img = img[:, :, :3]
        img_gray = color.rgb2gray(img)
    elif img.ndim == 2:
        img_gray = img
    else:
        raise ValueError("Некорректный формат изображения. Ожидался 2D или 3D массив.")

    gray_norm = (img_gray - np.min(img_gray)) / (np.max(img_gray) - np.min(img_gray))
    return gray_norm.astype(np.float32)


def plot_heatmap(gray_img: np.ndarray,
                 title: str = "Тепловая карта яркости",
                 show_axes: bool = True,
                 save_path: Optional[str] = None) -> None:
    """Отображение или сохранение тепловой карты."""
    plt.figure(figsize=(6, 6))
    heatmap = plt.imshow(gray_img, cmap='hot', origin='upper')
    plt.colorbar(heatmap, fraction=0.046, pad=0.04, label="Яркость (0–1)")
    plt.title(title)

    if not show_axes:
        plt.axis("off")
    else:
        plt.xlabel("X (пиксели)")
        plt.ylabel("Y (пиксели)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Тепловая карта сохранена в файл: {save_path}")
    else:
        plt.show()


def main(image: Optional[Union[str, np.ndarray]] = None,
         save_path: Optional[str] = "heatmap.png") -> None:
    """Основная функция построения тепловой карты."""
    img = load_image(image)
    gray_norm = preprocess_image(img)
    plot_heatmap(gray_norm, save_path=save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Построение тепловой карты яркости изображения")
    parser.add_argument("--image", type=str, help="Путь к изображению (по умолчанию встроенное)", default=None)
    parser.add_argument("--save", type=str, help="Файл для сохранения результата (PNG)", default="heatmap.png")
    parser.add_argument("--no-save", action="store_true", help="Не сохранять, а только показать")
    args = parser.parse_args()

    if args.no_save:
        main(image=args.image, save_path=None)
    else:
        main(image=args.image, save_path=args.save)
