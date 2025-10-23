import cv2
import numpy as np
import os
import time


# =====================================================================
# Функции, адаптированные из задачи CV-2-41 (фильтрация шума в изображении)
# =====================================================================

def estimate_noise(img):
    """
    Оценка уровня шума через дисперсию лапласиана.
    Взято из CV-2-41 (функция estimate_noise).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()


def adaptive_denoise(img, method="auto", ksize=5, sigma=75):
    """
    Адаптивная фильтрация изображения на основе уровня шума.
    Основано на функции adaptive_denoise из CV-2-41,
    дополнено параметрами для ручной настройки фильтров через trackbar.
    """
    noise_level = estimate_noise(img)

    if method == "auto":
        # Адаптивный выбор фильтра в зависимости от уровня шума
        if noise_level < 100:
            return cv2.GaussianBlur(img, (3, 3), 0.5), "GaussianBlur (3x3)"
        elif noise_level < 500:
            return cv2.medianBlur(img, 5), "MedianBlur (5)"
        else:
            return cv2.bilateralFilter(img, 9, 75, 75), "Bilateral (9,75,75)"
    elif method == "gaussian":
        return cv2.GaussianBlur(img, (ksize, ksize), 0), f"Gaussian ({ksize}x{ksize})"
    elif method == "median":
        return cv2.medianBlur(img, ksize), f"Median ({ksize})"
    elif method == "bilateral":
        return cv2.bilateralFilter(img, 9, sigma, sigma), f"Bilateral ({sigma})"
    else:
        raise ValueError("Некорректный метод фильтрации")


# =====================================================================
# 🔹 Функции для взаимодействия с пользователем (trackbar)
# =====================================================================

def setup_trackbar(window_name):
    """
    Создаёт trackbar для настройки параметров фильтрации.
    Новый код, добавленный для интерактивного управления.
    """
    cv2.createTrackbar("Filter", window_name, 0, 3, lambda x: None)
    cv2.createTrackbar("Kernel", window_name, 3, 15, lambda x: None)
    cv2.createTrackbar("Sigma", window_name, 75, 150, lambda x: None)
    cv2.createTrackbar("Save", window_name, 0, 1, lambda x: None)


def get_trackbar_values(window_name):
    """
    Возвращает текущие значения с trackbar.
    Новый код — используется для динамической настройки фильтра.
    """
    filter_id = cv2.getTrackbarPos("Filter", window_name)
    ksize = cv2.getTrackbarPos("Kernel", window_name)
    sigma = cv2.getTrackbarPos("Sigma", window_name)
    save_flag = cv2.getTrackbarPos("Save", window_name)

    filters = {0: "auto", 1: "gaussian", 2: "median", 3: "bilateral"}
    method = filters.get(filter_id, "auto")

    # kernel size должен быть нечётным
    if ksize % 2 == 0:
        ksize += 1

    return method, ksize, sigma, save_flag


# =====================================================================
# Запись видеофайла
# =====================================================================

def initialize_video_output(filename="denoised_output.avi", fps=30, frame_size=(640, 480)):
    """
    Подготавливает объект VideoWriter для записи результата в файл.
    Новый код (добавлен для реализации пункта "Сохранить результат как видео файл").
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter(filename, fourcc, fps, (frame_size[0] * 2, frame_size[1]))


# =====================================================================
# Главный цикл — адаптировано из кода CV-1-10 (работа с веб-камерой)
# =====================================================================

def main():
    """
    Захват видео с веб-камеры и адаптивная фильтрация в реальном времени.
    Основной цикл основан на коде CV-1-10, но дополнен:
      - адаптивной фильтрацией (из CV-2-41)
      - визуализацией двух видео (исходное + обработанное)
      - настройкой параметров через trackbar
      - сохранением результата
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру.")
        return

    window_name = "Noise Filtering (press 'q' to quit)"
    cv2.namedWindow(window_name)
    setup_trackbar(window_name)

    prev_time = time.time()
    writer = None
    print("Запуск. Нажмите 'q' для выхода.")

    while True:
        # Захват кадра с камеры
        ret, frame = cap.read()
        if not ret:
            print("⚠ Не удалось считать кадр с камеры.")
            break

        frame = cv2.resize(frame, (640, 480))

        # Получение параметров от пользователя
        method, ksize, sigma, save_flag = get_trackbar_values(window_name)

        # Фильтрация кадра (взято из CV-2-41, адаптировано под видео)
        denoised, method_name = adaptive_denoise(frame, method, ksize, sigma)

        # Расчёт FPS (взято из CV-1-10)
        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now

        # Визуализация исходного и обработанного кадров
        combined = np.hstack((frame, denoised))
        cv2.putText(combined, f"FPS: {fps:.1f} | {method_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow(window_name, combined)

        # Управление записью видео
        if save_flag and writer is None:
            os.makedirs("results", exist_ok=True)
            writer = initialize_video_output("results/denoised_output.avi", 30, (640, 480))
            print("💾 Запись видео включена.")
        elif not save_flag and writer is not None:
            writer.release()
            writer = None
            print("Запись видео остановлена.")

        # Запись обработанного видео
        if writer is not None:
            writer.write(combined)

        # Выход (из CV-1-10)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Очистка ресурсов
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("Работа завершена.")


# =====================================================================
# 🔹 Точка входа
# =====================================================================
if __name__ == "__main__":
    main()
