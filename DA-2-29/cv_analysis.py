"""
Анализ коэффициента вариации (CV = std / mean) для датасета diabetes из sklearn.

Функционал:
1. Загружает реальные данные (sklearn.datasets.load_diabetes()).
2. Считает CV для каждого числового признака.
3. Обрабатывает случай, когда данные стандартизированы (mean ≈ 0).
4. Автоматически создаёт "реалистичную" симулированную версию данных.
5. Строит график коэффициентов вариации и сохраняет его.
6. Выводит топ-3 самых изменчивых признака.
7. Пишет интерпретацию.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes


# ============================================================
#                    ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def compute_cv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет коэффициент вариации (CV = std / mean) для всех числовых признаков.

    Parameters
    ----------
    df : pd.DataFrame
        Таблица с числовыми признаками.

    Returns
    -------
    pd.DataFrame
        Таблица с именами признаков и их CV.
    """
    cv_values = {}
    for col in df.columns:
        mean = df[col].mean()
        std = df[col].std()
        cv = np.nan if np.isclose(mean, 0) else std / mean
        cv_values[col] = cv
    return pd.DataFrame(list(cv_values.items()), columns=["feature", "CV"]).sort_values(by="CV", ascending=False)


def simulate_realistic_data(original: pd.DataFrame) -> pd.DataFrame:
    """
    Создает симулированные данные на основе реальных диапазонов признаков.
    Это нужно, т.к. diabetes dataset стандартизирован (mean≈0).

    эти значения НЕ соответствуют реальным медицинским показателям, 
    так как оригинальный датасет diabetes анонимизирован и стандартизирован.

    Parameters
    ----------
    original : pd.DataFrame
        Оригинальный DataFrame из sklearn.

    Returns
    -------
    pd.DataFrame
        Симулированные данные с реалистичными диапазонами.
    """
    np.random.seed(42)
    n = len(original)
    simulated = pd.DataFrame({
        "age": np.random.normal(50, 10, n),
        "sex": np.random.choice([0, 1], n),
        "bmi": np.random.normal(27, 4, n),
        "bp": np.random.normal(80, 10, n),
        "s1": np.random.normal(150, 30, n),
        "s2": np.random.normal(130, 25, n),
        "s3": np.random.normal(90, 15, n),
        "s4": np.random.normal(4.5, 1.0, n),
        "s5": np.random.normal(5.0, 0.7, n),
        "s6": np.random.normal(90, 20, n)
    })
    return simulated


def plot_cv(cv_df: pd.DataFrame, filename: str = "cv_visualization.png"):
    """
    Визуализирует коэффициенты вариации.

    Parameters
    ----------
    cv_df : pd.DataFrame
        Таблица с признаками и CV.
    filename : str
        Имя файла для сохранения графика.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(cv_df["feature"], cv_df["CV"], color="skyblue")
    plt.title("Коэффициенты вариации признаков")
    plt.xlabel("Признак")
    plt.ylabel("CV (std/mean)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# ============================================================
#                    ОСНОВНАЯ ЛОГИКА ПРОГРАММЫ
# ============================================================

def main():
    """
    Основная функция: выполняет анализ CV для diabetes dataset.
    """
    try:
        # 1. Загружаем данные
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)

        print("Анализ реальных данных (sklearn.datasets.load_diabetes())")
        cv_real = compute_cv(df)
        print("\nКоэффициенты вариации:\n", cv_real.to_string(index=False))

        # Проверяем, есть ли осмысленные значения
        if cv_real["CV"].isna().all():
            print("\nВсе признаки стандартизированы, CV не имеет смысла (mean ≈ 0).")

            # 2. Генерируем реалистичные данные
            print("\nГенерация симулированных реалистичных данных...")
            simulated_df = simulate_realistic_data(df)

            # 3. Пересчитываем CV для симулированных данных
            cv_sim = compute_cv(simulated_df)

            # 4. Визуализация
            plot_cv(cv_sim)
            print("\nГрафик сохранён в: cv_visualization.png")

            # 5. Вывод топ-3
            top3 = cv_sim.head(3)
            print("\nТоп-3 признаков по изменчивости:\n", top3.to_string(index=False))
        else:
            plot_cv(cv_real)
            print("\nГрафик сохранён в: cv_visualization.png")

            top3 = cv_real.head(3)
            print("\nТоп-3 признаков по изменчивости:\n", top3.to_string(index=False))

        # 6. Интерпретация
        print("""
Интерпретация:
Коэффициент вариации (CV) показывает относительный разброс признака относительно его среднего.
• Чем выше CV — тем более изменчив признак (нестабильный, разнообразный).
• Низкий CV — признак более стабильный и однородный.
• Если mean ≈ 0 (стандартизированные данные), CV неинформативен.
""")

    except Exception as e:
        print(f"Ошибка при выполнении: {e}")


# ============================================================
#                           ЗАПУСК
# ============================================================

if __name__ == "__main__":
    main()
