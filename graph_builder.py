from matplotlib.figure import Figure
import numpy as np
from scipy.stats import gaussian_kde


def donut_chart(df):
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.set_facecolor('none')
    ax.text(0, 0, 'Статистика ОП', ha='center', va='center')
    forms = df['Форма обучения'].explode()
    counts = forms.value_counts()
    ax.pie(counts.values, radius=1, wedgeprops={'width': 0.2})
    qc = df['Квалификация'].value_counts()
    if len(qc) > 1:
        ax.pie(qc.values, radius=0.8, wedgeprops={'width': 0.2})
    return fig


def price_kde(df):
    fig = Figure(figsize=(8, 6))
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111)
    ax.set_facecolor('none')

    prices = df['Стоимость (в год)'].dropna()
    prices = prices[prices <= 1_000_000]

    ls = np.linspace(min(prices), max(prices), 200)
    kde = gaussian_kde(prices)
    ax.plot(ls, kde(ls), color='black', linewidth=2)

    ax.set_title('Распределение годовой стоимости (очное обучение)\n', fontsize=14)
    ax.set_xlabel('Стоимость, руб')
    ax.set_ylabel('Плотность')
    ax.ticklabel_format(style='plain', axis='x')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig


def points_kde(df):
    fig = Figure(figsize=(8, 6))
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111)
    ax.set_facecolor('none')

    points = df.apply(
        lambda row: row['Проходной балл на бюджет'] / row['Кол-во экзаменов']
        if row['Проходной балл на бюджет'] is not None and row['Кол-во экзаменов'] > 0
        else None,
        axis=1
    ).dropna()

    ls = np.linspace(min(points), max(points), 200)
    kde = gaussian_kde(points)
    ax.plot(ls, kde(ls), color='white', linewidth=2)

    ax.set_title('Распределение проходных баллов ЕГЭ\n', fontsize=14, color='white')
    ax.set_xlabel('Баллы', color='white')
    ax.set_ylabel('Плотность', color='white')
    ax.ticklabel_format(style='plain', axis='x')
    ax.tick_params(axis='both', colors='white')
    ax.grid(True, linestyle='--', alpha=0.5, color='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')

    return fig
