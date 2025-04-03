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
    target_key = "Очная, на русском, Полный курс, на базе 11 классов"

    fig = Figure(figsize=(8, 6))
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111)
    ax.set_facecolor('none')

    def get_prices(x):
        val = x.get(target_key, {}).get('Платное', {})
        if isinstance(val, dict):
            return val.get('Стоимость (в год)')
        return None

    prices = df['Варианты поступления'].dropna().apply(get_prices).dropna()
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
    target_key = "Очная, на русском, Полный курс, на базе 11 классов"

    fig = Figure(figsize=(8, 6))
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111)
    ax.set_facecolor('none')

    def get_points(x):
        ege = x.get('ЕГЭ', [])
        vp = x.get('Варианты поступления', {})
        if not isinstance(vp, dict):
            return None
        val = vp.get(target_key, {}).get('Бюджет')
        if isinstance(val, dict):
            score = val.get('Проходной балл')
            if score and 0 < len(ege) < 6:
                return score / len(ege)
        return None

    points = df.dropna(subset=['Варианты поступления']).apply(get_points, axis=1).dropna()

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
