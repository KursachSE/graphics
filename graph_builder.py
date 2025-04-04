from matplotlib.figure import Figure
import numpy as np
from scipy.stats import gaussian_kde


class GraphBuilder:
    GREEN = '#53b93f'
    FZ = 12
    TITLE_FZ = 14

    @staticmethod
    def get_transparent_fig(a, b):
        fig = Figure(figsize=(a, b))
        fig.patch.set_alpha(0)
        ax = fig.add_subplot(111)
        ax.set_facecolor('none')
        return fig, ax

    @staticmethod
    def set_scatter_signs(ax, title, x_label, y_label, color='black'):
        ax.set_title(title, fontsize=GraphBuilder.TITLE_FZ, color=color)
        ax.set_xlabel(x_label, fontsize=GraphBuilder.FZ, color=color)
        ax.set_ylabel(y_label, fontsize=GraphBuilder.FZ, color=color)
        ax.ticklabel_format(style='plain', axis='x')
        ax.tick_params(axis='both', colors=color)
        ax.grid(True, linestyle='--', alpha=0.5, color=color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(color)
        ax.spines['left'].set_color(color)

    @staticmethod
    def donut_chart(df):
        fig, ax = GraphBuilder.get_transparent_fig(6, 6)
        ax.text(0, 0, 'Статистика ОП', ha='center', va='center', fontsize=GraphBuilder.TITLE_FZ)
        forms = df['Форма обучения'].explode()
        counts = forms.value_counts()
        ax.pie(counts.values, radius=1, wedgeprops={'width': 0.2, 'edgecolor': (1, 1, 1, 0.3)})
        qc = df['Квалификация'].value_counts()
        if len(qc) > 1:
            ax.pie(qc.values, radius=0.8, wedgeprops={'width': 0.2, 'edgecolor': (1, 1, 1, 0.3)})
        return fig

    @staticmethod
    def build_kde_layout(var, title, x_label, color='black'):
        fig, ax = GraphBuilder.get_transparent_fig(8, 6)
        GraphBuilder.set_scatter_signs(ax, title, x_label, 'Плотность', color)
        ls = np.linspace(min(var), max(var), 200)
        kde = gaussian_kde(var)
        ax.plot(ls, kde(ls), color=color, linewidth=2)
        return fig

    @staticmethod
    def price_kde(df):
        title = 'Распределение годовой стоимости (очное обучение)\n'
        prices = df['Стоимость (в год)'].dropna()
        prices = prices[prices <= 1_000_000] / 1000
        return GraphBuilder.build_kde_layout(prices, title, 'Стоимость, тыс. руб')

    @staticmethod
    def points_kde(df):
        title = 'Распределение проходных баллов ЕГЭ\n'
        points = df.apply(
            lambda row: row['Проходной балл на бюджет'] / row['Кол-во экзаменов']
            if row['Проходной балл на бюджет'] is not None and row['Кол-во экзаменов'] > 0 else None,
            axis=1
        ).dropna()
        return GraphBuilder.build_kde_layout(points, title, 'Баллы', 'white')

    @staticmethod
    def price_to_points_scatter(df):
        title = 'Зависимость стоимости и проходного балла лучших вузов\n'
        fig, ax = GraphBuilder.get_transparent_fig(9, 6)
        GraphBuilder.set_scatter_signs(ax, title, 'Стоимость, тыс. руб', 'Проходной балл на бюджет')
        filtered_df = df[df['Место в топе'] < 13].copy()
        filtered_df['Кол-во экзаменов'] = filtered_df['Кол-во экзаменов'].replace(0, np.nan)
        filtered_df['Проходной балл на бюджет'] = filtered_df['Проходной балл на бюджет'] / filtered_df['Кол-во экзаменов']
        grouped = filtered_df.groupby('Университет').agg({'Стоимость (в год)': 'mean', 'Проходной балл на бюджет': 'mean'})
        grouped['Стоимость (в год)'] = grouped['Стоимость (в год)'] / 1000
        ax.scatter(grouped['Стоимость (в год)'], grouped['Проходной балл на бюджет'], color=GraphBuilder.GREEN, s=70)
        for uni, row in grouped.iterrows():
            ax.text(
                row['Стоимость (в год)'] + 10,
                row['Проходной балл на бюджет'],
                uni,
                fontsize=10,
                fontweight='bold',
                ha='left',
                va='center',
                color=GraphBuilder.GREEN
            )
        return fig
