import numpy as np
from scipy.stats import gaussian_kde
from base_plot_config import BasePlotConfig, PRICE, BUDGET_PTS


class GraphBuilder(BasePlotConfig):
    def donut_chart(self):
        fig, ax = BasePlotConfig._get_transparent_fig(6, 6)
        ax.text(0, 0, 'Статистика ОП', ha='center', va='center', fontsize=self.TITLE_FZ)

        forms = self.df['Форма обучения'].explode()
        counts = forms.value_counts()
        props = {'width': 0.2, 'edgecolor': (1, 1, 1, 0.3)}
        ax.pie(counts.values, radius=1, wedgeprops=props)

        qc = self.df['Квалификация'].value_counts()
        if len(qc) > 1:
            ax.pie(qc.values, radius=0.8, wedgeprops=props)
        return fig

    def _build_kde_layout(self, var, title, x_label, color='black'):
        fig, ax = BasePlotConfig._get_transparent_fig(8, 6)
        self._set_scatter_signs(ax, title, x_label, 'Плотность', color)

        ls = np.linspace(min(var), max(var), 1000)
        kde = gaussian_kde(var)
        ax.plot(ls, kde(ls), color=color, linewidth=2)
        return fig

    def price_kde(self):
        title = 'Распределение годовой стоимости (очное обучение)\n'
        prices = self.df[PRICE][self.df[PRICE] <= 1200]
        return self._build_kde_layout(prices, title, 'Стоимость, тыс. руб')

    def points_kde(self):
        title = 'Распределение проходных баллов ЕГЭ\n'
        points = self.df[BUDGET_PTS].dropna()
        return self._build_kde_layout(points, title, 'Баллы', 'white')

    def price_to_points_scatter(self):
        title = 'Зависимость стоимости и проходного балла лучших вузов\n'
        fig, ax = BasePlotConfig._get_transparent_fig(9, 6)
        self._set_scatter_signs(ax, title, 'Стоимость, тыс. руб', BUDGET_PTS)

        df = self.df[self.df['Место в топе'] < 13]
        grouped = df.groupby('Университет').agg({PRICE: 'mean', BUDGET_PTS: 'mean'})

        ax.scatter(grouped[PRICE], grouped[BUDGET_PTS], color=self.MAIN_COLOR, s=70)
        for uni, row in grouped.iterrows():
            ax.text(
                row[PRICE] + 10, row[BUDGET_PTS], uni, fontsize=10, fontweight='bold',
                ha='left', va='center', color=self.MAIN_COLOR
            )
        return fig
