import numpy as np
from matplotlib.figure import Figure


PRICE = 'Стоимость (в год)'
BUDGET_PTS = 'Проходной балл на бюджет'


class BasePlotConfig:
    def __init__(self, df, main_color='#53b93f', title_fz=14, font_size=12):
        self.df = BasePlotConfig._edit_columns(df)
        self.MAIN_COLOR = main_color
        self.TITLE_FZ = title_fz
        self.FZ = font_size

    @staticmethod
    def _edit_columns(df):
        df = df.copy()
        exams_num = 'Кол-во экзаменов'
        df[PRICE] /= 1000
        df[BUDGET_PTS] = df[BUDGET_PTS] / df[exams_num]
        df[BUDGET_PTS] = df[BUDGET_PTS].mask(df[BUDGET_PTS] > 103)
        df['Срок обучения'] = df['Срок обучения'].apply(
            lambda x: np.mean(x) if isinstance(x, list) and len(x) > 0 else np.nan
        )
        return df

    @staticmethod
    def _get_transparent_fig(a, b):
        fig = Figure(figsize=(a, b))
        fig.patch.set_alpha(0)
        ax = fig.add_subplot(111)
        ax.set_facecolor('none')
        return fig, ax

    def _set_scatter_signs(self, ax, title, x_label, y_label, color='black'):
        ax.set_title(title, fontsize=self.TITLE_FZ, color=color)
        ax.set_xlabel(x_label, fontsize=self.FZ, color=color)
        ax.set_ylabel(y_label, fontsize=self.FZ, color=color)
        ax.ticklabel_format(style='plain', axis='x')
        ax.tick_params(axis='both', colors=color)
        ax.grid(True, linestyle='--', alpha=0.5, color=color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(color)
        ax.spines['left'].set_color(color)
