import numpy as np
from base_plot_config import BasePlotConfig
from sklearn.cluster import KMeans, MeanShift, DBSCAN, estimate_bandwidth
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score


class ClusterAnalyzer(BasePlotConfig):
    SEED = 234

    def __init__(self, df, x_label, y_label):
        super().__init__(df)
        self.x_label = x_label
        self.y_label = y_label

    def show_clusters(self):
        title = f'Кластеризация методом {self.method.__name__}'
        fig, ax = BasePlotConfig._get_transparent_fig(8, 6)
        self._set_scatter_signs(ax, title, self.x_label, self.y_label)
        ax.scatter(self.X[:, 0], self.X[:, 1], c=self.clusters, cmap='viridis', alpha=0.7, edgecolor='black')
        return fig

    def _cluster_data(self, method, params=None):
        mask = self.df[[self.x_label, self.y_label]].dropna()
        cluster = method(**params)
        clusters = cluster.fit_predict(mask.values)

        self.X = mask.values
        self.clusters = clusters
        self.method = method

        self.df.loc[mask.index, 'Кластер'] = clusters
        cluster_groups = self.df.loc[mask.index].groupby('Кластер')['Ссылка'].agg(list).to_dict()
        return cluster_groups

    def kmeans_cluster_data(self, n_clusters=None):
        if n_clusters is None:
            n_clusters = self._auto_kmeans_clusters()
        params = {'n_clusters': n_clusters, 'random_state': self.SEED}
        return self._cluster_data(KMeans, params)

    def mean_shift_cluster_data(self, bandwidth=None):
        if bandwidth is None:
            data = self.df[[self.x_label, self.y_label]].dropna().values
            bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=len(data))
        return self._cluster_data(MeanShift, {'bandwidth': bandwidth})

    def dbscan_cluster_data(self, eps=None, min_samples=None):
        if eps is None or min_samples is None:
            eps, min_samples = self._estimate_dbscan_params()
        params = {'eps': eps, 'min_samples': min_samples}
        return self._cluster_data(DBSCAN, params)

    def _auto_kmeans_clusters(self):
        k_min, k_max = 2, 6
        data = self.df[[self.x_label, self.y_label]].dropna().values
        best_k = k_min
        best_score = -1

        for k in range(k_min, k_max + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.SEED)
            labels = kmeans.fit_predict(data)
            score = silhouette_score(data, labels)
            if score > best_score:
                best_score = score
                best_k = k

        return best_k

    def _elbow_plot(self, max_k=10):
        X = self.df[[self.x_label, self.y_label]].dropna().values
        inertias = []
        Ks = list(range(1, max_k + 1))
        for k in Ks:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)

        fig, ax = BasePlotConfig._get_transparent_fig(8, 6)
        self._set_scatter_signs(ax, 'Метод локтя', 'Количество кластеров', 'Inertia')
        ax.plot(Ks, inertias, marker='o', color=self.MAIN_COLOR, linewidth=2)
        return fig

    def _dendrogram(self):
        data = self.df[[self.x_label, self.y_label]].dropna()
        Z = linkage(data.values, method='ward')
        fig, ax = BasePlotConfig._get_transparent_fig(10, 8)
        self._set_scatter_signs(ax, 'Дендограмма', 'Объекты', 'Расстояние')
        dendrogram(Z, ax=ax, labels=data.index.astype(str).tolist(), leaf_rotation=90)
        return fig

    def _estimate_dbscan_params(self, k=None):
        data = self.df[[self.x_label, self.y_label]].dropna().drop_duplicates().values
        if k is None:
            k = data.shape[1] + 1

        nbrs = NearestNeighbors(n_neighbors=k)
        nbrs.fit(data)
        distances, _ = nbrs.kneighbors(data)
        k_distances = np.sort(distances[:, k - 1])
        eps_opt = np.median(k_distances)

        return eps_opt, k