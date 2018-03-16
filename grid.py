import pickle

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def silhouette_metric(estimator, X):
    clusters = estimator.fit_predict(X)
    score = silhouette_score(estimator.steps[0][1].transform(X), clusters)
    return score


with open('Tf-Idf_data.pickle', 'rb') as model_pickle:
    data = pickle.load(model_pickle)

pipe = Pipeline(steps=[
    ('lsa', Pipeline(steps=[
        ('svd', TruncatedSVD()),
        ('norm', Normalizer(copy=False))
    ])),
    ('kmeans', KMeans(precompute_distances=True, n_jobs=-1))
])

parms = {
    'kmeans__n_clusters': range(3, 200),
    'lsa__svd__n_components': range(50, 650, 50)
}

grid = GridSearchCV(pipe, param_grid=parms, scoring=silhouette_metric, verbose=2, n_jobs=-1)
grid.fit(data)

with open('grid.pickle', 'wb') as f:
    pickle.dump(grid, f)
