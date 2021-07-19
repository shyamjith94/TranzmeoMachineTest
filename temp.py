import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from geopy.distance import great_circle
from shapely.geometry import MultiPoint

df = pd.read_csv("data/latitude_longitude_details.csv")
df = df.sort_values(by=['latitude', 'longitude'])
df.plot(kind='scatter', x='longitude', y='latitude', alpha=0.5, linewidth=0)
plt.show()
df_coords = df[['latitude', 'longitude']]
kms_per_radian = 6371.0088
epsilon = 10 / kms_per_radian
start_time = time.time()
db = DBSCAN(eps=epsilon, min_samples=10, algorithm='ball_tree', metric='haversine').fit(np.radians(df_coords))
cluster_labels = db.labels_
unique_labels = set(cluster_labels)
num_clusters = len(set(cluster_labels))
fig, ax = plt.subplots()
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
# for each cluster label and color, plot the cluster's points
for cluster_label, color in zip(unique_labels, colors):

    size = 150
    if cluster_label == -1:  # -1 is noise
        color = 'r'
        size = 30

    # plot the points that match the current cluster label
    # X.iloc[:-1]
    # df.iloc[:, 0]
    x_coords = df_coords.iloc[:, 0]
    y_coords = df_coords.iloc[:, 1]
    ax.scatter(x=x_coords, y=y_coords, c=color, edgecolor='k', s=size, alpha=0.5)

ax.set_title('Number of clusters: {}'.format(num_clusters))
plt.show()

# set eps low (1.5km) so clusters are only formed by very close points
epsilon = 0.1 / kms_per_radian

# set min_samples to 1 so we get no noise - every point will be in a cluster even if it's a cluster of 1
start_time = time.time()
db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(df_coords))
cluster_labels = db.labels_
unique_labels = set(cluster_labels)

# get the number of clusters
num_clusters = len(set(cluster_labels))

# all done, print the outcome
message = 'Clustered {:,} points down to {:,} clusters, for {:.1f}% compression in {:,.2f} seconds'
print(message.format(len(df), num_clusters, 100 * (1 - float(num_clusters) / len(df)), time.time() - start_time))

coefficient = metrics.silhouette_score(df_coords, cluster_labels)
print('Silhouette coefficient: {:0.03f}'.format(metrics.silhouette_score(df_coords, cluster_labels)))

# number of clusters, ignoring noise if present
num_clusters = len(set(cluster_labels))  # - (1 if -1 in labels else 0)
print('Number of clusters: {}'.format(num_clusters))

# create a series to contain the clusters - each element in the series is the points that compose each cluster
clusters = pd.Series([df_coords[cluster_labels == n] for n in range(num_clusters)]).reset_index()
print(clusters.tail())


















# def get_centermost_point(cluster):
#     centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
#     centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
#     return tuple(centermost_point)
#
#
# def dbscan_reduce(df, epsilon, x='longitude', y='latitude'):
#     start_time = time.time()
#     # represent points consistently as (lat, lon) and convert to radians to fit using haversine metric
#     coords = df.values
#     db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
#     cluster_labels = db.labels_
#     num_clusters = len(set(cluster_labels))
#     print('Number of clusters: {:,}'.format(num_clusters))
#
#     clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
#
#     # find the point in each cluster that is closest to its centroid
#     centermost_points = clusters.map(get_centermost_point)
#
#     # unzip the list of centermost points (lat, lon) tuples into separate lat and lon lists
#     lats, lons = zip(*centermost_points)
#     rep_points = pd.DataFrame({x: lons, y: lats})
#     rep_points.tail()
#
#     # pull row from original data set where lat/lon match the lat/lon of each row of representative points
#     rs = rep_points.apply(lambda row: df[(df[y] == row[y]) & (df[x] == row[x])].iloc[0], axis=1)
#
#     # all done, print outcome
#     message = 'Clustered {:,} points down to {:,} points, for {:.2f}% compression in {:,.2f} seconds.'
#     print(message.format(len(df), len(rs), 100 * (1 - float(len(rs)) / len(df)), time.time() - start_time))
#     return rs
#
#
# data = pd.read_csv("data/latitude_longitude_details.csv")
# kms_per_radian = 6371.0088
# eps_rad = 5 / kms_per_radian
# df_clustered = dbscan_reduce(data, epsilon=eps_rad)
#
# sample_rate = 100
# df_sampled = data.iloc[range(0, len(data), sample_rate)]
# print(len(df_sampled))
# df_combined = pd.concat([df_clustered, df_sampled], axis=0)
# df_combined = df_combined.reset_index().drop(labels='index', axis=1)
# eps_rad = 3 / kms_per_radian
# df_final = dbscan_reduce(df_combined, epsilon=eps_rad)
# print(df_final)


# def haversine(lonlat1, lonlat2):
#     """
#     Calculate the great circle distance between two points
#     on the earth (specified in decimal degrees)
#     """
#     # convert decimal degrees to radians
#     lat1, lon1 = lonlat1
#     lat2, lon2 = lonlat2
#     lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
#
#     # haversine formula
#     dlon = lon2 - lon1
#     dlat = lat2 - lat1
#     a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
#     c = 2 * asin(sqrt(a))
#     r = 6371  # Radius of earth in kilometers. Use 3956 for miles
#     return c * r
#
# data = pd.read_csv("data/latitude_longitude_details.csv")
# coords = squareform(pdist(data, (lambda u, v: haversine(u, v))))
# kms_per_radian = 6371.0088
# epsilon = 3 / kms_per_radian
#
# db = DBSCAN(eps=3, min_samples=5, algorithm='auto', metric='precomputed')
# y_db = db.fit_predict(coords)
# data['cluster'] = y_db
#
# plt.scatter(data['latitude'], data['longitude'], c=data['cluster'])
# plt.show()


# centermost_points = clusters.map(get_centermost_point)

# unzip the list of centermost points (lat, lon) tuples into separate lat and lon lists
# lats, lons = zip(*centermost_points)
# rep_points = pd.DataFrame({'x': lons, 'y': lats})
# rs = rep_points.apply(lambda row: df[(df[y]==row[y]) & (df[x]==row[x])].iloc[0], axis=1)
# print(clusters)
# print(len(clusters))
# print(len(clusters))

# kms_per_radian = 6371.0088
# epsilon = 3 / kms_per_radian
# db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
# cluster_labels = db.labels_
# num_clusters = len(set(cluster_labels))
# clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
# print('Number of clusters: {}'.format(num_clusters))
#
#
# def get_centermost_point(cluster):
#     centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
#     centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
#     return tuple(centermost_point)
#
#
# centermost_points = clusters.map(get_centermost_point)
# lats, lons = zip(*centermost_points)
# rep_points = pd.DataFrame({'longitude': lons, 'latitude': lats})
# print(rep_points)
# rs = rep_points.apply(
#     lambda row: data[(data['latitude'] == row['latitude']) & (data['longitude'] == row['longitude'])].iloc[0], axis=1)
# print(rs)
# fig, ax = plt.subplots(figsize=[10, 6])
# rs_scatter = ax.scatter(rs['longitude'], rs['latitude'], c='#99cc99', edgecolor='None', alpha=0.7, s=120)
# df_scatter = ax.scatter(data['longitude'], data['latitude'], c='k', alpha=0.9, s=3)
# ax.set_title('Full data set vs DBSCAN reduced set')
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')
# ax.legend([df_scatter, rs_scatter], ['Full set', 'Reduced set'], loc='upper right')
# plt.show()


#
# def distance(origin, destination):  # found here https://gist.github.com/rochacbruno/2883505
#     lat1, lon1 = origin[0], origin[1]
#     lat2, lon2 = destination[0], destination[1]
#     radius = 6371  # km
#     dlat = math.radians(lat2 - lat1)
#     dlon = math.radians(lon2 - lon1)
#     a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) \
#         * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
#     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
#     d = radius * c
#     return d
#
#
# def create_clusters(number_of_clusters, points):
#     kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(points)
#     l_array = np.array([[label] for label in kmeans.labels_])
#     clusters = np.append(points, l_array, axis=1)
#     return clusters
#
#
# def validate_cluster(max_dist, cluster):
#     distances = cdist(cluster, cluster, lambda ori, des: int(round(distance(ori, des))))
#     print("distances", distances)
#     for item in distances.flatten():
#         if item > max_dist:
#             return False
#     return True
#
#
# def validate_solution(max_dist, clusters):
#     _, __, n_clust = clusters.max(axis=0)
#     n_clust = int(n_clust)
#     for i in range(n_clust):
#         two_d_cluster = clusters[clusters[:, 2] == i][:, np.array([True, True, False])]
#         print("two_d_cluster", two_d_cluster)
#         if not validate_cluster(max_dist, two_d_cluster):
#             return False
#         else:
#             continue
#     return True
#
#
# for i in range(2, len(points)):
#     print(i)
#     print(validate_solution(5, create_clusters(i, points)))
#     # print(create_clusters(i, points))
