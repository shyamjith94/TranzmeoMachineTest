import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from connection import DbConnection
import json


def cluster_distance_based(matrix) -> None:
    """Perform DBSCAN clustering from vector array or distance matrix.

    eps The maximum distance between two samples

    :parameter
    :param matrix: matrix
    """
    db = DBSCAN(eps=0.5, min_samples=5, metric='precomputed')
    y_db = db.fit_predict(matrix)
    df['cluster'] = y_db
    plt.scatter(df['latitude'], df['longitude'], c=df['cluster'])
    plt.show()


def haversine(lonlat1, lonlat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    Radius of earth in kilometers. Use 3956 for miles
    """
    # convert decimal degrees to radians
    lat1, lon1 = lonlat1
    lat2, lon2 = lonlat2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return round(c * r, 2)


def distance_matrix_value(distance_matrix) -> dict:
    """distance matrix

    :parameter
    :param distance_matrix: matrix
    """
    boundary = []
    road = []
    river = []
    civil = []
    new_df = pd.DataFrame(data=distance_matrix)
    for index, row in new_df.iteritems():
        new_df[index] = np.where(new_df[index] == 0, "b",
                                 np.where(new_df[index] == 0.5, "r",
                                          np.where(new_df[index] == 1.5, "ri",
                                                   np.where(new_df[index] == 1.5, "c", new_df[index]))))
        if len(np.where(new_df[index] == "b")[0].tolist()) > 0:
            boundary.append([np.where(new_df[index] == "b")[0].tolist(), index])
        if len(np.where(new_df[index] == "r")[0].tolist()) > 0:
            road.append([np.where(new_df[index] == "r")[0].tolist(), index])
        if len(np.where(new_df[index] == "ri")[0].tolist()) > 0:
            river.append([np.where(new_df[index] == "ri")[0].tolist(), index])
        if len(np.where(new_df[index] == "c")[0].tolist()) > 0:
            civil.append([np.where(new_df[index] == "c")[0].tolist(), index])
    db_data = {"boundary": boundary, "road": road, "river": river, "civil": civil}
    print(db_data)
    return db_data


def db_data_load(matrix):
    db_data = distance_matrix_value(matrix)
    for i, j in db_data.items():
        db = DbConnection()
        for pair_list in j:
            parent = df.iloc[pair_list[1]].values
            child = df.iloc[pair_list[0]].values
            pair = {"parent": parent.tolist(), "child": child.tolist()}
            query = db.get_cursor.mogrify("insert into lat_long_pairs (terain, pairs) values (%s,%s)",
                                          (i, json.dumps(pair)))
            db.insrt_to_db(query)


def db_data_road():
    db = DbConnection()
    con = db.connection_db()
    cursor = con.cursor()
    query = "select lat_long_pairs.pairs from lat_long_pairs where lat_long_pairs.terain='road'"
    cursor.execute(query)
    data = cursor.fetchall()
    import pprint
    pprint.pprint(data)


if __name__ == '__main__':
    df = pd.read_csv("data/latitude_longitude_details.csv")
    distance_matrix = np.asarray(squareform(pdist(df, (lambda u, v: haversine(u, v))))).round(2)
    db_data_road()
    # db_data_load(distance_matrix)
    # cluster_distance_based(distance_matrix)
