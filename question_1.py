import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def find_straight_line(data, lat_min_v, lat_max_v, lon_min_v, lon_max_v, p=0.25) -> None:
    """ Equation applying for straight line.
    y = mx + b.
    This in effect uses x as a parameter and writes
    y as a function of x: y = f(x) = mx+b. When x = 0, y = b and the point (0,b) is the -
    intersection of the line  with the y-axis.
    :parameter

    :param data: data frame
    :param lat_min_v: min value of latitude
    :param lat_max_v:  max value of latitude
    :param lon_min_v:  min value of longitude
    :param lon_max_v:   max value of longitude
    :param p: 0.25 to 0.75

    :return:
    """
    df = data
    m = (lat_max_v - lat_min_v) / (lon_max_v - lon_min_v)
    z = lat_min_v - m * (lon_min_v + p * (lon_max_v - lon_min_v))
    xa = lon_min_v + p * (lon_max_v - lon_min_v)
    xb = lon_max
    df['calculated'] = df['longitude'] * m + z
    df = df[df['longitude'] * m + z < df['latitude']]
    plt.plot([xa, xb], [m * xa + z, m * xb + z])
    plt.plot(df.longitude, df.latitude, 'ro')
    plt.show()


if __name__ == '__main__':
    data = pd.read_csv("data/latitude_longitude_details.csv")
    lat_max = data["latitude"].max()
    lat_min = data["latitude"].min()
    lon_max = data["longitude"].max()
    lon_min = data["longitude"].min()
    find_straight_line(data, lat_min, lat_max, lon_min, lon_max)

