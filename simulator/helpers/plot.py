from pandas import read_csv
from matplotlib import pyplot

def plot_time_series_from_file(csv_file):
    series = read_csv(csv_file, header=0, index_col=0, squeeze=True)

def plot_time_series(series):
    series.plot()
    pyplot.show()
