import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import numpy as np

class DataSeries:

    def __init__(
        self, 
        name=['time (seconds)', 'data'], 
        series_id_filter=(1000, 5000),
        no_filter=False):
        # logging
        self.logger = logging.getLogger(__name__)

        # range of interest ids for metrics collection
        self.series_id_filter = series_id_filter

        # flag to indicate beginning of logging 
        self.begin_logging = False

        # number of ids collected in range of interest ids
        self.filtered_ids = 0

        # collect metrics in range of interest times
        self.no_filter = no_filter

        # metrics are a data series of two-dimensional (x, y) datapoints
        self.data_series = list()
        # column names of x, y datatpoints for data collection
        self.name = name

        # most recently collected y datapoint for incremental updates 
        # to aid incremental updates to y datapoints
        self.last_data_y = 0

    def __getstate__(self):
        d = self.__dict__.copy()
        if 'logger' in d:
            d['logger'] = d['logger'].name

        return d

    def __setstate__(self, d):
        if 'logger' in d:
            d['logger'] = logging.getLogger(d['logger'])
            self.__dict__.update(d)

    def __len__(self):
        return len(self.data_series)

    # add a new x, y datapoint
    def put(self, data_x, data_y, series_id):
        self.last_data_y = data_y
        if self.series_id_filter[0] <= series_id < self.series_id_filter[1]:
            self.begin_logging = True
            self.data_series.append((data_x, data_y))
        else:
            if self.no_filter:
            #if self.no_filter and self.begin_logging:
                self.data_series.append((data_x, data_y))

    # get most recently collected y datapoint
    def peek_y(self):
        return self.last_data_y

    # convert list of x, y datapoints to a pandas dataframe
    def get_df(self):
        return pd.DataFrame(self.data_series, columns=self.name)

    # add a new x, y datapoint as an incremental (delta) update to 
    # recently collected y datapoint
    def put_delta(self, data_x, data_y_delta, series_id):
        last_data_y = self.peek_y()
        data_y = last_data_y + data_y_delta
        self.put(data_x, data_y, series_id)

    def get_mean_between(self, df, x, y, colname, mean_colname):
        return df.loc[df[colname].between(x,y), mean_colname].mean()

    def plot_step(self, path='./', mean=False, must_print=False, serv_id=None, metric=None):
        plt.figure()
        df = self.get_df()
        if "%H:%M:%S" in self.name[0]:
            df[self.name[0]] = df[self.name[0]].astype('float64')
            df[self.name[0]] = pd.to_datetime(df[self.name[0]], unit='s')
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d,%H:%M:%S'))
            plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=600))
        if "hours" in self.name[0]:
            df[self.name[0]] = df[self.name[0]]/3600
        ax = df.plot(x=self.name[0], drawstyle="steps-post", linewidth=2)
        if mean:
            ymean = self.get_mean_between(df, 1000,2000,self.name[0], self.name[1])
            #print(ymean)
            plt.axhline(y=ymean, color='r', linestyle='--')
        if serv_id is not None or metric is not None:
          self.logger.info("Mean {} for server {} = {}".format(metric, serv_id, str(df[self.name[1]].mean())))   
        elif must_print:
          self.logger.info(
            "Mean of GPU demand : %s", str(df[self.name[1]].mean()))
        ax.set_ylabel(self.name[1])
        ax.set_xlabel(self.name[0])
        plt.gcf().autofmt_xdate()
        plt.savefig(path + "/" + self.name[1] + "_vs_time.png")
        plt.close()
        fname= self.name[1]  + path.replace('/', '') +  ".csv"
        df.to_csv(path + fname)

    def plot_cdf(self, path='./', prefix='job_completion'):
        plt.figure() 
        df = self.get_df()
        if "%H:%M:%S" in self.name[1]:
            df[self.name[1]] = df[self.name[1]].astype('float64')
            df[self.name[1]] = pd.to_datetime(df[self.name[1]], unit='s')
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d,%H:%M:%S'))
            plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=600))
        if "hours" in self.name[1]:
            df[self.name[1]] = df[self.name[1]]/3600
        self.logger.info(
            "Mean of Trace Durations: %s", str(df[self.name[1]].mean()))
        df['cdf'] = df[self.name[1]].rank(method = 'average', pct = True)
        ax = df.sort_values(self.name[1])\
               .plot(x = self.name[1], y = 'cdf', grid = True, linewidth=2,  marker='o', markersize=2, markerfacecolor='red', markeredgecolor='red')
        ax.set_ylabel('cdf')
        ax.set_xlabel(self.name[1])
        plt.xscale("log", basex=2)
        plt.gcf().autofmt_xdate()
        fname='/' + prefix + '_cdf.png'
        plt.savefig(path + fname)
        plt.close()
        fname= path + '/' + prefix +  ".csv"
        df.to_csv(fname)

# collection of several DataSeries
class DataSeriesCollection:

    def __init__(self):
        # logging
        self.logger = logging.getLogger(__name__)

        # collection of named DataSeries
        self.data_series_collection = dict()

    def __getstate__(self):
        d = self.__dict__.copy()
        if 'logger' in d:
            d['logger'] = d['logger'].name
        return d

    def __setstate__(self, d):
        if 'logger' in d:
            d['logger'] = logging.getLogger(d['logger'])
            self.__dict__.update(d)


    def contains(self, index):
        if index in self.data_series_collection:
            return True
        return False

    # add a new DataSeries
    def put(self, index, data_series):
        self.data_series_collection[index] = data_series

    def group_by_index(self, index_id=0):
        group_index_id = index_id
        non_group_index_id = (index_id + 1) % 2
        grouped_dfs = dict()
        for data_series_index in self.data_series_collection:
            non_group_index_name = list(data_series_index)
            group_index_name = data_series_index[group_index_id]
            if group_index_name in non_group_index_name:
                non_group_index_name.remove(group_index_name)
            #print(non_group_index_name)
            data_series = self.data_series_collection[data_series_index]
            df = data_series.get_df()
            if data_series_index[group_index_id] not in grouped_dfs:
                grouped_dfs[data_series_index[group_index_id]] = list()
            grouped_dfs[data_series_index[group_index_id]].append(
                (str(non_group_index_name), df))
                #(data_series_index[non_group_index_id], df))
        return grouped_dfs

    def plot_step(self):
        plt.figure()
        for data_series_index in self.data_series_collection:
            data_series = self.data_series_collection[data_series_index]
            df = data_series.get_df()
            if "%H:%M:%S" in df.columns[0]:
                df[df.columns[0]] = df[df.columns[0]].astype('float64')
                df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], unit='s')
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d,%H:%M:%S'))
                plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=600))
            plt.step(
                df[df.columns[0]], 
                df[df.columns[1]], 
                where='post',
                label=str(data_series_index))
            plt.xlabel(df.columns[0])
            plt.ylabel(df.columns[1])
            plt.gcf().autofmt_xdate()
        plt.legend()
        #plt.show()

    def plot_cdf(self, path='./'):
        grouped_dfs = self.group_by_index(index_id=1)
        done_once=False
        done_list=list()
        for group_name in grouped_dfs:
            plt.figure()
            #print(grouped_dfs)
            fname_mean = str(group_name) + '_avgJCT.csv'
            means = list()
            xvalues= list()
            for label_df in grouped_dfs[group_name]:
                label = label_df[0]
                #print(label, group_name)
                df = label_df[1]
                xvalues.append(label)
                #print(df)
                df['cdf'] = df[df.columns[1]].rank(method = 'average', pct = True)
                df = df.sort_values(df.columns[1])
                if "%H:%M:%S" in df.columns[1]:
                    df[df.columns[1]] = df[df.columns[1]].astype('float64')
                    df[df.columns[1]] = pd.to_datetime(df[df.columns[1]], unit='s')
                    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d,%H:%M:%S'))
                    plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=600))
                self.logger.info(
                    "jct cdf for %s : %s", np.round(group_name, 1), label)
                fname = str(label) + '_' + str(np.round(group_name, 1)) + '.csv'
                #print(fname)
                df.to_csv(fname)
                means.append((df[df.columns[1]].mean())/3600)
                #np.savetxt(fname, df[df.columns[1]]/3600, fmt='%f')
                if 'Synergy' in label or not done_once or label not in done_list:
                    plt.plot(
                    df[df.columns[1]]/3600,
                    df['cdf'],
                    label=label,
                    linewidth=2,
                    marker='o', markersize=1)
                    if 'Synergy' not in label:
                        done_once = True
                        #done_list.append(label)

                plt.ylabel('cdf')
                plt.xlabel(df.columns[1])
                print("Mean = ", df[df.columns[1]].mean()/3600)
                #print("Std = ", df[df.columns[1]].std()/3600)
                #print("90th = ", df[df.columns[1]].quantile(0.9)/3600)
                print("99th = ", df[df.columns[1]].quantile(0.99)/3600)
                print("99.9th = ", df[df.columns[1]].quantile(0.999)/3600)
                #print("99.99th = ", df[df.columns[1]].quantile(0.9999)/3600)
            #plt.xscale("log")
            plt.xscale("log", basex=2)
            plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', ncol=2)
            plt.gcf().autofmt_xdate()
            plt.title("CDF of JCT for load = " + str(np.round(group_name, 1)) + " jobs/hour")
            plt.savefig(path + "cdf_jct_load_"+str(np.round(group_name, 1))+".png", bbox_inches='tight', pad_inches=0.1, dpi=500)
            plt.close()
            c = [xvalues, means]
            #with open(fname_mean, 'w+') as file:
             #   for x in zip(*c):
             #       file.write("{0},{1}\n".format(*x))


    def plot_mean(
        self, 
        xlabel="Load (jobs/hour)", 
        ylabel="Avg. JCT (hours)",
        path="./"):
        grouped_dfs = self.group_by_index(index_id=0)
        plt.figure()
        for group_name in grouped_dfs:
            xvalues = list()
            means = list()
            stds = list()
            for label_df in grouped_dfs[group_name]:
                label = label_df[0].split(',')[0].split('[')[1]
                print(label)
                df = label_df[1]
                xvalues.append(label)
                #print(df[df.columns[1]]/3600)
                fname = group_name + '.csv'
                #np.savetxt(fname, df[df.columns[1]]/3600, fmt='%f')
                #print(" #jobs < 1 hr = ", df[df < 3600 ].count())
                #print(" #jobs > 64 hr = ", df[df > 3600*64 ].count())
                #print("Median = ", df[df.columns[1]].median()/3600)
                #print("Mean = ", df[df.columns[1]].mean()/3600)
                #print("Std = ", df[df.columns[1]].std()/3600)
                #print("90th = ", df[df.columns[1]].quantile(0.9)/3600)
                #print("99th = ", df[df.columns[1]].quantile(0.99)/3600)
                ##print("99.9th = ", df[df.columns[1]].quantile(0.999)/3600)
                #print("99.99th = ", df[df.columns[1]].quantile(0.9999)/3600)
                means.append((df[df.columns[1]].mean())/3600)
                stds.append(df[df.columns[1]].std()/3600)
            plt.plot(xvalues, means, label=group_name, marker="o")
            c = [xvalues, means]
            fname_mean = str(group_name) + "_avg_jct.csv"
            with open(fname_mean, 'w+') as file:
                for x in zip(*c):
                    file.write("{0},{1}\n".format(*x))
        #plt.locator_params(numticks=10)
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.ylim(ymin=0)
        # plt.yscale("log")
        plt.title(ylabel + " vs. " + xlabel)
        fname = "avg_jct_vs_load.png"
        plt.savefig(path + fname)
        plt.close()

    def plot_weighted_mean(
        self, 
        xlabel="Load (jobs/hour)", 
        ylabel="Avg. GPU Demand (%)"):
        grouped_dfs = self.group_by_index(index_id=0)
        plt.figure()
        for group_name in grouped_dfs:
            xvalues = list()
            means = list()
            for label_df in grouped_dfs[group_name]:
                label = label_df[0].split(',')[0].split('[')[1]
                #label = label_df[0]
                df = label_df[1]
                df_shifted = df.shift(-1)
                gpu_demand_area =\
                    sum(
                    (df_shifted[df.columns[0]][:-1] - df[df.columns[0]][:-1]) *\
                        df[df.columns[1]][:-1])
                total_time = df[df.columns[0]][df.index[-1]] - df[df.columns[0]][0]
                avg_gpu_demand = gpu_demand_area / total_time
                xvalues.append(label)
                means.append(avg_gpu_demand)
            plt.plot(xvalues, means, label=group_name, marker="o")
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(ylabel + " vs. " + xlabel)
        plt.savefig("avg_gpu_demand_vs_load.png")
        plt.close()
