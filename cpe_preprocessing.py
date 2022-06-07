import numpy as np
import pandas as pd
import operator
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from minisom import MiniSom
from pylab import plot, axis, show, pcolor, colorbar, bone
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import mutual_info_classif as MIC
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from collections import defaultdict
import os
from pathlib import Path
from joblib import dump, load
import pygeohash as gh
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from datetime import datetime
from datetime import timedelta
from sklearn.metrics import make_scorer, f1_score
from sklearn import model_selection
# ############################################### Parameters ####################################################
fragment_start = 5000
fragment_fin = 10000
ecgi_name = "2708756710728451.0"
packet_loss_anomaly_threshold = 0.05
time_delay_shift = 0
root_menu = str(os.getcwd())


def datetime_format_conv(cpu_date):
    cpe_datetime = datetime.strptime(cpu_date, '%Y/%m/%d %H:%M').strftime('%b %d, %Y @ %H:%M')
    return cpe_datetime


def switch_cell_format(ecgi_in):

    if ecgi_in == "2708756710728193.0":
        cell_name = "VME11"
    elif ecgi_in == "2708756710728449.0":
        cell_name = "VME21"
    elif ecgi_in == "2708756710728194.0":
        cell_name = "VME12"
    elif ecgi_in == "2708756710728707.0":
        cell_name = "VME33"
    elif ecgi_in == "2708756710728706.0":
        cell_name = "VME32"
    elif ecgi_in == "2708756710728450.0":
        cell_name = "VME22"
    elif ecgi_in == "2708756710728195.0":
        cell_name = "VME13"
    elif ecgi_in == "2708756710728451.0":
        cell_name = "VME23"
    elif ecgi_in == "2708756710728705.0":
        cell_name = "VME31"

    return cell_name


'''
print("KPI time format: 2021/10/30 9:35")
print("CPE data format: ", datetime_format_conv("2021/10/30 9:35"))
enc = LabelEncoder()
df['geohash'] = df.apply(lambda x: gh.encode(x.latitude, x.longitude, precision=12), axis=1)
cellular_split = df['modem_name'] == 'Cellular 1'
'''
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# ############################################## Pre-processing ##################################################
data = pd.read_csv("packet_loss_rate.csv")

data2 = data[data.columns].copy()

data2['ecgi'] = data2['ecgi'].astype(str)
# print(data2['ecgi'])
# ###########################Use when investigating a specific cell
ecgi_test = data2['ecgi'] == ecgi_name
data_f = data2[ecgi_test]
data_f = data_f.loc[:, ["starttime", "stoptime", "min5_pdcp_packets_lost_rate_dl", "min5_pdcp_packets_lost_rate_ul"]]
data_f.dropna(inplace=True)
data_f["starttime"] = pd.to_datetime(data_f["starttime"], format='%Y/%m/%d %H:%M')
data_f["stoptime"] = pd.to_datetime(data_f["stoptime"], format='%Y/%m/%d %H:%M')
print('Your new type is: {}'.format(type(data_f["stoptime"])))
# ##########################SCALING DATA
all_features = data_f.loc[:, ["min5_pdcp_packets_lost_rate_dl", "min5_pdcp_packets_lost_rate_ul"]]
data_kpis = all_features.values
# data_kpis = data_f[['min5_pdcp_packets_lost_rate_dl', 'min5_average_pdcp_sdu_delay_dl',
# 'min5_average_pdcp_sdu_drop_rate_dl']]
scaler_kpis = StandardScaler()
data_kpis_scaled = scaler_kpis.fit_transform(data_kpis)
data_kpis_scaled = pd.DataFrame(data_kpis_scaled, columns=all_features.columns)

# ##################################### Anomaly detection by thresholding ############################################
data_f['Target_AD'] = data_f.iloc[:, data_f.columns == 'min5_pdcp_packets_lost_rate_ul'].copy()
data_f.loc[operator.or_(data_f['min5_pdcp_packets_lost_rate_dl'] >= packet_loss_anomaly_threshold,
                        data_f['min5_pdcp_packets_lost_rate_ul'] >= packet_loss_anomaly_threshold), 'Target_AD'] = -1
data_f.loc[operator.and_(data_f['min5_pdcp_packets_lost_rate_dl'] < packet_loss_anomaly_threshold,
                         data_f['min5_pdcp_packets_lost_rate_ul'] < packet_loss_anomaly_threshold), 'Target_AD'] = 1
data_f.to_csv('clean_plr.csv', index=False)

# ###################################### target polt #######################################
data_plot_anomaly = data_f.loc[:, ['min5_pdcp_packets_lost_rate_dl', 'min5_pdcp_packets_lost_rate_ul']]
targ_plot_anomaly = data_f.loc[:, 'Target_AD'].copy()
targ_plot_anomaly['Targ'] = data_f.loc[:, 'Target_AD']

plt.title("Packet Loss Rate Manual Labelling", fontsize=20)
b1 = plt.scatter(data_plot_anomaly.loc[targ_plot_anomaly['Targ'] == 1, 'min5_pdcp_packets_lost_rate_dl'],
                 data_plot_anomaly.loc[targ_plot_anomaly['Targ'] == 1, 'min5_pdcp_packets_lost_rate_ul'],
                 c="green", s=20, edgecolor="k")
c = plt.scatter(data_plot_anomaly.loc[targ_plot_anomaly['Targ'] == -1, 'min5_pdcp_packets_lost_rate_dl'],
                data_plot_anomaly.loc[targ_plot_anomaly['Targ'] == -1, 'min5_pdcp_packets_lost_rate_ul'],
                c="red", s=20, edgecolor="k")
plt.axis("tight")
plt.xlim((-0.03, 1))
plt.ylim((-0.03, 1))
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel("min5 PDCP packets lost rate dl", fontsize=16)
plt.ylabel("min5 PDCP packets lost rate ul", fontsize=16)
plt.legend(
    [b1, c],
    ["regular observations", "abnormal observations"],
    loc="upper right", fontsize=13
)
plt.show()
# ###################################### Auto-AD isolation forest ###########################################
'''
model_itree = IsolationForest(n_estimators=100, max_samples='auto', max_features=1.0)
model_itree.fit(data_kpis_scaled[['min5_pdcp_packets_lost_rate_ul',
                                  'min5_pdcp_packets_lost_rate_dl']].values)

def plot_ad_result(data_name, ad_model, feat1, feat2, x_low, x_up, y_low, y_up, bool_name, score_name):
    # test model
    data_name[bool_name] = model_itree.predict(data_name[[feat1, feat2]].values)
    data_name[score_name] = model_itree.decision_function(data_name[[feat1, feat2]].values)

    data_plot_anomaly = data_name.loc[:, ['longitude', 'latitude']]
    targ_plot_anomaly = data_name.loc[:, bool_name].copy()
    targ_plot_anomaly['Targ'] = data_name.loc[:, bool_name]
    xx, yy = np.meshgrid(np.linspace(x_low, x_up, 50), np.linspace(y_low, y_up, 50))
    Z = model_itree.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.title(ad_model)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
    b1 = plt.scatter(data_plot_anomaly.loc[targ_plot_anomaly['Targ'] == 1, feat1],
                     data_plot_anomaly.loc[targ_plot_anomaly['Targ'] == 1, feat2],
                     c="green", s=20, edgecolor="k")
    c = plt.scatter(data_plot_anomaly.loc[targ_plot_anomaly['Targ'] == -1, feat1],
                    data_plot_anomaly.loc[targ_plot_anomaly['Targ'] == -1, feat2],
                    c="red", s=20, edgecolor="k")
    plt.axis("tight")
    plt.xlim((x_low, x_up))
    plt.ylim((y_low, y_up))
    plt.xlabel(feat1, fontsize=12)
    plt.ylabel(feat2, fontsize=12)
    plt.legend(
        [b1, c],
        ["regular observations", "abnormal observations"],
        loc="upper left",
    )

    plt.show()

plot_ad_result(data_f, "IsolationForest", 'min5_pdcp_packets_lost_rate_dl', 'min5_pdcp_packets_lost_rate_ul',
               -0.01, 0.01, -0.01, 0.02, 'Target_AD', 'Target_AD_score')

'''
# ############# connect
normal_plr_samp = data_f[data_f['Target_AD'] == 1]
normal_plr_samp.to_csv('clean_normal_plr.csv', index=False)
abnormal_plr_samp = data_f[data_f['Target_AD'] == -1]
abnormal_plr_samp.to_csv('clean_abnormal_plr.csv', index=False)


# ################################################################################################################## #
#                                     Add data with normal PLR to white list algorithm                               #
# ################################################################################################################## #


data = pd.read_csv("CPE_data.csv")

data_ue = data[data.columns].copy()
print(data_ue['current_cell_name'].value_counts())
cell_test = data_ue['current_cell_name'] == switch_cell_format(ecgi_name)
data_cpe = data_ue[cell_test]
feature_names = ["starttime","rsrp", "rsrq", "rssi", "signal_quality", "sinr", "rx", "tx", "usage",
                 "longitude", "latitude", "speed", "modem_name", "temperature", "predicted_cell_rsrp"]

data_cpe = data_cpe.loc[:, feature_names]
data_cpe = data_cpe.replace({'-': np.nan})
data_cpe = data_cpe.dropna()

kpi_names = ["rsrp", "rsrq", "rssi", "signal_quality", "sinr",
             "speed", "temperature"]

# ########################## Conversion ##################################################################

data_cpe["starttime"] = pd.to_datetime(data_cpe["starttime"], format='%b %d, %Y @ %H:%M:%S.%f')
data_plr = pd.read_csv("clean_normal_plr.csv")
data_plr_proc = data_plr.copy()
data_plr_proc = data_plr_proc.drop_duplicates(subset=['starttime'], keep='last')
n_samples = data_plr_proc.shape[0]
print(n_samples)

test_data = data_cpe.copy()
white_data_cpe = pd.DataFrame(columns=feature_names)
data_plr_proc["stoptime"] = pd.to_datetime(data_plr_proc["stoptime"], format='%Y/%m/%d %H:%M')
data_plr_proc["starttime"] = pd.to_datetime(data_plr_proc["starttime"], format='%Y/%m/%d %H:%M')


for sample in range(0, n_samples):
    start_time = data_plr_proc.loc[sample, 'starttime']
    end_time = data_plr_proc.loc[sample, 'stoptime']
    mask = (data_cpe['starttime'] > (start_time-timedelta(minutes=time_delay_shift))) & \
           (data_cpe['starttime'] <= (end_time-timedelta(minutes=time_delay_shift)))
    white_data_cpe = white_data_cpe.append(data_cpe.loc[mask, :])

file_name_temp = "CPE_White_List_temp" + switch_cell_format(ecgi_name) + ".csv"
file_name_alltime = "CPE_White_List_all" + switch_cell_format(ecgi_name) + ".csv"
white_data_cpe.to_csv(file_name_temp, index=True)

# ######################################## Turn it on when adding new data sets ########################################

if Path(file_name_alltime).is_file():
    all_ecgi_white_list = pd.read_csv(file_name_alltime)
    all_ecgi_white_list = all_ecgi_white_list.append(white_data_cpe)
    all_ecgi_white_list.to_csv(file_name_alltime, index=False)
else:
    white_data_cpe.to_csv(file_name_alltime, index=False)

