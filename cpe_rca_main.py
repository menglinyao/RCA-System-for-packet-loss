import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import operator
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from minisom import MiniSom
from pylab import plot, axis, show, pcolor, colorbar, bone
import matplotlib.pyplot as plt
from datetime import timedelta
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
from sklearn.metrics import make_scorer, f1_score
from sklearn import model_selection
from sklearn import svm
from locale import atof
import locale
# from pyod.models.auto_encoder import AutoEncoder
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.options.mode.chained_assignment = None  # default='warn'

fragment_start = 5000
fragment_fin = 10000
ecgi_name = "2708756710728451.0"
time_delay_shift = 0
rest_start = "23:00"
rest_end = "06:00"
occurrence_list = []
contam = 0.011


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


def calculate_vif_(X, thresh=5.0):
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables]


def plot_ad_result(data_name, train_name, ad_model, feat1, feat2, x_low, x_up, y_low, y_up, bool_name):
    # test model
    data_plot_anomaly = data_name.loc[:, ['longitude', 'latitude']]
    data_plot_train = train_name.loc[:, ['longitude', 'latitude']]
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
    d = plt.scatter(data_plot_train.loc[:, feat1],
                    data_plot_train.loc[:, feat2],
                    c="white", marker='o', s=20, edgecolor="k")
    plt.axis("tight")
    plt.xlim((x_low, x_up))
    plt.ylim((y_low, y_up))
    plt.xlabel(feat1, fontsize=12)
    plt.ylabel(feat2, fontsize=12)
    plt.legend(
        [b1, c, d],
        ["regular observations", "abnormal observations", "Training Data"],
        loc="upper left",
    )

    plt.show()


# ########################################### Start ####################################################
data = pd.read_csv("CPE_data.csv")

data_ue = data[data.columns].copy()
print(data_ue['current_cell_name'].value_counts())
cell_test = data_ue['current_cell_name'] == switch_cell_format(ecgi_name)
data_cpe = data_ue[cell_test]
feature_names = ["starttime","rsrp", "rsrq", "rssi", "signal_quality", "sinr", "rx", "tx", "usage",
                 "longitude", "latitude", "speed", "modem_name", "temperature", "predicted_cell_rsrp"]
data_cpe["starttime"] = pd.to_datetime(data_cpe["starttime"], format='%b %d, %Y @ %H:%M:%S.%f')
data_cpe = data_cpe.loc[:, feature_names]
data_cpe = data_cpe.replace({'-': np.nan})
data_cpe = data_cpe.dropna()
locale.setlocale(locale.LC_NUMERIC, '')
data_cpe['rx'] = data_cpe['rx'].map(atof)
data_cpe['tx'] = data_cpe['tx'].map(atof)
data_cpe['usage'] = data_cpe['usage'].map(atof)
kpi_names = ["rsrp", "rsrq", "rssi", "sinr", "rx", "tx", "usage", "speed"]
rf_kpi = ["rsrp", "rsrq", "rssi", "sinr"]
kpi_plot = ['RSRP', 'RSRQ', 'RSSI', 'SINR', 'DL Usage', 'UL Usage', 'Total Usage', 'Speed']
plot_seq = ['Location', 'RSRP', 'RSRQ', 'RSSI', 'SINR', 'DL Usage', 'UL Usage', 'Total Usage', 'Speed']
scaler_kpis = StandardScaler()


read_file_str = "CPE_White_List_temp" + switch_cell_format(ecgi_name) + ".csv"
white_ecgi = pd.read_csv(read_file_str, index_col=0)
test_data = data_cpe.copy()
print(white_ecgi)
test_data = test_data.drop(white_ecgi.index, axis=0)  # remove white list data from data set
test_data = test_data.reset_index()
white_data_cpe = white_ecgi.reset_index()
read_file_str_all = "CPE_White_List_all" + switch_cell_format(ecgi_name) + ".csv"
white_data_cpe_alltime = pd.read_csv(read_file_str_all)
white_data_cpe_alltime['rx'] = white_data_cpe_alltime['rx'].map(atof)
white_data_cpe_alltime['tx'] = white_data_cpe_alltime['tx'].map(atof)
white_data_cpe_alltime['usage'] = white_data_cpe_alltime['usage'].map(atof)
print("Whitelist sample: "+str(white_data_cpe_alltime.shape[0]))
train_data = white_data_cpe_alltime

# train_data, test_data = train_test_split(white_data_cpe_alltime, test_size=0.2)
print("test sample: "+str(test_data.shape[0]))
# test_data = test_data.reset_index()


# ######################################## AD on GPS ####################################################
# test_data = pd.read_csv("CPE_White_List_tempVME11.csv")
# test_data = polluted_cpe

# model_itree = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.01)
model_itree = svm.OneClassSVM(nu=contam, kernel="rbf", gamma=0.4)
# model_itree = IsolationForest(n_estimators=80, random_state=42, contamination=0.01)

model_itree.fit(train_data[['longitude', 'latitude']].values)

test_data['GPS_AD_bool'] = model_itree.predict(test_data[['longitude', 'latitude']].values)
test_data['GPS_AD_score'] = model_itree.decision_function(test_data[['longitude', 'latitude']].values)

# white_data_cpe.to_csv('CPE_VME23_white_Report.csv', index=False)
x_lowlim = data_cpe['longitude'].min()-0.05
x_uplim = data_cpe['longitude'].max()+0.05
y_lowlim = data_cpe['latitude'].min()-0.05
y_uplim = data_cpe['latitude'].max()+0.05
# white_data_cpe_alltime['GPS_AD_bool'] = model_itree.predict(white_data_cpe_alltime[['longitude', 'latitude']].values)
# plot_ad_result(white_data_cpe_alltime, 'OneClassSVM', 'longitude', 'latitude', x_lowlim, x_uplim, y_lowlim, y_uplim, 'GPS_AD_bool')
plot_ad_result(test_data, train_data, 'GPS Novelty Detection', 'longitude', 'latitude', x_lowlim, x_uplim, y_lowlim, y_uplim, 'GPS_AD_bool')

occurrence_list.append(test_data[test_data.GPS_AD_bool == -1].shape[0])

# ######################################## AD on other ####################################################
bool_list = []
train_data[kpi_names] = scaler_kpis.fit_transform(train_data[kpi_names])
test_data[kpi_names] = scaler_kpis.transform(test_data[kpi_names])
model_oc = LocalOutlierFactor(n_neighbors=80, novelty=True, contamination=contam)
# model_oc = svm.OneClassSVM(nu=0.001, kernel="rbf", gamma=0.001)


for feat in kpi_names:
    model_oc.fit(train_data[feat].values.reshape(-1, 1))
    label_str = feat+"_AD_bool"
    bool_list.append(label_str)
    test_data[label_str] = model_oc.predict(test_data[feat].values.reshape(-1, 1))
    occurrence_list.append(len(test_data[test_data[label_str] == -1]))



ind = 0
x = [1, 2, 3, 4, 5, 6, 7, 8]
data_plot_anomaly = test_data.loc[:, kpi_names]
targ_plot_anomaly = test_data.loc[:, bool_list].copy()

# #####################################################################
for feat in kpi_names:
    label_str = feat + "_AD_bool"
    x_p = data_plot_anomaly.loc[targ_plot_anomaly[label_str] == 1, feat]
    x_n = data_plot_anomaly.loc[targ_plot_anomaly[label_str] == -1, feat]
    y_p = np.zeros_like(x_p)+x[ind]
    y_n = np.zeros_like(x_n)+x[ind]
    b_p = plt.scatter(x_p, y_p, s=25, c="green")
    c_n = plt.scatter(x_n, y_n, s=25, c="red", marker='x')
    ind += 1

plt.ylim(0, 9.5)
plt.yticks(x, kpi_plot, fontsize=14)
plt.xticks(fontsize=12)
plt.xlabel("Standardised Feature Values", fontsize=12)
plt.title("Abnormality Distribution under PLR Standard", fontsize=14)
plt.legend(
    [b_p, c_n],
    ["regular observations", "abnormal observations"],
    loc="upper right", fontsize=10
)
plt.xlim(-10,15)
plt.grid()
plt.show()

print(np.zeros_like(x)+1)

test_data[kpi_names] = scaler_kpis.inverse_transform(test_data[kpi_names])

# ################ throughput is time sensitive, Consider the rest time Situation, Nobody use, no throughput ##########
# during midnight and early morning, workers are resting, no one use internet hence no throughput is normal
index = pd.DatetimeIndex(test_data['starttime'])
test_data.loc[index.indexer_between_time('23:00', '7:00'), 'rx_AD_bool'] = 1
test_data.loc[index.indexer_between_time('23:00', '7:00'), 'tx_AD_bool'] = 1
test_data.loc[index.indexer_between_time('23:00', '7:00'), 'usage_AD_bool'] = 1

# ######################################## AD on Speed: situation sensitive ###########################################
# when the vessel is stationary/low speed, nothing will happen related to speed
test_data.loc[test_data['speed'] < 3, 'speed_AD_bool'] = 1
test_data.loc[test_data['sinr'] > 20, 'sinr_AD_bool'] = 1

# ####################################### after tune ##########################################
test_data[kpi_names] = scaler_kpis.transform(test_data[kpi_names])

data_plot_anomaly = test_data.loc[:, kpi_names]
targ_plot_anomaly = test_data.loc[:, bool_list].copy()
ind=0
# #####################################################################
for feat in kpi_names:
    label_str = feat + "_AD_bool"
    x_p = data_plot_anomaly.loc[targ_plot_anomaly[label_str] == 1, feat]
    x_n = data_plot_anomaly.loc[targ_plot_anomaly[label_str] == -1, feat]
    y_p = np.zeros_like(x_p)+x[ind]
    y_n = np.zeros_like(x_n)+x[ind]
    b_p = plt.scatter(x_p, y_p, s=25, c="green")
    c_n = plt.scatter(x_n, y_n, s=25, c="red", marker='x')
    ind += 1

plt.ylim(0, 9.5)
plt.yticks(x, kpi_plot, fontsize=14)
plt.xticks(fontsize=12)
plt.xlabel("Standardised Feature Values", fontsize=12)
plt.title("Abnormality Distribution under PLR Standard after Tuning", fontsize=14)
plt.legend(
    [b_p, c_n],
    ["regular observations", "abnormal observations"],
    loc="upper right", fontsize=10
)
plt.xlim(-10,15)
plt.grid()
plt.show()
test_data[kpi_names] = scaler_kpis.inverse_transform(test_data[kpi_names])

# ####################################### Create Anomaly Report ##########################################
print(occurrence_list)
anomaly_cpe_samp = test_data[(test_data == -1).any(axis=1)]
anomaly_cpe_samp.to_csv('CPE_RC_Report.csv', index=False)
print(plot_seq)
print(anomaly_cpe_samp.shape[0]/test_data.shape[0])
fig, ax = plt.subplots()
bars = ax.bar(plot_seq, occurrence_list)

ax.bar_label(bars)
plt.bar(plot_seq, occurrence_list)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12, rotation=45, ha='right', rotation_mode='anchor')
plt.ylabel("Anomaly Occurrence", fontsize=12)
plt.title("Features' Occurrence as Possible Root Cause", fontsize=14)
plt.grid(axis='y')
plt.show()

acc = [99.62656898656898, 99.62723086691086, 99.56509444571428, 98.97789610033641, 98.86616782472903,
       98.9166858311651, 98.67301756147478, 98.44487985432406, 98.28014429584898, 98.03309611610392]
bcc = [x / 100 for x in acc]

AD = [0.5084444444444445, 2.4675555555555557, 5.703111111111111, 9.834666666666667, 14.709333333333333,
      19.754666666666665, 24.95644444444445, 30.464000000000002, 36.462222222222216, 42.95466666666667]

cont = [0.001, 0.003, 0.005, 0.007, 0.009, 0.011, 0.013, 0.015, 0.017, 0.019]

delta = [0.001, 0.003, 0.005, 0.007, 0.009, 0.011, 0.013, 0.015, 0.017, 0.019]

fig, ax = plt.subplots()
ax.plot(cont, bcc, marker='o')
ax.set_xlabel("Contamination Rate Parameter (c)", fontsize=14)
ax.set_ylabel("Normal Prediction Proportion", fontsize=14)
ax.set_xticks(cont)
ax.set_title("Normal Rate vs. Contamination Parameter", fontsize=14)
plt.grid()
plt.show()

# ##########################################################################################
for z in range(0,10):
    delta[z] = abs(cont[z]-(1-acc[z]/100))

print(delta)
fig, ax = plt.subplots()
ax.plot(cont, delta, marker='o')
ax.set_xlabel("Contamination Rate Parameter (c)", fontsize=14)
ax.set_ylabel("Contamination Rate Error", fontsize=14)
ax.set_xticks(cont)
ax.set_title("Contamination Rate Error Analysis", fontsize=14)
plt.grid()
plt.show()

