# This file is used for cpe RCA algorithm validation
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


def plot_ad_result(data_name, ad_model, feat1, feat2, x_low, x_up, y_low, y_up, bool_name):
    # test model
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
plot_seq = ['Location', 'rsrp', 'rsrq', 'rssi', 'sinr', 'downlink usage', 'uplink usage', 'Throughput', 'Speed']
scaler_kpis = StandardScaler()

# #################### deal with conversion, select a feame only contains inliers######################################
# prepare a set of data that exclude white list

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
train_data = white_data_cpe_alltime
over_all = [0,0,0,0,0,0,0,0,0]
rep = 1

ctmn = []
sys_acc = []
for contmn in range(1, 20, 2):
    contmn = contmn/1000
    ctmn.append(contmn)
    for repti in range(0, rep):
        occurrence_list = []
        train_data, test_data = train_test_split(white_data_cpe_alltime, test_size=0.2)
        test_data = test_data.reset_index()

        # ######################################## AD on GPS ####################################################
        # test_data = pd.read_csv("CPE_White_List_tempVME11.csv")
        # test_data = polluted_cpe

        # model_itree = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.01)
        model_itree = svm.OneClassSVM(nu=contmn, kernel="rbf", gamma=0.1)
        # model_itree = IsolationForest(n_estimators=80, random_state=42, contamination=0.01)

        model_itree.fit(train_data[['longitude', 'latitude']].values)

        test_data['GPS_AD_bool'] = model_itree.predict(test_data[['longitude', 'latitude']].values)
        test_data['GPS_AD_score'] = model_itree.decision_function(test_data[['longitude', 'latitude']].values)

        occurrence_list.append(test_data[test_data.GPS_AD_bool == -1].shape[0])

        # ######################################## AD on other ####################################################

        train_data[kpi_names] = scaler_kpis.fit_transform(train_data[kpi_names])
        test_data[kpi_names] = scaler_kpis.transform(test_data[kpi_names])
        model_oc = LocalOutlierFactor(n_neighbors=80, novelty=True, contamination=contmn)
        # model_oc = svm.OneClassSVM(nu=0.001, kernel="rbf", gamma=0.001)
        for feat in kpi_names:
            model_oc.fit(train_data[feat].values.reshape(-1, 1))
            label_str = feat+"_AD_bool"
            test_data[label_str] = model_oc.predict(test_data[feat].values.reshape(-1, 1))
            occurrence_list.append(len(test_data[test_data[label_str] == -1]))

        rate = [x / test_data.shape[0] for x in occurrence_list]
        over_all = [sum(x) for x in zip(over_all, rate)]

    over_all = [x / rep for x in over_all]
    sys_acc.append((np.mean(over_all))*100)

print(sys_acc)
print(ctmn)
plt.plot(ctmn, sys_acc, linestyle='-', marker='o', color='b')
plt.title('Contamination vs. Accuracy')
plt.xlabel('Contamination Parameter', fontsize=12)
plt.ylabel('Normal Prediction Proportion(%)', fontsize=12)
plt.show()
