import numpy as np
import pandas as pd
import operator
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
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

from statsmodels.stats.outliers_influence import variance_inflation_factor

# ############################################## Pre-processing ##################################################
root_menu = str(os.getcwd())
similarity_threshold = 0.8
min_similarity = 0.6
packet_loss_anomaly_threshold = 0.01
# casting
fragment_start = 9280
fragment_fin = 9415
add_model_mode = 1
plr_target = "DL"
# if the test sample is too little, add some normal data to be predicted
discontinuous = 0
disc_test_start = 333
disc_test_end = 333
disc_ref_start = 432
disc_ref_end = 439
data = pd.read_csv("clean_cell_pm.csv")
# data = pd.read_csv("train_test_pm.csv")


if os.path.exists(root_menu + "\\UL_Model_lib") is False:
        os.mkdir(root_menu + "\\UL_Model_lib")

if os.path.exists(root_menu + "\\DL_Model_lib") is False:
        os.mkdir(root_menu + "\\DL_Model_lib")

if os.path.exists(root_menu + "\\temp_result") is False:
        os.mkdir(root_menu + "\\temp_result")

if os.path.exists(root_menu + "\\UL_Scaler_lib") is False:
        os.mkdir(root_menu + "\\UL_Scaler_lib")

if os.path.exists(root_menu + "\\DL_Scaler_lib") is False:
        os.mkdir(root_menu + "\\DL_Scaler_lib")


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


# ######################################## Correlations ##################################################
def corr_display(data_to_corr, target):
    # type of correlations
    corr_pearson = pd.DataFrame(data_to_corr).corr('pearson').round(2)
    # corr_spearman = pd.DataFrame(data_to_corr).corr('spearman').round(2)

    # Visualizer
    # plt.figure(figsize=(8, 8))
    # sns.heatmap(corr_spearman, annot=True, cmap='coolwarm', linewidths=0.5).set_title('Seaborn')
    # plt.title("Spearman Correlation")
    plt.show()
    sns.heatmap(corr_pearson, annot=True, cmap='mako', linewidths=0.5).set_title('Seaborn')
    plt.title("Pearson Correlation")
    plt.show()

    # Correlation with output variable
    # cor_target = abs(corr_spearman[target])
    # Selecting highly correlated features
    # relevant_features = cor_target[cor_target > 0.3]
    # print("Noticeable Correlation to the Target: \n", relevant_features)


data2 = data[data.columns]
data2["starttime"] = pd.to_datetime(data2["starttime"], format='%Y/%m/%d %H:%M')
data2["stoptime"] = pd.to_datetime(data2["stoptime"], format='%Y/%m/%d %H:%M')
df_obj = data2.select_dtypes(['object'])
data2[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
data2['ecgi'] = data2['ecgi'].astype(str)

# ###########################Use when investigating a specific cell
ecgi_test = data2['ecgi'] == '2708756710728450'

# ######################### set anomaly standard
pl_anomaly = data2[(data2['min5_pdcp_packets_lost_rate_dl'] >= packet_loss_anomaly_threshold)]
pl_ud_anomaly = data2[(data2['min5_pdcp_packets_lost_rate_dl'] >= packet_loss_anomaly_threshold) &
                      (data2['min5_pdcp_packets_lost_rate_ul'] >= packet_loss_anomaly_threshold)]

# build new data set from selected data
data_f = data2.copy()
# data_f = data2[ecgi_test]

# ##########################select kpis##
feature_names = ["min5_pdcp_packets_lost_rate_dl",
                 "min5_pdcp_packets_lost_rate_ul",
                 "min5_effective_dl_cell_throughput",
                 "min5_effective_ul_cell_throughput",
                 "pdcp_total_ul_dl_user_data_payload",
                 "pdcp_user_data_volume_dl_all_qci",
                 "pdcp_user_data_volume_ul_all_qci",
                 "min5_average_pdcp_sdu_delay_dl",
                 "min5_average_pdcp_sdu_drop_rate_dl",
                 "min5_active_and_idle_ues",
                 "min5_active_ues_with_data_beaerer"]
data_kpis = data_f.loc[:, feature_names]



# ##########################SCALING DATA
'''
scaler_kpis = MinMaxScaler(feature_range=(0, 1))
data_kpis_scaled = scaler_kpis.fit_transform(data_kpis)
data_kpis_scaled = pd.DataFrame(data_kpis_scaled, columns=all_features.columns)
'''
# ######################################### Colinearity Elimmination #################################################

# data_kpis_scaled = calculate_vif_(data_kpis_scaled[feature_names])

# ###################################### select train features / Detection rule #######################################
if plr_target == "UL":
    target_AD = data2['UL_AD']
    data_kpis = data_kpis.loc[:, ["min5_pdcp_packets_lost_rate_dl",
                                                "min5_pdcp_packets_lost_rate_ul",
                                                "min5_effective_ul_cell_throughput",
                                                "pdcp_user_data_volume_ul_all_qci",
                                                "pdcp_total_ul_dl_user_data_payload"]]
    path_nav = root_menu + "/UL_Model_lib"
    scaler_path_nav = root_menu + "/UL_Scaler_lib"
else:
    target_AD = data2['DL_AD']
    data_kpis = data_kpis.loc[:, ["min5_pdcp_packets_lost_rate_dl",
                                  "min5_pdcp_packets_lost_rate_ul",
                                  "pdcp_total_ul_dl_user_data_payload",
                                  "min5_average_pdcp_sdu_delay_dl",
                                  "min5_average_pdcp_sdu_drop_rate_dl",
                                  "min5_active_and_idle_ues"]]
    print(data_kpis)
    path_nav = root_menu + "/DL_Model_lib"
    scaler_path_nav = root_menu + "/DL_Scaler_lib"
'''
target = target_AD.iloc[1:9000, 0]
Features = data_kpis_scaled.iloc[1:9000, operator.and_(data_kpis_scaled.columns != 'min5_pdcp_packets_lost_rate_dl',
                                                       data_kpis_scaled.columns != 'min5_pdcp_packets_lost_rate_ul')]
'''
# test data
if discontinuous == 0:
    target_test = target_AD.iloc[fragment_start:fragment_fin]
    Features_test = data_kpis.iloc[fragment_start:fragment_fin, operator.and_(
        data_kpis.columns != 'min5_pdcp_packets_lost_rate_dl',
        data_kpis.columns != 'min5_pdcp_packets_lost_rate_ul')]
else:
    target_to_test = target_AD.iloc[disc_test_start:disc_test_end]
    Features_to_test = data_kpis.iloc[disc_test_start:disc_test_end, operator.and_(
        data_kpis.columns != 'min5_pdcp_packets_lost_rate_dl',
        data_kpis.columns != 'min5_pdcp_packets_lost_rate_ul')]
    target_ref = target_AD.iloc[disc_ref_start:disc_ref_end]
    Features_ref = data_kpis.iloc[disc_ref_start:disc_ref_end, operator.and_(
        data_kpis.columns != 'min5_pdcp_packets_lost_rate_dl',
        data_kpis.columns != 'min5_pdcp_packets_lost_rate_ul')]
    target_test = pd.concat([target_to_test, target_ref], axis=0)
    Features_test = pd.concat([Features_to_test, Features_ref], axis=0)

# ######################################### Decision Tree #################################################
# create model
my_file = Path("/record")
counter = 0
num_of_trees = sum([len(files) for r, d, files in os.walk(path_nav)])
acc_score_note = np.zeros((1, num_of_trees))
name_tag = ["Total Payload",
            "SDU Delay",
            "SDU Drop Rate",
            "Active User"]
class_tag = ["Normal", "Abnormal"]
corr_display(Features_test,'min5_pdcp_packets_lost_rate_dl')

if add_model_mode == 0:
    if Path(path_nav+"/RC_tree_0.joblib").is_file():
        Scaler = load(scaler_path_nav + "/RC_tree_0.joblib")
        for x in range(0, num_of_trees):
            # Scaler = load(scaler_path_nav + "/RC_tree_" + str(x) + ".joblib")
            DT_clf = load(path_nav + "/RC_tree_" + str(x) + ".joblib")
            # np_Features_test = Scaler.transform(Features_test.values)
            # Features_test = pd.DataFrame(np_Features_test, columns=Features_test.columns)
            y_pred_test = DT_clf.predict(Features_test)
            acc_score_note[0, x] = accuracy_score(target_test, y_pred_test)

        max_accuracy = np.amax(acc_score_note)
        max_accuracy_index = np.argmax(acc_score_note)
        if max_accuracy >= similarity_threshold:
            DT = load(path_nav + "/RC_tree_" + str(max_accuracy_index) + ".joblib")
            feature_importances = pd.Series(permutation_importance(DT, Features_test, target_test, n_repeats=10,
                                            random_state=0).importances_mean, index=Features_test.columns)

            # use inbuilt class feature_importances of tree based classifiers
            print("Accuracy Score list: ", acc_score_note)
            print("Root Cause Found: No." + str(max_accuracy_index))
            print("accuracy score of most fit RC model: " + str(max_accuracy))
            feature_importances.nlargest(Features_test.shape[1]).plot(kind='barh')
            plt.title("permutation_importance")
            plt.show()
            DT_feat_importances = pd.Series(DT.feature_importances_, index=Features_test.columns)
            DT_feat_importances.nlargest(Features_test.shape[1]).plot(kind='barh')
            plt.title("Decision Tree Embedded Importance")
            plt.show()
            plt.figure(figsize=(8, 8))
            tree.plot_tree(DT, max_depth=3, filled=True, fontsize=6)
            plt.show()
        else:
            print("None of the existing root cause model fits, training new model to analyse")
            print("Accuracy Score: ", acc_score_note)
            # np_Features_test = Scaler.inverse_transform(Features_test.values)
            # new_scaler = MinMaxScaler(feature_range=(0, 1))
            # np_Features_test = new_scaler.fit_transform(np_Features_test)
            # Features_test = pd.DataFrame(np_Features_test, columns=Features_test.columns)
            new_DT = DecisionTreeClassifier(random_state=0, criterion='gini')
            new_DT.fit(Features_test, target_test)
            DT_feat_importances = pd.Series(new_DT.feature_importances_, index=Features_test.columns)
            DT_feat_importances.nlargest(Features_test.shape[1]).plot(kind='barh')
            plt.show()
            plt.figure(figsize=(8, 8))
            tree.plot_tree(new_DT, feature_names=Features_test.columns, max_depth=3, filled=True, fontsize=6)
            plt.show()
            tree_quality = new_DT.get_depth()
            print("depth of the model: " + str(tree_quality))
            if tree_quality <= 30:
                print("New decision tree qualified, added to library, please add root cause note")
                confirm = input("Save this model to library? (Y/N)")
                if confirm == "Y":
                    file_name = "/RC_tree_" + str(num_of_trees) + ".joblib"
                    dump(new_DT, path_nav + file_name)
                    # dump(new_scaler, scaler_path_nav + file_name)
                else:
                    print("Model not saved")
            else:
                print("The quality of this new model is poor, disqualified for model library")
                print("This anomaly is caused by an unknown factor, insufficient feature to analyse")
    else:
        print("No existing model, please add model")
else:
    scaler_kpis = MinMaxScaler(feature_range=(0, 1))
    # np_Features_test = scaler_kpis.fit_transform(Features_test.values)
    # Features_test = pd.DataFrame(np_Features_test, columns=Features_test.columns)
    # create model
    model = DecisionTreeClassifier(random_state=0, max_leaf_nodes=12, criterion='gini', min_samples_leaf=3, min_samples_split=3)
    model.fit(Features_test, target_test)
    y_pred_test = model.predict(Features_test)
    ####
    Features_test.nunique().plot.bar(figsize=(12, 6))
    plt.ylabel('Number of unique categories', fontsize=14)
    plt.xticks([0,1,2,3],name_tag,rotation=0, fontsize=14)
    plt.title('Cardinality', fontsize=16)
    print(accuracy_score(target_test, y_pred_test))
    corr_display(Features_test,'min5_pdcp_packets_lost_rate_dl')
    # plot permu imp

    feature_importances = pd.Series(permutation_importance(model, Features_test, target_test, n_repeats=10,
                                                           random_state=0).importances_mean, index=Features_test.columns)

    # use inbuilt class feature_importances of tree based classifiers
    feature_importances.nlargest(Features_test.shape[1]).plot(kind='barh')
    plt.title("Permutation Importance", fontsize=16)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.xlabel("Importance Score", fontsize=16)
    plt.show()
    # plot DT imp
    DT_feat_importances = pd.Series(model.feature_importances_, index=Features_test.columns)
    DT_feat_importances.nlargest(Features_test.shape[1]).plot(kind='barh')
    plt.title("Decision Tree Embedded Importance", fontsize=16)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.xlabel("Importance Score", fontsize=16)
    plt.show()
    # plot tree
    plt.figure(figsize=(10, 9))
    tree_quality = model.get_depth()
    print("depth of the model: " + str(tree_quality))
    tree.plot_tree(model, feature_names=name_tag, class_names=class_tag, max_depth=9, filled=True, fontsize=11)
    plt.show()

    confirm = input("Save this model to library? (Y/N)")
    if confirm == "Y":
        file_name = "/RC_tree_" + str(num_of_trees) + ".joblib"
        dump(model, path_nav + file_name)
        # dump(scaler_kpis, scaler_path_nav + file_name)
    else:
        print("Model not saved")

# ######################################### Random Forest #################################################
fig, ax = plt.subplots()
bars = ax.bar(["Scenario 1 RC", "Scenario 2 RC"], [0.46478873, 1])

ax.bar_label(bars)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.ylabel("Model Prediction Accuracy", fontsize=12)
plt.title("Root Cause Model Prediction Accuracy Comparison", fontsize=14)
plt.grid(axis='y')
plt.show()