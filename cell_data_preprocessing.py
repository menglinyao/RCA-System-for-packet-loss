import numpy as np
import pandas as pd
import operator
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

fragment_start = 1
fragment_fin = 25000
packet_loss_anomaly_threshold = 0.03
data = pd.read_csv("Joined_PM_Data-Feb.csv")
data_sel = data[data.columns]
'''
data_sel = data_sel.loc[:, ["starttime",
                            "ecgi",
                            "min5_pdcp_packets_lost_rate_dl",
                            "min5_pdcp_packets_lost_rate_ul"]]
'''
data_sel = data_sel.loc[:, ["starttime",
                            "stoptime",
                            "ecgi",
                            "min5_pdcp_packets_lost_rate_dl",
                            "min5_pdcp_packets_lost_rate_ul",
                            "min5_effective_dl_cell_throughput",
                            "min5_effective_ul_cell_throughput",
                            "pdcp_total_ul_dl_user_data_payload",
                            "pdcp_user_data_volume_dl_all_qci",
                            "pdcp_user_data_volume_ul_all_qci",
                            "min5_average_pdcp_sdu_delay_dl",
                            "min5_average_pdcp_sdu_drop_rate_dl",
                            "min5_active_and_idle_ues",
                            "min5_active_ues_with_data_beaerer"]]

data_sel.dropna(inplace=True)


# Label target (packet loss rate) as classes
data_sel['UL_AD'] = data_sel.iloc[:, data_sel.columns == 'min5_pdcp_packets_lost_rate_ul'].copy()
data_sel['DL_AD'] = data_sel.iloc[:, data_sel.columns == 'min5_pdcp_packets_lost_rate_ul'].copy()
# ######################################### Anomaly Detection By threshold ########################################
data_sel.loc[data_sel['min5_pdcp_packets_lost_rate_dl'] >= packet_loss_anomaly_threshold, 'DL_AD'] = 1
data_sel.loc[data_sel['min5_pdcp_packets_lost_rate_dl'] < packet_loss_anomaly_threshold, 'DL_AD'] = 0
data_sel.loc[data_sel['min5_pdcp_packets_lost_rate_ul'] >= packet_loss_anomaly_threshold, 'UL_AD'] = 1
data_sel.loc[data_sel['min5_pdcp_packets_lost_rate_ul'] < packet_loss_anomaly_threshold, 'UL_AD'] = 0

# isolatioin forest
'''
model_itree1 = IsolationForest(n_estimators=100, max_samples='auto', max_features=1.0)
data_sel['UL_AD'] = model_itree1.fit_predict(data_sel['min5_pdcp_packets_lost_rate_ul'].values.reshape(-1, 1))
model_itree2 = IsolationForest(n_estimators=100, max_samples='auto', max_features=1.0, contamination=0.08)
data_sel['DL_AD'] = model_itree2.fit_predict(data_sel['min5_pdcp_packets_lost_rate_dl'].values.reshape(-1, 1))

'''


ind = 0
x = [1, 2]
kpi_names =["min5_pdcp_packets_lost_rate_ul", "min5_pdcp_packets_lost_rate_dl"]
kpi_plot = ['PDCP UL PLR', 'PDCP DL PLR']
bool_list = ['UL_AD', 'DL_AD']
data_plot_anomaly = data_sel.loc[:, kpi_names]
targ_plot_anomaly = data_sel.loc[:, bool_list].copy()

# ##################################################################### plot
for feat in kpi_names:
    x_p = data_plot_anomaly.loc[targ_plot_anomaly[bool_list[ind]] == 1, feat]
    x_n = data_plot_anomaly.loc[targ_plot_anomaly[bool_list[ind]] == -1, feat]
    y_p = np.zeros_like(x_p)+x[ind]
    y_n = np.zeros_like(x_n)+x[ind]
    b_p = plt.scatter(x_p, y_p, s=25, c="green")
    c_n = plt.scatter(x_n, y_n, s=25, c="red", marker='x')
    ind += 1

plt.ylim(0, 2.5)
plt.yticks(x, kpi_plot, fontsize=14)
plt.xticks(fontsize=12)
plt.xlabel("Standardised Feature Values", fontsize=12)
plt.title("Abnormality Distribution under PLR Standard", fontsize=14)
plt.legend(
    [b_p, c_n],
    ["regular observations", "abnormal observations"],
    loc="upper right", fontsize=10
)
plt.xlim(0,0.05)
plt.grid()
plt.show()

# ############################## manual, multiclass


'''
data_sel.loc[operator.and_(data_sel['min5_pdcp_packets_lost_rate_dl'] < packet_loss_anomaly_threshold,
                           data_sel['min5_pdcp_packets_lost_rate_ul'] >= packet_loss_anomaly_threshold), 'Target_AD']=2
data_sel.loc[operator.and_(data_sel['min5_pdcp_packets_lost_rate_dl'] >= packet_loss_anomaly_threshold,
                           data_sel['min5_pdcp_packets_lost_rate_ul'] >= packet_loss_anomaly_threshold), 'Target_AD']=3
data_sel.loc[operator.and_(data_sel['min5_pdcp_packets_lost_rate_dl'] < packet_loss_anomaly_threshold,
                           data_sel['min5_pdcp_packets_lost_rate_ul'] < packet_loss_anomaly_threshold), 'Target_AD']=0

# ##################################### Anomaly Detection By Isolation Forest #####################################

target_AD = data_sel.iloc[:, data_sel.columns == 'min5_pdcp_packets_lost_rate_dl'].copy()
target_AD = target_AD.rename(columns={"min5_pdcp_packets_lost_rate_dl": "Target_AD"})

model_itree = IsolationForest(n_estimators=50, max_samples='auto', max_features=1.0)
model_itree.fit(data_sel[['min5_pdcp_packets_lost_rate_ul', 'min5_pdcp_packets_lost_rate_dl']].values)
target_AD['Target_AD'] = model_itree.predict(data_sel[['min5_pdcp_packets_lost_rate_ul',
                                                    'min5_pdcp_packets_lost_rate_dl']].values).astype(int)
print(target_AD['Target_AD'].dtypes)

data_plot_anomaly = data_sel.loc[fragment_start:fragment_fin, ['min5_pdcp_packets_lost_rate_dl',
                                                            'min5_pdcp_packets_lost_rate_ul']]
targ_plot_anomaly = target_AD.loc[fragment_start:fragment_fin, 'Target_AD'].copy()
targ_plot_anomaly['Targ'] = target_AD.loc[fragment_start:fragment_fin, 'Target_AD']
xx, yy = np.meshgrid(np.linspace(-0.01, 0.12, 50), np.linspace(-0.01, 0.3, 50))
Z = model_itree.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("Isolation Forest Anomaly Detection", fontsize=20)
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
b1 = plt.scatter(data_plot_anomaly.loc[targ_plot_anomaly['Targ'] == 1, 'min5_pdcp_packets_lost_rate_dl'],
                 data_plot_anomaly.loc[targ_plot_anomaly['Targ'] == 1, 'min5_pdcp_packets_lost_rate_ul'],
                 c="green", s=20, edgecolor="k")
c = plt.scatter(data_plot_anomaly.loc[targ_plot_anomaly['Targ'] == -1, 'min5_pdcp_packets_lost_rate_dl'],
                data_plot_anomaly.loc[targ_plot_anomaly['Targ'] == -1, 'min5_pdcp_packets_lost_rate_ul'],
                c="red", s=20, edgecolor="k")
plt.axis("tight")
plt.xlim((-0.01, 0.12))
plt.ylim((-0.01, 0.12))
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.xlabel("min5_pdcp_packets_lost_rate_dl", fontsize=12)
plt.ylabel("min5_pdcp_packets_lost_rate_ul", fontsize=12)
plt.legend(
    [b1, c],
    ["regular observations", "abnormal observations"],
    loc="upper right", fontsize=15
)
plt.show()

'''

data_sel.to_csv('clean_cell_pm.csv', index=False)
# ######################################## sdu sim #####################################

norm_range = data_sel.iloc[0:200, :]

mask = (norm_range['UL_AD'] == 0) &\
       ((norm_range['DL_AD'] == 0) & (norm_range['min5_average_pdcp_sdu_delay_dl'] < 89))

train_test_samp = norm_range.loc[mask, :]
abnormal_mask = (data_sel['DL_AD'] == 1) & (data_sel['min5_average_pdcp_sdu_delay_dl'] > 89)
train_test_samp = train_test_samp.append(data_sel.loc[abnormal_mask, :])
train_test_samp = train_test_samp.sample(frac=1).reset_index(drop=True)
train_test_samp.to_csv('train_test_pm.csv', index=False)
