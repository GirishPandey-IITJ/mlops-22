from sklearn import datasets,svm,metrics,tree
from sklearn.model_selection import train_test_split
import pdb
import pandas as pd
import numpy as np
from joblib import dump,load
digits = datasets.load_digits()
label = digits.target
n_samples = len(label)
data = digits.images.reshape((n_samples,-1))
print(list(data[45]))
l = []
for i in data[45]:
    z = str(i)
    l.append(z)
print(l)
'''g_ls = [0.05,0.025,0.01]
c_ls = [1,1.5,2]
from utilsGirish import (get_h_prm,h_comb,tune_save)
#h_pcomb = getall_h_params_comb(params)
svm_p = {}
svm_p['gamma'] = g_ls
svm_p['C'] = c_ls
svm_hyper_p = get_h_prm(svm_p)

d = [10,20,30,40]

dt_p = {}
dt_p['max_depth'] = d
dt_hyper_p = get_h_prm(dt_p)

svm_dt_hyper = {'dt':dt_hyper_p , 'svm':svm_hyper_p}
#print(svm_dt_hyper)
measure = [metrics.accuracy_score , metrics.mean_absolute_error]
hyper_ms = metrics.accuracy_score

folds = 5

report = {}
f_train = 0.7
f_test = 0.15
f_dev = 0.15
for k in range(folds):
    dev_test_frac = 0.3
    x_train, x_dev_test, y_train, y_dev_test = train_test_split(
        data, label, test_size=dev_test_frac, shuffle=True
    )
    x_test, x_dev, y_test, y_dev = train_test_split(
        x_dev_test, y_dev_test, test_size=(f_dev) / dev_test_frac, shuffle=True
    )
    alg = {"svm":svm.SVC() , "dt":tree.DecisionTreeClassifier()}
    for name in alg:
        mdl = alg[name]
        print("Run [{}] hyper_p tuning {}".format(k,name))
        cur_model_path = tune_save(mdl,x_train,y_train,x_dev,y_dev,hyper_ms,
        svm_dt_hyper[name],model_path=None)

        top_model = load(cur_model_path)
        prd = top_model.predict(x_test)
        if not name in report:
            report[name] = []
        report[name].append(
        {m.__name__:m(y_pred=prd, y_true=y_test) for m in measure}
        )
        cm_dt = pd.DataFrame(metrics.confusion_matrix(y_true = y_test , y_pred = prd))
        print("Confusion Matrix for label comparison\n",cm_dt)
        print(
            f"Classification report for classifier {mdl}:\n"
            f"{metrics.classification_report(y_test, prd)}\n"
        )

k = report
svm_acc = [k['svm'][i]['accuracy_score'] for i in range(5)]
dt_acc = [k['dt'][i]['accuracy_score'] for i in range(5)]
d = {'run':[1,2,3,4,5],'svm_acc':svm_acc , 'dt_acc':dt_acc}
dt_svm = pd.DataFrame(d).set_index('run')

print(dt_svm , "\n")
'''

print("\nMean\n" , dt_svm.mean() , "\n\n" , "Standard Deviation\n" , dt_svm.std() )