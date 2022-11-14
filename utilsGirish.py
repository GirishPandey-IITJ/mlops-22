from sklearn.model_selection import train_test_split
from joblib import dump
from sklearn import svm, tree,metrics
import pdb

def combs(param_vals, param_name, combs_so_far):
    new_combs_so_far = []        
    for c in combs_so_far:        
        for v in param_vals:
            cc = c.copy()
            cc[param_name] = v
            new_combs_so_far.append(cc)
    return new_combs_so_far


def get_h_prm(prm):
    h_prm = [{}]
    for p_nm in prm:
        current_comb = []        
        for i in h_prm:        
            for k in prm[p_nm]:
                cc = i.copy()
                cc[p_nm] = k
                current_comb.append(cc)
        h_prm = current_comb
    #return new_combs_so_far


    return h_prm

def tune_h_prm(h_prm,mdl,x_train,y_train,x_dev, y_dev, metric, verbose=False):
    #print(h_prm)
    top_mtr = -1.0
    top_model = None
    top_h_prm = None

    for i_prm in h_prm:
        #print(i_prm)
        hyper_params = i_prm
        mdl.set_params(**hyper_params)
        mdl.fit(x_train,y_train)

        dev_prd = mdl.predict(x_dev)

        mtr = metric(y_pred=dev_prd, y_true=y_dev)
        if mtr > top_mtr:
            top_mtr  = mtr
            top_model = mdl
            top_h_prm = i_prm
            if verbose:
                print("Found new best metric with :" + str(i_prm))
                print("New best val metric:" + str(mtr))
    return top_model,top_mtr,top_h_prm

def tune_save(
    mdl, x_train, y_train, x_dev,y_dev, metric, h_prm, model_path):
    top_model,top_mtr,top_h_prm = tune_h_prm(h_prm,mdl, x_train, y_train, x_dev,y_dev, metric)

    top_config = "_".join(
        [i + "=" + str(top_h_prm[i]) for i in top_h_prm]
        )
    print(str(mdl))
    if str(mdl)[0] == "D":
        print('dt')
        algo = 'dt'
    if str(mdl)[0] == "S":
        print('svm')
        algo = 'svm'
    top_model_name = algo + '_' + top_config +".joblib"

    if model_path == None:
        model_path = top_model_name
    dump(top_model,model_path)

    print("Best hyperparameters were:" + str(top_h_prm))

    print("Best Metric on Dev was:{}".format(top_mtr))

    return model_path

def h_comb(prm):
    h_param = [{}]
    for i in prm:
        x = []
        for j in h_param:
            for k in prm[i]:
                c = j.copy()
                c[i] = k
                x.append(c)
        h_param = x