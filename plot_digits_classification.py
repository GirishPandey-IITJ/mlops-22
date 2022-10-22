from sklearn import datasets,svm,metrics
from sklearn.model_selection import train_test_split

import pandas as pd

digits = datasets.load_digits()
n_samples = len(digits.target)
data = digits.images.reshape((n_samples,-1))

g_ls = [0.2,0.09,0.05,0.025]
c_ls = [0.2,0.5,0.9,1,1.5]

params = {}
params['gamma'] = g_ls
params['C'] = c_ls

from utils import getall_h_comb
h_pcomb = getall_h_params_comb(params)
hyper_prm = [{'gamma':g,'C':c} for g in g_ls for c in c_ls ]


split_l = [0.2,0.3]
for i in split_l:

    dv_acc, train_acc, test_acc = 0,0,0
    best_model = None
    best_h_params = None
    train_frac = 1-i
    test_frac = i/2
    dev_frac = i/2
    print("train_frac : " , train_frac , " test_frac : "  , test_frac, "dev_frac : " , dev_frac)
    dev_test_frac = 1-train_frac
    x_train, x_dev_test,y_train,y_dev_test = train_test_split(data , 
        digits.target , test_size = dev_test_frac,shuffle = True)

    x_test, x_dev,y_test,y_dev = train_test_split(x_dev_test ,
        y_dev_test, test_size = (dev_frac)/(dev_test_frac) ,
        shuffle = True)
    ls = []
    g_ls = []
    c_ls = []
    tracc_ls = []
    dvacc_ls = []
    tsacc_ls = []

    for cur_h_params in hyper_prm:

        
        #PART-4: Define the model

        # Create a classifier: a support vector classifier
        clf = svm.SVC() #support vector machine

        #PART-4.1: Setting up the hyperparameters
        hyper_params = cur_h_params
        clf.set_params(**hyper_params)

        #PART-6: Train the model
        # Learn the digits on the train subset

        # 2.Train the model for every combination of hyperparameters
        clf.fit(x_train, y_train)

        #print(cur_h_params)

        #PART-7: Get dev set predictions
        predicted_dev = clf.predict(x_dev)
        predicted_train = clf.predict(x_train)
        predicted_test = clf.predict(x_test)

        # 3.Compute the accuracy on the validation set
        cur_acc = metrics.accuracy_score(y_pred = predicted_dev, y_true=y_dev)
        cur_acc_train = metrics.accuracy_score(y_pred = predicted_train, y_true=y_train)
        cur_acc_test = metrics.accuracy_score(y_pred = predicted_test, y_true=y_test)

        
        g_ls.append(cur_h_params['gamma'])
        c_ls.append(cur_h_params['C'])
        tracc_ls.append(cur_acc_train)
        tsacc_ls.append(cur_acc_test)
        dvacc_ls.append(cur_acc)


        # 4.Identify the best combination of hyperparameters for which validation set accuracy is highest
        if cur_acc > dv_acc:
            dv_acc = cur_acc
            best_h_params = cur_h_params
            #print("Current best accuracy with : " + str(best_h_params))
            #print("New best validation accuracy :" + str(best_acc))
        if cur_acc_train > train_acc:
            train_acc= cur_acc_train
        
        if cur_acc_test > test_acc:
            test_acc = cur_acc_test

    dct = {'G': g_ls, 'C': c_ls, 'train_acc': tracc_ls, 'dev_acc': dvacc_ls
    , 'test_acc': tsacc_ls}

    df = pd.DataFrame.from_dict(dct)

    print(df)

    print("Best hyperparameters: ",str(best_h_params))
    print("Best train acc: ",str(train_acc))
    print("Best dev acc: ",str(dv_acc))
    print("Best test acc: ",str(test_acc))
    print("\n")
    print(df.describe())
    print("\n\n")
