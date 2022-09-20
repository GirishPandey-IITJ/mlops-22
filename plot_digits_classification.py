# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

#PART-1: Library Dependencies

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from skimage.transform import rescale
from sklearn.model_selection import train_test_split
import pandas as pd


# 1.Set ranges for hyperparameters
g_ls = [0.1,0.07,0.05,0.01]
c_ls = [0.1,0.3,0.8,1,2.5]


h_param_comb = [{'gamma': g, 'C': c}for g in g_ls for c in c_ls]


train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.


#PART-2: LOAD DATASETS s
digits = datasets.load_digits()


#PART-2.1: Sanity Check Visualization of the data

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.


#PART-3: Data Preprocessing -- to remove some noise, normalize data, 
# transformation to format the data to be consumed by the model

# flatten the images
n_samples = len(digits.images)



img_res_ls = [16,32,64]


#for image in digits.images:
 #   print(image.shape)

#best_acc, best_acc_train, best_acc_test = -1.0, -1.0, -1.0
#best_model = None
#best_h_params = None


for i in img_res_ls:

    dv_acc, train_acc, test_acc = 0,0,0
    best_model = None
    best_h_params = None
    #PART-5: Define train/test/dev splits of experiment protocol
    #train to train model
    #dev to set hyperparameters of the model
    #test to evaluate performance of the model (test on unseen data, to avoid overestimation of performance of the model)
    img = rescale(image, i, anti_aliasing=False)
    print("Resolution : " + str(img.shape))

    data = digits.images.reshape((n_samples, -1))
    #model cannot take a 2d image as input so we convert every images into single array

    dev_test_frac = 1 - train_frac
    #Split into 80:10:10 :: train:dev:test
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(
        data, digits.target, test_size= dev_test_frac, shuffle=True
    )
    #Resplit
    #Split dev_test variables
    X_test, X_dev, y_test, y_dev = train_test_split(
        X_dev_test, y_dev_test, test_size=(dev_frac)/(dev_test_frac), shuffle=True
    )
    '''
    X_test, X_dev, y_test, y_dev = train_test_split(
        X_dev_test, y_dev_test, test_size=(dev_frac)/(test_frac + dev_frac), shuffle=True
    )
    or
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_dev_test, y_dev_test, test_size=(test_frac)/(test_frac + dev_frac), shuffle=True
    )
    '''
    #model hyperparameters
    #GAMMA = 0.001
    #C = 0.5
    ls = []
    g_ls = []
    c_ls = []
    tracc_ls = []
    dvacc_ls = []
    tsacc_ls = []

    for cur_h_params in h_param_comb:

        
        #PART-4: Define the model

        # Create a classifier: a support vector classifier
        clf = svm.SVC() #support vector machine

        #PART-4.1: Setting up the hyperparameters
        hyper_params = cur_h_params
        clf.set_params(**hyper_params)

        #PART-6: Train the model
        # Learn the digits on the train subset

        # 2.Train the model for every combination of hyperparameters
        clf.fit(X_train, y_train)

        #print(cur_h_params)

        #PART-7: Get dev set predictions
        predicted_dev = clf.predict(X_dev)
        predicted_train = clf.predict(X_train)
        predicted_test = clf.predict(X_test)

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

#PART-7.1: Get test set predictions
# Predict the value of the digit on the test subset
predicted = best_model.predict(X_test)


#PART-8: Sanity check of predictions
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")


#print(cur_h_params)

# 5.Report the test set accuracy with the chosen combo of parameters

#PART-9: Compte Evaluation matrices
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.

#disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
#disp.figure_.suptitle("Confusion Matrix")
#print(f"Confusion matrix:\n{disp.confusion_matrix}")
#plt.show()

print("Best hyperparameters were: " + str(cur_h_params))
