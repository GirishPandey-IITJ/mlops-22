import matplotlib.pyplot as plt
from sklearn import datasets,svm,metrics
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1,ncols=4,figsize = (10,3))

GAMMA = 0.008
train_frac = 0.7
test_frac = 0.2
dev_frac = 0.1

for ax,image,lb in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image,cmap = plt.cm.gray_r,interpolation = 'nearest')
    ax.set_title("Training : %i" % lb)

n_samples = len(digits.target)
data = digits.images.reshape((n_samples,-1))

dev_test_frac = 1-train_frac
x_train, x_dev_test,y_train,y_dev_test = train_test_split(data , 
    digits.target , test_size = dev_test_frac,shuffle = True)

x_test, x_dev,y_test,y_dev = train_test_split(x_dev_test ,
    y_dev_test, test_size = (dev_frac)/(dev_test_frac) ,
    shuffle = True)


clf = svm.SVC(gamma = 0.01)
hyper_prm = {'gamma':GAMMA}
clf.set_params(**hyper_prm )

clf.fit(x_train , y_train)

prd = clf.predict(x_test)

_, axes = plt.subplots(nrows=1,ncols=4,figsize = (10,3))

for ax,image,prediction in zip(axes,x_test,prd):
    ax.set_axis_off()
    image = image.reshape(8,8)
    ax.imshow(image,cmap = plt.cm.gray_r , interpolation = 'nearest')
    ax.set_title(f"prediction : {prediction}")

print(
    f"Classification report of {clf} :\n"
    f"{metrics.classification_report(y_test,prd)}\n"
)