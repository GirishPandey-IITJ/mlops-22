## mlops-22 Assignment 4
RUN[5](5th split) confusion matrix comparison between labels  
(this is done for all runs)(STEP 5 Explanation)
    0   1   2   3   4   5   6   7   8   9
0  25   0   0   0   0   1   0   0   0   0
1   0  27   0   0   0   0   0   0   1   0
2   2   1  18   0   0   0   1   0   0   0
3   0   0   0  26   0   1   1   0   3   0
4   0   1   0   0  29   0   0   1   0   0
5   0   0   1   0   0  27   0   0   0   1
6   0   1   0   0   2   1  16   0   1   0
7   0   1   0   0   1   0   0  22   0   1
8   0   8   2   1   0   0   0   0  17   1
9   0   0   0   0   0   0   0   1   0  27

# FINAL REPORT
    svm_acc    dt_acc
run
1    0.748148  0.855556
2    0.844444  0.818519
3    0.862963  0.877778
4    0.751852  0.844444
5    0.781481  0.866667

# STEP 3
Mean
svm_acc    0.797778
dt_acc     0.852593
dtype: float64

Standard Deviation
 svm_acc    0.053068
dt_acc     0.022741
dtype: float64