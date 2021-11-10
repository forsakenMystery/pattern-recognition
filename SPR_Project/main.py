import numpy as np
import sys
import os
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE

debug = False


if sys.version_info[0] == 3:
    from urllib.request import urlopen
else:
    from urllib import urlopen

download_dir = "Input"


def make_directory(save_path):
    if not os.path.exists(save_path):
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
            os.makedirs(save_path)
        else:
            os.makedirs(save_path)


def download(file_name, url):
    save_path = os.path.join(download_dir, file_name)
    make_directory(save_path)

    x = os.path.join(save_path, "x.npy")
    y = os.path.join(save_path, "y.npy")

    if not os.path.exists(x) or not os.path.exists(y):
        with urlopen(url) as raw_data:
            print("downloading data")
            digit = np.genfromtxt(raw_data, delimiter=',')
            X_train, y_train = digit[:, :-1], digit[:, -1:].squeeze()
            if not os.path.exists(x):
                np.save(x, X_train)
            if not os.path.exists(y):
                np.save(y, y_train)
    else:
        print("loading data")
        X_train = np.load(x)
        y_train = np.load(y)
    return X_train, y_train


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra"
file_name = "train"
x_train, y_train = download(file_name=file_name, url=url)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes"
file_name = "test"
x_test, y_test = download(file_name=file_name, url=url)

train_size, feature_size = x_train.shape
print(train_size)
X = np.vstack((x_train, x_test))
Y = np.hstack((y_train, y_test))
X, Y = shuffle(X, Y)

if not debug:
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(X[np.where(Y == i)][0].reshape((8, 8)))
        plt.title(str(i))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def feature_selection(X, Y):
    # be jaye inkar ke ye kase konim mishe ye kase nakard o hamoon data ro normalized kard
    # yaani x_train jaye X va x_test o normalized o az onja raft jelo vali in ye kase be nazaram behtare!

    # ba 0.2 split va chizaye dge ham mishe, mishe ham az khode chizi ke ona dadan estefade kard to data set ke manteghi tar i think chon ghabele moghayese!!

    normalized_X = MinMaxScaler().fit_transform(X)
    X = normalized_X
    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    x_train = X[:train_size]
    x_test = X[train_size:]
    y_train = Y[:train_size]
    y_test = Y[train_size:]

    # mishe random state dad vase javab yeksan beyne run ha ye mokhtalef ba adad 9
    # print(X[0])
    features = np.arange(0, X[0].shape[0])
    # print("features name")
    # print(features)
    # print("normalized X")
    # print(normalized_X[0])

    # mishe random state dad vase javab yeksan beyne run ha ye mokhtalef ba adad 42 also mishe movazi kard ke nvm chon bahs time nist n_jobs -1 midam
    # clf = RandomForestClassifier(n_estimators=100, criterion='entropy')
    clf = SVC()
    S = []
    # print("shape of data")
    print(X.shape)
    # print("shape of y data")
    # print(Y.shape)
    for h in range(X.shape[1]):
        print(h+1, "/", X.shape[1])
        kmeans = MiniBatchKMeans(init='k-means++', n_clusters=np.unique(Y).shape[0], batch_size=12)
        ff = kmeans.fit_predict(X[:, h].reshape(-1, 1))
        s = normalized_mutual_info_score(Y, ff)
        S.append(s)
    # print("end of for")
    # print(S)
    print(S)

    plt.bar(features, S, 0.8)
    plt.xticks(features)
    plt.xlim([-1, len(features)])
    plt.title("Feature Quality For Digit Recognition")
    plt.xlabel("Feature Number")
    plt.ylabel("Ranking Values")
    plt.show()

    Z = [x for _, x in sorted(zip(S, features), reverse=True)]

    last = [Z[0]]
    # print(last)

    prev = 0
    print()
    print()

    for i in range(len(Z)):
        # print(last)
        x_train1 = x_train[:, last]
        x_test1 = x_test[:, last]
        clf.fit(x_train1, y_train)
        y_pred = clf.predict(x_test1)
        acc = accuracy_score(y_test, y_pred)
        print("Accuracy: " + str(acc))
        if acc > prev:
            if i != len(Z) - 1:
                last.append(Z[i + 1])
                prev = acc
            else:
                print(last)
        else:
            del last[-1]
            if i != len(Z) - 1:
                last.append(Z[i + 1])
            else:
                print(last)
    tf = len(last)

    print("Considering all features")
    clf_temp = clf.fit(x_train, y_train)
    y_pred = clf_temp.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:   " + str(acc))

    print("Classification")
    print(classification_report(y_test, y_pred))

    clf_temp = clf.fit(x_train[:, last], y_train)

    y_pred = clf_temp.predict(x_test[:, last])

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:    "+str(acc))

    print("Classification")
    print(classification_report(y_test, y_pred))

    Z_temp = [x for _, x in sorted(zip(S, features), reverse=False)]

    last = []
    for i in range(len(Z_temp)):
        last.append(Z_temp[i])
    acc = 0
    prev = 0
    ma = []
    for i in range(len(Z_temp)):
        print(last)
        print(len(last))
        x_train1 = x_train[:, last]
        x_test1 = x_test[:, last]
        clf.fit(x_train1, y_train)
        y_pred = clf.predict(x_test1)
        acc = accuracy_score(y_test, y_pred)
        ma.append(float(str(f"%.2f" % (100*acc))))
        print("Removing pixel " + str(i) + "   " + str(acc))
        print("===========================")
        if acc > prev:
            prev = acc
            j = i
        del last[0]
    last1 = []

    print(ma)
    plt.bar(features, ma, 0.8)
    plt.xticks(features)
    plt.xlim([-1, len(features)])
    plt.title("Accuracy vs. Feature Number For Digit Recognition")
    plt.xlabel("Number of feature eliminated")
    plt.ylabel("Accuracy Percentage")
    plt.show()

    for i in range(len(Z_temp)):
        last1.append(Z_temp[i])
    last1 = last1[j:]
    clf_temp = clf.fit(x_train[:, last1], y_train)
    y_pred = clf_temp.predict(x_test[:, last1])
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy: " + str(acc))

    print("\nClassification - \n")
    print(classification_report(y_test, y_pred))


feature_selection(X, Y)
