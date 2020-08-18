#!/usr/bin/env python
# coding: utf-8

# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


# In[9]:


def getChallengingReqs(y_test, predictions):
    dif = y_test - predictions
    fps, fns = [], []
    index = 0
    for i in dif:
        if i == -2:
            fps.append(y_ids[index])
        elif i == 2:
            fns.append(y_ids[index])
        index = index + 1
    fps_reqs, fns_reqs = [], []
    for i in fps:
        fps_reqs.append(rmix[i])
    for i in fns:
        fns_reqs.append(rmix[i])
    return fps_reqs, fns_reqs


# In[10]:


def updateDic(dic, kname, vname, key):
    index = 0
    flag = True
    for k in dic[kname]:
        if k == key:
            dic[vname][index] = dic[vname][index] + 1
            flag = False
            break
        index = index + 1
    if (flag):
        dic[kname].append(key)
        dic[vname].append(1)
    return dic


# In[11]:


def fitClassifier(Xs, Ys):
    clf = ensemble.ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    kfold = 10
    skf = StratifiedShuffleSplit(n_splits=kfold)
    sets = []
    for train, test in skf.split(Xs, Ys):
        sets.append({'x':train, 'y':test})
    k = 0
    errs = []
    x_errs = []
    while k < kfold:
        x_ids = sets[k]['x']
        y_ids = sets[k]['y']
        x_train = Xs[x_ids]
        x_test = Xs[y_ids]
        y_train, y_test = [], []
        for i in x_ids:
            y_train.append(Ys[i])
        for i in y_ids:
            y_test.append(Ys[i])
        clf.fit(x_train, y_train)
        
        preds = clf.predict(x_test)
        i, w = 0, 0
        eof = len(x_test)
        while i < eof:
            if preds[i] != y_test[i]:
                w = w + 1
            i = i + 1
        errs.append(w/eof)
        
        preds = clf.predict(x_train)
        i, w = 0, 0
        eof = len(x_train)
        while i < eof:
            if preds[i] != y_train[i]:
                w = w + 1
            i = i + 1
        x_errs.append(w/eof)
        k = k + 1
    return x_errs, errs


# In[12]:


import matplotlib.pyplot as plt
def drawLC(x, y):
    train_sizes = [15, 50, 100, 150, 200, 250, 300, 350, 400,  449, 500]
    mean_errs = []
    x_mean_errs = []
    for eof in train_sizes:
        x_errs, errs = fitClassifier(x[:eof], y[:eof])
        s = 0
        for n in errs:
            s = s + n
        s = s / len(errs)
        mean_errs.append(s)
        s = 0
        for n in x_errs:
            s = s + n
        s = s / len(x_errs)
        x_mean_errs.append(s)
    
    
        get_ipython().run_line_magic('matplotlib', 'inline')

    plt.style.use('seaborn')

    plt.plot(train_sizes, x_mean_errs, label = 'Training error')
    plt.plot(train_sizes, mean_errs, label = 'Validation Error')

    plt.ylabel('Mean Squared Error', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    #plt.title('Learning curve', fontsize = 18, y = 1.03)
    plt.legend()
    plt.ylim(0, 0.6)


# In[13]:


def reportHyperParameterTuning(X, y, clf_func, tuned_parameters):
    #skf = StratifiedShuffleSplit(n_splits=1)
    #train_and_test_sets = []
    #for train, test in skf.split(X, Y):
    #    train_and_test_sets.append({'train':train, 'test':test})
    #train_ids = train_and_test_sets[0]['train']
    #test_ids = train_and_test_sets[0]['test']
    #X_train = X[x_ids]
    #X_test = X[test_ids]
    #y_train, y_test = [], []
    #for i in train_ids:
    #    y_train.append(Y[i])
    #for i in test_ids:
    #    y_test.append(Y[i])
    
    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0)
    
    # Set the parameters by cross-validation
    #tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
    #                     'C': [1, 10, 100, 1000]},
    #                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    scores = ['precision', 'recall']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(clf_func, tuned_parameters, cv=5,
                           scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()


# In[ ]:




