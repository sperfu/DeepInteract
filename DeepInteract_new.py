# -*- coding: utf-8 -*-
###THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import svm, grid_search
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn import metrics

#import tensorflow as tf
#tf.python.control_flow_ops = tf
from sklearn.cross_validation import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.cluster import KMeans,Birch,MiniBatchKMeans
import gzip
import pandas as pd
import pdb
import random
from random import randint
import scipy.io
import xlwt

from keras.layers import Input, Dense
from keras.engine.training import Model
from keras.models import Sequential, model_from_config
from keras.layers.core import  Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam
from keras.layers import normalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras import regularizers
from keras.constraints import maxnorm

from DeepFunction import multiple_layer_autoencoder,autoencoder_two_subnetwork_fine_tuning,last_layer_autoencoder

def prepare_data(seperate=False):
    print "loading data"
    #miRNA_fea = np.loadtxt("circRNA_functional_sim.txt",dtype=float,delimiter=",")
    miRNA_fea = np.loadtxt("circRNA_sim.txt",dtype=float,delimiter=",")
    #disease_fea = np.loadtxt("disease_functional_sim2.txt",dtype=float,delimiter=",")
    disease_fea = np.loadtxt("disease_sim2.txt",dtype=float,delimiter=",")
    #interaction = np.loadtxt("circRNA_disease_matrix_array_inter2.txt",dtype=int,delimiter=",")
    interaction2 = np.loadtxt("p_value_list.txt",dtype=float,delimiter=",")
    interaction = np.loadtxt("circRNA_disease_matrix_array5.txt",dtype=int,delimiter=",")
    
    link_number = 0
    train = []
    label = []
    label2 = []
    link_position = []
    nonLinksPosition = []  # all non-link position^M
    for i in range(0, interaction.shape[0]):
        for j in range(0, interaction.shape[1]):
            label.append(interaction[i,j])
            label2.append(interaction2[i,j])
            if interaction[i, j] == 1:
                link_number = link_number + 1
                link_position.append([i, j])
                miRNA_fea_tmp = list(miRNA_fea[i])
		disease_fea_tmp = list(disease_fea[j])
                
            elif interaction[i,j] == 0:
                nonLinksPosition.append([i, j])
                miRNA_fea_tmp = list(miRNA_fea[i])
		disease_fea_tmp = list(disease_fea[j])
	    if seperate:
                tmp_fea = (miRNA_fea_tmp,disease_fea_tmp)
            else:
		tmp_fea = miRNA_fea_tmp + disease_fea_tmp
            train.append(tmp_fea)
    return np.array(train), label,label2

def calculate_performace(test_num, pred_y,  labels):
    tp =0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] ==1:
            if labels[index] == pred_y[index]:
                tp = tp +1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn +1
            else:
                fp = fp + 1               
            
    acc = float(tp + tn)/test_num
    precision = float(tp)/(tp+ fp)
    sensitivity = float(tp)/ (tp+fn)
    specificity = float(tn)/(tn + fp)
    MCC = float(tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    return acc, precision, sensitivity, specificity, MCC 

def transfer_array_format(data):
    formated_matrix1 = []
    formated_matrix2 = []
    #pdb.set_trace()
    #pdb.set_trace()
    for val in data:
        #formated_matrix1.append(np.array([val[0]]))
        formated_matrix1.append(val[0])
        formated_matrix2.append(val[1])
        #formated_matrix1[0] = np.array([val[0]])
        #formated_matrix2.append(np.array([val[1]]))
        #formated_matrix2[0] = val[1]      
    
    return np.array(formated_matrix1), np.array(formated_matrix2)

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

def DNN():
    model = Sequential()
    model.add(Dense(input_dim=1027, output_dim=500,init='glorot_normal')) ## 1027  128
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    #model.add(Dense(input_dim=300, output_dim=300,init='glorot_normal'))  ##500
    #model.add(Activation('relu'))
    #model.add(Dropout(0.3))

    model.add(Dense(input_dim=500, output_dim=300,init='glorot_normal'))  ##500
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(input_dim=300, output_dim=2,init='glorot_normal'))  ##500
    model.add(Activation('sigmoid'))
    #sgd = SGD(l2=0.0,lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)
    model.compile(loss='binary_crossentropy', optimizer=adadelta, class_mode="binary")##rmsprop sgd
    return model

def DNN2():
    model = Sequential()
    model.add(Dense(input_dim=64, output_dim=500,init='glorot_normal')) ## 1027  128 32
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(input_dim=500, output_dim=500,init='glorot_normal'))  ##500
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(input_dim=500, output_dim=300,init='glorot_normal'))  ##500
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(input_dim=300, output_dim=2,init='glorot_normal'))  ##500
    model.add(Activation('sigmoid'))
    #sgd = SGD(l2=0.0,lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)
    model.compile(loss='binary_crossentropy', optimizer=adadelta, class_mode="binary")##rmsprop sgd
    return model



def DeepInteract():
    X, labels, labels2 = prepare_data(seperate = True)
    '''
    neg_tmp = [index for index,value in enumerate(labels) if value == 0]
    np.random.shuffle(neg_tmp)
    pos_tmp = [index for index,value in enumerate(labels) if value == 1]
    pos_X = X[pos_tmp]
    neg_X = X[neg_tmp[:len(pos_tmp)]]
    pos_labels = [labels[item] for item in pos_tmp]
    neg_labels = [labels[item] for item in neg_tmp[:len(pos_tmp)]]
    X_new = np.vstack((pos_X,neg_X))
    labels_new = pos_labels+neg_labels
    '''
    import pdb
    X_data1, X_data2 = transfer_array_format(X) # X X_new
    print X_data1.shape,X_data2.shape
    y, encoder = preprocess_labels(labels)# labels labels_new
    y2 = np.array(labels2)# labels labels_new

    num = np.arange(len(y))
    np.random.shuffle(num)
    '''
    X_data1 = X_data1[num]
    X_data2 = X_data2[num]
    y = y[num]
    y2 = y2[num]
    '''
    num_cross_val = 5
    all_performance = []
    all_performance_rf = []
    all_performance_bef = []
    all_performance_DNN = []
    all_performance_SDADNN = []
    all_performance_blend = []
    all_labels = []
    all_prob = {}
    num_classifier = 3
    all_prob[0] = []
    all_prob[1] = []
    all_prob[2] = []
    all_prob[3] = []
    all_averrage = []
    for fold in range(num_cross_val):
        train1 = np.array([x for i, x in enumerate(X_data1) if i % num_cross_val != fold])
        test1 = np.array([x for i, x in enumerate(X_data1) if i % num_cross_val == fold])
        train2 = np.array([x for i, x in enumerate(X_data2) if i % num_cross_val != fold])
        test2 = np.array([x for i, x in enumerate(X_data2) if i % num_cross_val == fold])
        train_label = np.array([x for i, x in enumerate(y) if i % num_cross_val != fold])
        test_label = np.array([x for i, x in enumerate(y) if i % num_cross_val == fold])
	#pdb.set_trace()
        #train_label2 = np.array([x for i, x in enumerate(y2) if i % num_cross_val != fold])
        train_label2 = np.array([x for i, x in enumerate(y2) if i % num_cross_val != fold])
        test_label2 = np.array([x for i, x in enumerate(y2) if i % num_cross_val == fold])
  
          
        real_labels = []
        for val in test_label:
            if val[0] == 1:
                real_labels.append(0)
            else:
                real_labels.append(1)
	'''
	real_labels2 = []
        for val in test_label2:
            if val[0] == 1:
                real_labels2.append(0)
            else:
                real_labels2.append(1)
	'''
        train_label_new = []
        for val in train_label:
            if val[0] == 1:
                train_label_new.append(0)
            else:
                train_label_new.append(1)
        
        blend_train = np.zeros((train1.shape[0], num_classifier)) # Number of training data x Number of classifiers
        blend_test = np.zeros((test1.shape[0], num_classifier)) # Number of testing data x Number of classifiers 
        skf = list(StratifiedKFold(train_label_new, num_classifier))  
        class_index = 0
        #prefilter_train, prefilter_test, prefilter_train_bef, prefilter_test_bef = autoencoder_two_subnetwork_fine_tuning(train1, train2, train_label, test1, test2, test_label)
        #prefilter_train_bef, prefilter_test_bef = autoencoder_two_subnetwork_fine_tuning(train1, train2, train_label, test1, test2, test_label)
        prefilter_train_bef, prefilter_test_bef = autoencoder_two_subnetwork_fine_tuning(X_data1, X_data2, train_label, test1, test2, test_label)
        #X_train1_tmp, X_test1_tmp, X_train2_tmp, X_test2_tmp, model = autoencoder_two_subnetwork_fine_tuning(train1, train2, train_label, test1, test2, test_label)
        #model = autoencoder_two_subnetwork_fine_tuning(train1, train2, train_label, test1, test2, test_label)
        #model = merge_seperate_network(train1, train2, train_label)
        #proba = model.predict_proba([test1, test2])[:1]
        
        real_labels = []
        for val in test_label:
            if val[0] == 1:
                real_labels.append(0)
            else:
                real_labels.append(1)
                
        all_labels = all_labels + real_labels
	all_data_labels = real_labels + train_label_new

	all_prefilter_data = np.vstack((prefilter_test_bef,prefilter_train_bef))
	all_label2_data = np.vstack((test_label2.reshape(test_label2.shape[0],1),train_label2.reshape(train_label2.shape[0],1)))
        #prefilter_train, new_scaler = preprocess_data(prefilter_train, stand =False)
        #prefilter_test, new_scaler = preprocess_data(prefilter_test, scaler = new_scaler, stand = False)
	true_data = np.hstack((train1[46529,:],train2[46529,:])) # 61713
	#true_data = np.vstack((prefilter_train_bef[46529,:],prefilter_train_bef[64833,:])) # 61713
        #false_data = np.vstack((prefilter_train_bef[46528,:],prefilter_train_bef[64834,:]))
        false_data = np.hstack((train1[46528,:],train2[46529,:]))
	#pdb.set_trace()
        '''
        prefilter_train1 = xgb.DMatrix( prefilter_train, label=train_label_new)
        evallist  = [(prefilter_train1, 'train')]
        num_round = 10
        clf = xgb.train( plst, prefilter_train1, num_round, evallist )
        prefilter_test1 = xgb.DMatrix( prefilter_test)
        ae_y_pred_prob = clf.predict(prefilter_test1)
        '''
	'''
        tmp_aver = [0] * len(real_labels)
        print 'deep autoencoder'
        clf = RandomForestClassifier(n_estimators=50)
        clf.fit(prefilter_train_bef, train_label_new)
        ae_y_pred_prob = clf.predict_proba(prefilter_test_bef)[:,1]
        all_prob[class_index] = all_prob[class_index] + [val for val in ae_y_pred_prob]
        tmp_aver = [val1 + val2/3 for val1, val2 in zip(ae_y_pred_prob, tmp_aver)]
        proba = transfer_label_from_prob(ae_y_pred_prob)
        #pdb.set_trace()            
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), proba,  real_labels)
	fpr, tpr, auc_thresholds = roc_curve(real_labels, ae_y_pred_prob)
	auc_score = auc(fpr, tpr)
	#scipy.io.savemat('deep',{'fpr':fpr,'tpr':tpr,'auc_score':auc_score})
	## AUPR score add 
        precision1, recall, pr_threshods = precision_recall_curve(real_labels, ae_y_pred_prob)
        aupr_score = auc(recall, precision1)
	#scipy.io.savemat('deep_aupr',{'recall':recall,'precision':precision1,'aupr_score':aupr_score})
        print acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score
	all_performance.append([acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score])
	'''
	print 'deep autoencoder without fine tunning'
        class_index = class_index + 1
        #clf = RandomForestClassifier(n_estimators=50)
	#pdb.set_trace()
	#clf = KMeans(n_clusters=2, random_state=0).fit(prefilter_train_bef)
	#clf = MiniBatchKMeans(n_clusters=2, init=np.vstack((false_data,true_data)),max_iter=1).fit(np.vstack((false_data,true_data)))
        #clf = KMeans(n_clusters=2, init=np.vstack((false_data,true_data)),max_iter=1).fit(np.vstack((false_data,true_data)))
        #clf.fit(prefilter_train_bef, train_label_new)
        #ae_y_pred_prob = clf.predict(prefilter_test_bef)#[:,1]
        pdb.set_trace()            
	#prefilter_train_bef2 = np.hstack((all_prefilter_data,all_label2_data))
	prefilter_train_bef2 = np.hstack((prefilter_train_bef,y2.reshape(y2.shape[0],1)))
	prefilter_test_bef2 = np.hstack((prefilter_test_bef,test_label2.reshape((test_label2.shape[0],1))))
	#ae_y_pred_prob = last_layer_autoencoder(prefilter_train_bef2,all_data_labels, activation = 'sigmoid', batch_size = 100, nb_epoch = 100, last_dim = 2)
	ae_y_pred_prob = last_layer_autoencoder(prefilter_train_bef2,all_data_labels, activation = 'sigmoid', batch_size = 100, nb_epoch = 100, last_dim = 2)
	workbook = xlwt.Workbook()
	worksheet = workbook.add_sheet('My')
	i_tmp =0
	for line_i in range(ae_y_pred_prob.shape[0]):
	    if round(ae_y_pred_prob[line_i,1],4) > 0.5:
	        worksheet.write(i_tmp,0,line_i)
	        worksheet.write(i_tmp,1,line_i/104)
	        worksheet.write(i_tmp,2,round(ae_y_pred_prob[line_i,1],4))
	        worksheet.write(i_tmp,3,line_i%104+1000)
	        worksheet.write(i_tmp,4,"Undirected")
		i_tmp = i_tmp + 1
	workbook.save('cluster_Workbook1.xls')
	
	workbook = xlwt.Workbook()
        worksheet = workbook.add_sheet('My')
	i_tmp =0
        for line_i in range(ae_y_pred_prob.shape[0]):
	    if round(ae_y_pred_prob[line_i,0],4) > 0.5:
                worksheet.write(i_tmp,0,line_i)
                worksheet.write(i_tmp,1,line_i/104)
                worksheet.write(i_tmp,2,round(ae_y_pred_prob[line_i,0],4))
                worksheet.write(i_tmp,3,line_i%104+1000)
                worksheet.write(i_tmp,4,"Undirected")
		i_tmp = i_tmp + 1
        workbook.save('cluster_Workbook2.xls')
	pdb.set_trace()
	clf = KMeans(n_clusters=2, random_state=0).fit(prefilter_train_bef2)
	#clf = KMeans(n_clusters=2, random_state=0).fit(all_prefilter_data)
	#ae_y_pred_prob = clf.predict(prefilter_train_bef2)#(prefilter_train_bef2)
	ae_y_pred_prob = clf.predict(prefilter_train_bef2)
	'''
	if ae_y_pred_prob[0][0] > ae_y_pred_prob[0][1]:
	    aha = 1
	else:
	    aha = 0
        '''
	#pdb.set_trace()            
        proba = transfer_label_from_prob(ae_y_pred_prob)
        #pdb.set_trace()            
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(all_data_labels), proba,  all_data_labels)
        fpr, tpr, auc_thresholds = roc_curve(all_data_labels, ae_y_pred_prob)
        auc_score = auc(fpr, tpr)
	#scipy.io.savemat('deep_without',{'fpr':fpr,'tpr':tpr,'auc_score':auc_score})
        ## AUPR score add 
        precision1, recall, pr_threshods = precision_recall_curve(all_data_labels, ae_y_pred_prob)
        aupr_score = auc(recall, precision1)
        #scipy.io.savemat('deep_without_aupr',{'recall':recall,'precision':precision1,'aupr_score':aupr_score})
        print acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score
        all_performance_bef.append([acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score])

	print 'random forest using raw feature'
        class_index = class_index + 1
        prefilter_train = np.concatenate((train1, train2), axis = 1)
        prefilter_test = np.concatenate((test1, test2), axis = 1)

        #clf = RandomForestClassifier(n_estimators=50)
        clf = AdaBoostClassifier(n_estimators=50)
        #clf = DecisionTreeClassifier()
        clf.fit(prefilter_train_bef, train_label_new)
        ae_y_pred_prob = clf.predict_proba(prefilter_test_bef)[:,1]
        all_prob[class_index] = all_prob[class_index] + [val for val in ae_y_pred_prob]
        tmp_aver = [val1 + val2/3 for val1, val2 in zip(ae_y_pred_prob, tmp_aver)]
        proba = transfer_label_from_prob(ae_y_pred_prob)

        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), proba,  real_labels)
        fpr, tpr, auc_thresholds = roc_curve(real_labels, ae_y_pred_prob)
        auc_score = auc(fpr, tpr)
	scipy.io.savemat('raw',{'fpr':fpr,'tpr':tpr,'auc_score':auc_score})
	## AUPR score add 
        precision1, recall, pr_threshods = precision_recall_curve(real_labels, ae_y_pred_prob)
        aupr_score = auc(recall, precision1)
	#scipy.io.savemat('raw_aupr',{'recall':recall,'precision':precision1,'aupr_score':aupr_score})
        print acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score
        all_performance_rf.append([acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score])
        ### Only RF
 	clf = RandomForestClassifier(n_estimators=50)
        #clf = AdaBoostClassifier(n_estimators=50)
        #clf = DecisionTreeClassifier()
        clf.fit(prefilter_train_bef, train_label_new)
        ae_y_pred_prob = clf.predict_proba(prefilter_test_bef)[:,1]
        #all_prob[class_index] = all_prob[class_index] + [val for val in ae_y_pred_prob]
        #tmp_aver = [val1 + val2/3 for val1, val2 in zip(ae_y_pred_prob, tmp_aver)]
        proba = transfer_label_from_prob(ae_y_pred_prob)

        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), proba,  real_labels)
        fpr, tpr, auc_thresholds = roc_curve(real_labels, ae_y_pred_prob)
        auc_score = auc(fpr, tpr)
        #scipy.io.savemat('raw',{'fpr':fpr,'tpr':tpr,'auc_score':auc_score})
        ## AUPR score add
        precision1, recall, pr_threshods = precision_recall_curve(real_labels, ae_y_pred_prob)
        aupr_score = auc(recall, precision1)
        #scipy.io.savemat('raw_aupr',{'recall':recall,'precision':precision1,'aupr_score':aupr_score})
        print "RF :", acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score

        ## DNN 
        class_index = class_index + 1
        prefilter_train = np.concatenate((train1, train2), axis = 1)
        prefilter_test = np.concatenate((test1, test2), axis = 1)
        model_DNN = DNN()
        train_label_new_forDNN = np.array([[0,1] if i == 1 else [1,0] for i in train_label_new])
        model_DNN.fit(prefilter_train,train_label_new_forDNN,batch_size=200,nb_epoch=20,shuffle=True,validation_split=0)
        proba = model_DNN.predict_classes(prefilter_test,batch_size=200,verbose=True)
        ae_y_pred_prob = model_DNN.predict_proba(prefilter_test,batch_size=200,verbose=True)
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), proba,  real_labels)
        fpr, tpr, auc_thresholds = roc_curve(real_labels, ae_y_pred_prob[:,1])
        auc_score = auc(fpr, tpr)
        scipy.io.savemat('raw_DNN',{'fpr':fpr,'tpr':tpr,'auc_score':auc_score})
        ## AUPR score add 
        precision1, recall, pr_threshods = precision_recall_curve(real_labels, ae_y_pred_prob[:,1])
        aupr_score = auc(recall, precision1)
        print "RAW DNN:",acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score
        all_performance_DNN.append([acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score])

	## SDA + DNN
        class_index = class_index + 1
        model_DNN = DNN2()
        train_label_new_forDNN = np.array([[0,1] if i == 1 else [1,0] for i in train_label_new])
        model_DNN.fit(prefilter_train_bef,train_label_new_forDNN,batch_size=200,nb_epoch=20,shuffle=True,validation_split=0)
        proba = model_DNN.predict_classes(prefilter_test_bef,batch_size=200,verbose=True)
        ae_y_pred_prob = model_DNN.predict_proba(prefilter_test_bef,batch_size=200,verbose=True)
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), proba,  real_labels)
        fpr, tpr, auc_thresholds = roc_curve(real_labels, ae_y_pred_prob[:,1])
        auc_score = auc(fpr, tpr)
        scipy.io.savemat('SDA_DNN',{'fpr':fpr,'tpr':tpr,'auc_score':auc_score})
        ## AUPR score add
        precision1, recall, pr_threshods = precision_recall_curve(real_labels, ae_y_pred_prob[:,1])
        aupr_score = auc(recall, precision1)
        print "SDADNN :",acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score
        all_performance_SDADNN.append([acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score])
        pdb.set_trace()
    print 'mean performance of deep autoencoder'
    print np.mean(np.array(all_performance), axis=0)
    print '---' * 50 
    print 'mean performance of deep autoencoder without fine tunning'
    print np.mean(np.array(all_performance_bef), axis=0)
    print '---' * 50 
    print 'mean performance of ADA using raw feature'
    print np.mean(np.array(all_performance_rf), axis=0)
    print '---' * 50    
    print 'mean performance of DNN using raw feature'
    print np.mean(np.array(all_performance_DNN), axis=0)
    print '---' * 50
    print 'mean performance of SDA DNN'
    print np.mean(np.array(all_performance_SDADNN), axis=0)
    #print 'mean performance of stacked ensembling'
    #print np.mean(np.array(all_performance_blend), axis=0)
    #print '---' * 50

    fileObject = open('resultListAUC_aupr_ADA5_inter2.txt', 'w')
    for i in all_performance:
        k=' '.join([str(j) for j in i])
	fileObject.write(k+"\n")
    fileObject.write('\n')
    for i in all_performance_bef:
        k=' '.join([str(j) for j in i])
        fileObject.write(k+"\n")
    fileObject.write('\n')
    for i in all_performance_rf:
        k=' '.join([str(j) for j in i])
        fileObject.write(k+"\n")
    fileObject.write('\n')
    for i in all_performance_DNN:
        k=' '.join([str(j) for j in i])
        fileObject.write(k+"\n")
    fileObject.write('\n')
    for i in all_performance_SDADNN:
        k=' '.join([str(j) for j in i])
        fileObject.write(k+"\n")
    #for i in all_performance_blend: 
    #    k=' '.join([str(j) for j in i])
    #    fileObject.write(k+"\n")

    fileObject.close()

def transfer_label_from_prob(proba):
    label = [1 if val>=0.9 else 0 for val in proba]
    return label


if __name__=="__main__":
    DeepInteract()
