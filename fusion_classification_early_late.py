# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 15:33:11 2023

@author: hlw69
"""


# https://github.com/braindecode/braindecode

import pandas as pd
import numpy as np
np.random.seed(42)
from random import randint
import random
#import seaborn as sns
import itertools
from operator import itemgetter
import mne

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score,StratifiedKFold, GridSearchCV, KFold
from sklearn.metrics import accuracy_score#, cross_val_predict
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, SelectPercentile, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from nilearn.decoding import Decoder
from nilearn.input_data import NiftiMasker
from nilearn.datasets import load_mni152_template
from nilearn.image import index_img, concat_imgs
from sklearn.metrics import classification_report,f1_score
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier 



""" PARAMS """
subjects = ['2'] # decode one subject or multisubject
TR = 2160 # ms
trial_num = 160 # should be constant across subjects and sessions
iterations = 10
session = 'combine' # can be 'combine', 'first', or 'second
binary = False #  8-class words, or binary social vs. number
check_mask = False # produce html with mask overlapped on beta image
mni_space = True # either normalised to mni space, or in native space
if mni_space == True:
    mask_strategy = 'whole-brain-template' # epi is not appropriate to use for beta images
elif mni_space == False:
    mask_strategy = 'background'   


nn = 150

"""  [1]   LOAD AND PROCESS fMRI DATA AND LABELS """
masker = NiftiMasker(mask_strategy=mask_strategy, smoothing_fwhm = 5)


def load_labels(subject,session):
    """
    

    Parameters
    ----------
    subject : string
        refers to the participant number.
    session : string
        refers to the session, will have the value of 1 or 2.

    Returns
    -------
    labels : list of strings
        all the labels for the trials in the correct order

    """
    # load the labels for beta files
    txtFile = 'X:\\CompSci\\ResearchProjects\\EJONeill\\Neuroimaging\\foteini\\matlabtiming\\beta_labels_subject' + subject +'_session'+session + '.txt'
    data = pd.read_fwf(txtFile,header=None)
    data.columns = ['label', 'beta_file']
    filtered =[]
    for i in data['label']:
        filtered.append("".join(filter(str.isalpha,i)))
    data['label'] = filtered
    labels = data['label']
    return labels

def load_fmri(subject, session):
    """
    loads up the beta images from the the file path as one volume
    extracts just the first 160 slices, as some ppts have an extra couple of images.
    

    Parameters
    ----------
    subject : string
        a value of 1,2,3 or 5.
    session : string
        a value of 1 or 2.

    Returns
    -------
    epi_img_data : Nifti data object
        DESCRIPTION.

    """
    # load fmri data
    print("loading up brain for subject and session ", subject, session)
    if mni_space == True:
        print("In mni space")
        fmripath = 'X:\\CompSci\\ResearchProjects\\EJONeill\\Neuroimaging\\foteini\\NFTII\\sub-0'+subject+'\\single_trial_sess'+session+'\\'
    elif mni_space == False:
        fmripath = 'X:\\CompSci\\ResearchProjects\\EJONeill\\Neuroimaging\\foteini\\NFTII\\sub-0'+subject+'\\single_trial_sess'+session+'_native\\'  
        print("In native space")
    #epi_img_data = nib.load(fmripath+'sub'+subject+'_sess'+session+'_betas_4D.nii')
    epi_img_data = index_img(fmripath+'sub'+subject+'_sess'+session+'_betas_4D.nii', slice(0, trial_num))
    print("shape of the fMRI data is ", epi_img_data.shape)
    return epi_img_data


def separate_or_combine_session(sessions, sess1_labels = None, sess1_fmri= None, sess2_labels= None,sess2_fmri= None ):
    if sessions =='combine':
        epi_img_data =concat_imgs([sess1_fmri,sess2_fmri ])
        labels = sess1_labels.append(sess2_labels).tolist()
    elif sessions == 'first':
        epi_img_data = sess1_fmri
        labels = sess1_labels
    elif sessions =='second':
        epi_img_data = sess2_fmri
        labels = sess2_labels
    return epi_img_data, labels


def multiple_subjects():
    if len(subjects) == 1:
        print("Working with individual subjects data")
        s1l = load_labels(subjects[0], '1')
        s1f = load_fmri(subjects[0], '1')
        s2l = load_labels(subjects[0], '2')
        s2f = load_fmri(subjects[0], '2') 
        epi, lab = separate_or_combine_session(sessions = session,sess1_labels=s1l,sess1_fmri=s1f, sess2_labels=s2l,sess2_fmri=s2f)  
        return epi, lab
        
    elif len(subjects) >1:
        print("combining multiple subjects ", subjects)
        list_labels  = []
        list_fmri = []
        for sub in subjects:
            print('subject', sub)
            s1l = load_labels(sub, '1')
            s1f = load_fmri(sub, '1')
            s2l = load_labels(sub, '2')
            s2f = load_fmri(sub, '2') 
            epi, lab = separate_or_combine_session(sessions = session,sess1_labels=s1l,sess1_fmri=s1f, sess2_labels=s2l,sess2_fmri=s2f)  
            list_fmri.append(epi)
            list_labels.append(lab)
        epi_img_data =concat_imgs(list_fmri)
        labels =  list(itertools.chain.from_iterable(list_labels))
        return epi_img_data, labels   
                           
epi_img_data, fmri_labels = multiple_subjects()
print("Shape of fMRI data ", epi_img_data.shape, len(fmri_labels))
# because some of the trials are incorrectly labelled as 'aughter'
for i in range(len(fmri_labels)):
     #   print(labels[i])
     if fmri_labels[i] =='aughter':
         fmri_labels[i] = 'daughter'
         
# can either categorise each of the 8 words separately, or split into the two semantic categories
def make_binary(labels):
    # split labels into binary
    for i in range(len(labels)):
     #   print(labels[i])
        if labels[i] in ['four','ten','six','three']:
         #   print("number")
            labels[i] = 'number'
        elif labels[i] in ['daughter', 'father', 'wife', 'child', 'aughter']:
            labels[i] = 'social'
    return labels
if binary == True:
    fmri_labels = make_binary(fmri_labels)

# run visualisation make sure that background mask on fMRI data looks as expected
def check_masking_strategy(mask_strategy):
    masker = NiftiMasker(mask_strategy=mask_strategy)
    masker.fit(epi_img_data)
    report = masker.generate_report()
    report.open_in_browser()

if check_mask == True:
    check_masking_strategy(mask_strategy)
#apply mask to the data
epi_img_data = masker.fit_transform(epi_img_data)
print("Shape of fMRI data after masking ", epi_img_data.shape) 
    
# sort the fmri data by labels so it can be matched with the EEG data
fmri_labels, epi_img_data = map(list, zip(*sorted(zip(fmri_labels, epi_img_data), reverse=True, key=itemgetter(0))))
print("fMRI labels are as follows ", fmri_labels)
    
    
    
    
"""  [2]   LOAD AND PROCESS EEG DATA AND LABELS """

subject = subjects[0] # iffy variable changes between eeg and fmri ahaha
path = 'X:\\CompSci\\ResearchProjects\\EJONeill\\Neuroimaging\\foteini\\EEG_Bimodal_V2\\IspeecAIproc\\EEG-proc\\epoched\\new\\'
eeg_path = path + 'subject0'+subject+'_session01_eeg-epo.fif.gz'
eeg_epochs = mne.read_epochs(eeg_path)
eeg_epochs = eeg_epochs.crop(tmin=-0.2, tmax=0.8)
eeg_events = mne.read_events(eeg_path) 
print("EEG labels are as follows ", eeg_epochs.event_id)
eeg_labels = eeg_epochs.events[:, -1].tolist()
eeg_data =  eeg_epochs.get_data()
print("Shape of EEG is, ", eeg_data.shape)
for i in range(len(eeg_labels)):
    our = [k for k in eeg_epochs.event_id if eeg_epochs.event_id[k] == eeg_labels[i]]
    eeg_labels[i] = our[0]    
if binary == True:
    eeg_labels = make_binary(eeg_labels)
    
# organise EEG data by labels so it can be mapped onto fMRI data   
eeg_labels, eeg_data = map(list, zip(*sorted(zip(eeg_labels, eeg_data), reverse=True, key=itemgetter(0))))
print(eeg_labels)
if subject == '5': # needs to be the same length as fMRI, but subject 5 has an extra eeg trial
    del eeg_data[139] #extra six trial
    del eeg_labels[139] #extra six trial
eeg_data = np.array(eeg_data)
epi_img_data = np.array(epi_img_data)
eeg_data = np.reshape(eeg_data, (eeg_data.shape[0], eeg_data.shape[1]*eeg_data.shape[2]))

""" [3]  Combine the eeg and fMRI data into bimodal """
print("The shape of eeg and fmri separately are ", eeg_data.shape, epi_img_data.shape)
bimodal_data = np.concatenate((eeg_data,epi_img_data), axis =1)
print("shape of bimodal data ", bimodal_data.shape)
print("Shared labels are ", eeg_labels)

# This check that EEG and fMRI have the same ordering of labels
a = set(eeg_labels)
b = set(fmri_labels)
if a == b:
    print("EEG and fMRI label are equal")
else:
    print("EEG and fMRI label are not equal")

def most_frequent(List):
    return max(set(List), key = List.count)

print(most_frequent(eeg_labels))

from sklearn.manifold import LocallyLinearEmbedding, TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#X_lle = embedding.fit_transform(X)
""" [4] Perform classification  """
import torchio as tio

transforms = (
    tio.RandomNoise,

)

transform = tio.Compose(transforms)
def array_transform(x):
    print(x[0])
    return np.array([transform(xi) for xi in x])

#from tio.transforms import (
   # ZNormalization,
   # RandomNoise,
   # )



# method to create random pairing between eeg and fMRI with shared labels
def augment_duplicate(data, labels, repeats = 8):
    new_data = np.repeat(data, repeats, axis=0)
  #  print(new_data[0][0])
  #  new_data = array_transform(new_data)
    new_data_df = pd.DataFrame(new_data)
    new_labels = np.repeat(labels, repeats, axis=0)
    new_labels_df = pd.DataFrame(new_labels)
    
    new_data_df['order'] = new_labels
    new_labels_df['order'] = new_labels
        
    new_labels_df = shuffle(new_labels_df)
    new_labels_df = new_labels_df.reset_index(drop=True)
    new_data_df = shuffle(new_data_df)
    new_data_df = new_data_df.reset_index(drop=True)
  #  print(new_data_df)
    
    # make into two data frames, shuffle, then sort them using same ordering
    new_data_df =  new_data_df.sort_values(by=['order'])
    new_labels_df= new_labels_df.sort_values(by=['order'])
    new_data_df = new_data_df.drop(columns=['order'])
    new_labels_df = new_labels_df.drop(columns=['order'])
    new_labels_df = new_labels_df.reset_index(drop=True)
    new_data_df = new_data_df.reset_index(drop=True)
    #print(new_data_df)

    
 #   new_data = shuffle(new_data, random_state=0)
 #   new_labels = shuffle(new_labels, random_state=5)
    
    
  #  new_labels, new_data = map(list, zip(*sorted(zip(new_labels, new_data), reverse=True, key=itemgetter(0))))
   # labels, data = map(list, zip(*sorted(zip(labels, data), reverse=True, key=itemgetter(0))))
  #  print("should be different ", np.array_equiv(new_data[0:39],data[0:39]))
  #  print("should be true ", np.array_equiv(new_labels[0:39],labels[0:39]))
  #  print(len(data))
  #  print(new_labels, labels)
  # convert to arrays
    return new_data_df.to_numpy(), new_labels_df.to_numpy().ravel()


""" LATE FUSION METHOD USED IN PAPER"""
def ensemble_approach(eeg, fmri, y, cv = 4):
    """
        Train two separate models for EEG and fMRI data. These output
        probability vectors which are then concatentated and fed into 
        a new joint model. The joint model is tested on unseen data.
        Manually implemented cross validation
        
    """
    overall_joint_score = []
    overall_eeg_score = []
    overall_fmri_score = [] 
    F1 = []
    
    # shuffle data once with random seed
    # take test and train splits based on indexes, use same for each model
    # get the sds, save the list of scores so can do a paired sample t-test
    
    fmri, eeg, y = shuffle(fmri, eeg, y, random_state = 42)
    kf = model_selection.StratifiedKFold(n_splits=cv, random_state=None)
    le =  LabelEncoder()
    y = le.fit_transform(y)
    for train_index, test_index in kf.split(eeg, y):
        scaler = preprocessing.Normalizer(norm='l2')
     #   print("Indexes are ", train_index, test_index)
        eegX_train, eegX_test = eeg[train_index], eeg[test_index]
        # L2 normalise the EEG data
     #   eegX_train = scaler.fit_transform(np.reshape(eegX_train,(eegX_train.shape[0],-1)))
     #   eegX_test = scaler.fit_transform(np.reshape(eegX_test,(eegX_test.shape[0],-1)))
        
        fmriX_train, fmriX_test = fmri[train_index], fmri[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
 #   for cross_val in range(cv):
      #  random_state = random.randint(0,50)
        # make pipelines
        eeg_feature_selection = SelectPercentile(f_classif, percentile=2)
        fmri_feature_selection = SelectPercentile(f_classif, percentile=1)
        fmri_extraction = LinearDiscriminantAnalysis(n_components=1)#TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)#LocallyLinearEmbedding(n_components=10)
        eeg_clf_percentile = make_pipeline(StandardScaler(),eeg_feature_selection, RandomForestClassifier(n_estimators=100,random_state=42)) # KNeighborsClassifier(n_neighbors=nn)
        fmri_clf_percentile = make_pipeline( StandardScaler(), fmri_feature_selection, SVC(probability=True, kernel = 'linear', C=0.1,random_state=42))
        joint_clf_percentile = make_pipeline(LinearSVC())
        
        # split the data
     #   fmriX_train, fmriX_test, eegX_train, eegX_test, y_train, y_test = train_test_split(epi_img_data, eeg_data, eeg_labels, test_size = 0.2, random_state=random_state, shuffle=True)
        print(fmriX_train.shape)
        # make extra combinations of the fmri and eeg data:
        # train *fMRI* and *EEG* models separately, then get predictions on test set
        eeg_model = eeg_clf_percentile.fit(eegX_train, y_train)
        eeg_scores = eeg_model.score(eegX_test, y_test)
        print("The eeg scores are ", eeg_scores)
        overall_eeg_score.append(eeg_scores)
        
        fmri_model = fmri_clf_percentile.fit(fmriX_train, y_train)
        fmri_scores = fmri_model.score(fmriX_test, y_test)
        overall_fmri_score.append(fmri_scores)
        print("The fmri scores are ", fmri_scores)
        
        train_eeg_prob_predictions = eeg_model.predict_proba(eegX_train)
        train_fmri_prob_predictions = fmri_model.predict_proba(fmriX_train)
        
        joint_prob_predictions_train= np.concatenate((train_eeg_prob_predictions, train_fmri_prob_predictions), axis=1)
        print(joint_prob_predictions_train.shape)
        joint_model = joint_clf_percentile.fit(joint_prob_predictions_train, y_train)
        
        
        ## test
        test_eeg_prob_predictions = eeg_model.predict_proba(eegX_test)
        test_fmri_prob_predictions = fmri_model.predict_proba(fmriX_test)
        joint_prob_predictions_test= np.concatenate((test_eeg_prob_predictions, test_fmri_prob_predictions), axis=1)
        scores = joint_model.score(joint_prob_predictions_test, y_test)
     #   F1.append(f1_score(joint_prob_predictions_test, y_test, average = 'macro'))
        
        print("The joint scores are ", scores)
        overall_joint_score.append(scores)
    print("For subject ",subjects[0],  ": The overall joint score is mean:", np.mean(overall_joint_score), ", sd: ",np.std(overall_joint_score), " The separate accuracies are: ", overall_joint_score)
    print("For subject ", subjects[0],": The overall eeg score is mean:", np.mean(overall_eeg_score), ", sd: ", np.std(overall_eeg_score), " The separate accuracies are: ", overall_eeg_score)
    print("For subject ",subjects[0], ": The overall fmri score is mean:", np.mean(overall_fmri_score),", sd: ", np.std(overall_fmri_score), " The separate accuracies are: ", overall_fmri_score)
   
    



# this method is used within the following grid search method
def early_fusion(eeg, fmri, y, kernel='linear', svc_c=0.1,augment = True, cv=4):
    accuracies = []
    F1 = []
    # takes top one percentile of discriminative features using ANOVA
    extraction = LocallyLinearEmbedding(n_components=10)
    feature_selection = SelectPercentile(f_classif, percentile=1)
    clf_percentile = make_pipeline( StandardScaler(), feature_selection, SVC(kernel = kernel, C=svc_c, random_state=42))
    le =  LabelEncoder()
    
    y = le.fit_transform(eeg_labels)
   # eeg, fmri,y = shuffle(eeg_data, epi_img_data, y, random_state=42)        
    # concatenate the data together
  #  joint_train = np.concatenate((eeg, fmri), axis=1)
    eeg, fmri, y, = shuffle(eeg, fmri, y, random_state = 42)
    # k fold train test split 
    kf = model_selection.StratifiedKFold(n_splits=cv, random_state=None)
    for train_index, test_index in kf.split(eeg, y):
        eeg_X_train, eeg_X_test = eeg[train_index], eeg[test_index]
        fmri_X_train, fmri_X_test = fmri[train_index], fmri[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Augment via random pairing of eeg and fMRI - this is done separately within test and train to avoid data leakage
        if augment == True:
            r = 15 #repetitions
            fmri_X_train,yi = augment_duplicate(fmri_X_train, y_train,repeats = r)
            fmri_X_test, yi = augment_duplicate(fmri_X_test, y_test,repeats = r)
            eeg_X_train,y_train = augment_duplicate(eeg_X_train, y_train,repeats = r)
            eeg_X_test, y_test = augment_duplicate(eeg_X_test, y_test,repeats = r)
            print("Shapes of augmented data are ", len(y_train), len(fmri_X_train))
            
        # fuse data together
        joint_train = np.concatenate((eeg_X_train, fmri_X_train), axis=1)
        joint_test = np.concatenate((eeg_X_test, fmri_X_test), axis=1)

        
        model = clf_percentile.fit(joint_train, y_train)
        test_predictions = model.predict(joint_test)
        F1.append(f1_score(test_predictions, y_test, average = 'macro'))
        scores = model.score(joint_test, y_test)
        print(scores)
        accuracies.append(scores)
    print("Subject ", subjects[0], ": the overall accuracy is ", np.mean(accuracies), np.std(accuracies), np.mean(F1))
    
    results = "\n" + "for kernel " + kernel + " and C " + str(svc_c) +  " the acc, std and f1 are :" +  str(np.mean(accuracies)) + " " + str(np.std(accuracies))+ " "+ str(np.mean(F1)) + "\n"
    print(results)
    print("Accuracies are ", accuracies)
    # save results
    text_file = open("X:\\CompSci\\ResearchProjects\\EJONeill\\Neuroimaging\\foteini\\"+subjects[0]+"resultsgridaugment.txt", "a")
  #  text_file.write(results)
    text_file.close()
    return np.mean(accuracies), np.std(accuracies), np.mean(F1)
    


""" EARLY FUSION METHOD USED IN PAPER"""
def grid_search_early_fusion(cv=4):
    svc_kernel = ['linear']#, 'rbf','poly']
    svc_c = [0.1, 0.5]#, 1, 5, 10]
    eeg, fmri, y, = shuffle(eeg_data, epi_img_data, eeg_labels, random_state = 42)
    for kernel in svc_kernel:
        for C in svc_c:
            print(kernel, " " ,C)
            acc, std, f1 = early_fusion(eeg, fmri, y,kernel = kernel, svc_c = C, augment=True)
            results = "\n" + "Grid search, for kernel " + kernel + " and C " + str(C) +  " the acc, std and f1 are :" +  str(acc) + " " + str(std)+ " "+ str(f1) + "\n"
            print(results)
            # save results
            text_file = open("X:\\CompSci\\ResearchProjects\\EJONeill\\Neuroimaging\\foteini\\"+subjects[0]+"resultsgridaugment_15.txt", "a")
            text_file.write(results)
            text_file.close()
            
            
 
            
#acc, std, f1 = early_fusion(eeg_data, epi_img_data, eeg_labels,kernel = 'linear', svc_c = 0.1, augment=True)
    
#grid_search_early_fusion(cv=10)
ensemble_approach(eeg_data, epi_img_data, eeg_labels,cv=4)



#acc, std, f1= manual_cross_val_concat()
# ignore this method as it couldn't be used alongside data augmentation
def grid_search_concat():
    
    
    bimodal_data = np.concatenate((eeg_data,epi_img_data), axis =1)
    print(bimodal_data.shape)
    fmriX_train, fmriX_test, eegX_train, eegX_test, y_train, y_test = train_test_split(epi_img_data, eeg_data, eeg_labels, test_size = 0.2, random_state=random_state, shuffle=True)
    print(fmriX_train.shape)
    
    
    
    augment = True
    if augment == True:
        fmriX_train,y = augment_duplicate(fmriX_train, y_train)
        fmriX_test, y = augment_duplicate(fmriX_test, y_test)
        eegX_train,y_train = augment_duplicate(eegX_train, y_train)
        eegX_test, y_test = augment_duplicate(eegX_test, y_test)
            
        y_train, eegX_train_redundant = map(list, zip(*sorted(zip(y_train, eegX_train), reverse=True, key=itemgetter(0))))
        y_test, eegX_test_redundant = map(list, zip(*sorted(zip(y_test, eegX_test), reverse=True, key=itemgetter(0))))
        print("Subject ", subjects[0], ". Shapes of augmented data are ", len(y_train), len(fmriX_train))
    X = bimodal_data
    y = eeg_labels
    print(X.shape, len(y))
    param_grid = {'svc__kernel': ('linear', 'rbf','poly') , 
                  'svc__C':[5, 10, 50, 100]}
    classifier =  LinearSVC()#LogisticRegression()#LinearSVC()
    feature_selection = SelectPercentile(f_classif, percentile=5)
    #clf = Pipeline([
     # ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
     # ('classification', SVC(C=1, kernel='linear'))
    #])
     # refers to the most significant features
    clf_selected = make_pipeline(SelectKBest(f_classif, k=6), StandardScaler(),  LinearSVC())
    # f_classif is ANOVA , another option is mutual_info_classif
    clf_percentile = make_pipeline(feature_selection, StandardScaler(),  SVC())
    
    
    #print(clf_percentile.get_params())
    
    
    
    param_grid = {'svc__kernel': ('linear', 'rbf','poly') , 
                  'svc__C':[5, 10, 50, 100]}
                 # 'svc__gamma': [1,0.1,0.01,0.001]}
                #  'svc__degree' : [1,2,3,4,5,6]}
    
    
    #SVC().get_params.keys()
    #pca = PCA()
    # df = pd.DataFrame(est.cv_results_
    #grid_search = GridSearchCV(estimator = clf_percentile, param_grid = param_grid, n_jobs=2, verbose = 3)
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
    #X_train, X_test, y_train, y_test = train_test_split(epi_img_data, labels, test_size=0.1, random_state=0)
    
    # nested cross validation
    grid_search = GridSearchCV(estimator = clf_percentile, param_grid = param_grid, n_jobs=2, verbose = 3,  scoring='accuracy',error_score='raise')
    scores = cross_val_score(grid_search, X, y, scoring='accuracy', cv=cv_outer, n_jobs=-1)
    # report performance
    print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    print(scores)


