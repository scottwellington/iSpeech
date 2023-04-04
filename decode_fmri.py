# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 14:05:27 2023

@author: hlw69
"""


import pandas as pd
import nibabel as nib
import numpy as np
from random import randint
import seaborn as sns
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score,StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, SelectPercentile, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from nilearn.decoding import Decoder
from nilearn.input_data import NiftiMasker
from nilearn.datasets import load_mni152_template
from nilearn.image import index_img, concat_imgs
import itertools



subjects = ['1'] # decode one subject or multisubject
TR = 2160 # ms
trial_num = 160 # should be constant across subjects and sessions

iterations = 30
session = 'second' # can be 'combine', 'first', or 'second
binary = False #  8-class words, or binary social vs. number
check_mask = True # produce html with mask overlapped on beta image
mni_space = False # either normalised to mni space, or in native space
if mni_space == True:
    mask_strategy = 'whole-brain-template' # epi is not appropriate to use for beta images
elif mni_space == False:
    mask_strategy = 'background'
    
    

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
        if session == 'first':
            s1l = load_labels(subjects[0], '1')
            s1f = load_fmri(subjects[0], '1')
            epi, lab = separate_or_combine_session(sessions = session,sess1_labels=s1l,sess1_fmri=s1f, sess2_labels=None,sess2_fmri=None)  
        elif session =='second':
            s2l = load_labels(subjects[0], '2')
            s2f = load_fmri(subjects[0], '2') 
            epi, lab = separate_or_combine_session(sessions = session,sess1_labels=None,sess1_fmri=None, sess2_labels=s2l,sess2_fmri=s2f)  
        elif session =='combine':
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
            
                  
epi_img_data, labels = multiple_subjects()
#print(len(labels), epi_img_data,shape)

#labels = sess1_labels.append(sess2_labels).tolist()
#epi_img_data, labels = separate_or_combine_session(sessions = session,sess1_labels=s1l,sess1_fmri=s1f, sess2_labels=s2l,sess2_fmri=s2f)
print("Shapes ", epi_img_data.shape, len(labels))

# because some of the trials are incorrectly labelled as 'aughter'
for i in range(len(labels)):
     #   print(labels[i])
     if labels[i] =='aughter':
         labels[i] = 'daughter'
         


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
    labels = make_binary(labels)
#print(labels)
    
y = np.array(labels)
#print(" labels are " ,y.shape, y)

def check_masking_strategy(mask_strategy):
    masker = NiftiMasker(mask_strategy=mask_strategy)
    masker.fit(epi_img_data)
    report = masker.generate_report()
    report.open_in_browser()

if check_mask == True:
    check_masking_strategy(mask_strategy)
    

#folder_path = fmripath
#beta_files = [file.path for file in os.scandir(folder_path) if file.name.endswith('.nii')]
#print(beta_files)
#beta_images = [nib.load(file) for file in beta_files]
#print(beta_images)
#X = concatenate([image.get_data()[np.newaxis] for image in beta_images])

#print("Shape X ,", X.shape)



masker = NiftiMasker(mask_img=None,  smoothing_fwhm=5, standardize=False, mask_strategy=mask_strategy, verbose=0, reports=True)
#scaler = StandardScaler()
classifier =  LinearSVC()#LogisticRegression()#LinearSVC()
#clf = Pipeline([
 # ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
 # ('classification', SVC(C=1, kernel='linear'))
#])
 # refers to the most significant features
clf_selected = make_pipeline(SelectKBest(f_classif, k=6), StandardScaler(),  LinearSVC())
# f_classif is ANOVA , another option is mutual_info_classif
clf_percentile = make_pipeline(SelectPercentile(f_classif, percentile=2), StandardScaler(),  classifier)


import matplotlib as plt
from sklearn import svm, datasets


def basic_pipeline(data, clf,  iterations = 10):
    """ Uses a masker object to smooth the data and compute a whole-brain template.
        No standardisation is done until after the train test split to protect against data leakeage
    """
    data = masker.fit_transform(data)
    print("Shape after masking ", data.shape)
    X = data
   # data = data.get_fdata()
    
    # swap axes
 #   print(data.shape)
  #  data = np.swapaxes(data, 0, 3)
   # print(data.shape)
    # reshape to 2D
 #   X = np.reshape(data,(data.shape[0], data.shape[1]* data.shape[2]* data.shape[3]))
    # standardise   
  #  scaler = StandardScaler()
  #  X = scaler.fit_transform(X)
    averages = []
    for iter in range(iterations):
        print("Iteration number : ", iter)
        cv = StratifiedKFold(n_splits=5, shuffle = True, random_state=randint(0,50))        # Setting random_state is not necessary here
        scores = cross_val_score(clf, X, y, cv=cv)
        print(scores)
        averages.append(np.mean(scores))
    
        # create a mesh to plot in
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        h = (x_max / x_min)/100
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h))
        plt.subplot(1, 1, 1)
        Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.xlim(xx.min(), xx.max())
        plt.title(title)
        plt.show()
        
        
        # Split the data into training and test sets

      #  X_train = scaler.fit_transform(X_train)
       # X_test = scaler.fit_transform(X_test)
        # Train a support vector machine classifier#
      #  clf = SVC(C=1, kernel='linear')
       # clf = LogisticRegression(solver='lbfgs')
       
       
       # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randint(0, 60))
       # clf.fit(X_train, y_train)
        
       # # Make predictions on the test set
       # y_pred = clf.predict(X_test)
        
       # # Print the accuracy
       # acc = accuracy_score(y_test, y_pred)
       # print("Accuracy: {:.2f}%".format(acc * 100))
    print("Averages are :", np.mean(averages), " and the rest ", averages)
    return clf

"""
X= epi_img_data


from sklearn.model_selection import LeaveOneGroupOut
cv = LeaveOneGroupOut()

decoder = Decoder(estimator='svc', standardize=True,
                  screening_percentile=5, scoring='accuracy')


# Compute the prediction accuracy for the different folds (i.e. session)
decoder.fit(X, y)
y_pred = decoder.predict(X)
print(y_pred)

# Print the CV scores
#print(decoder.cv_scores_['face'])




"""

    
# try with GM template
    
' \param_validation.py:197: UserWarning: Brain mask is smaller than .5% of the volume human brain. This object is probably not tuned tobe used on such data.'
def select_background_img(choice = 'mni'):
    " choice can be mni, native, or anat"
    if choice == 'mni':
        img = load_mni152_template(resolution=None)
    elif choice =='native':
        # load up sample  functional nifti image
        functionalpath = 'X:\\CompSci\\ResearchProjects\\EJONeill\\Neuroimaging\\foteini\\NFTII\\sub-0'+subject+'\\Session'+session+'\\CMRR\\' + 'auCMRR_sub0'+subject+'_sess0'+session+'.nii'
        img = nib.load(functionalpath)
        img = index_img(functionalpath, slice(0, 1))
    elif choice == 'anat':
        anat_path = 'X:\\CompSci\\ResearchProjects\\EJONeill\\Neuroimaging\\foteini\\NFTII\\sub-03\\anat\\'
        anat_path = anat_path  + 'T_sub0'+subject+'.nii'
        img=  nib.load(anat_path)
    print('shape of '+choice+' space is ',img.shape)
    return img
        

def anova_masked_pipeline(data):
    X = data #svc # logistic
    decoder = Decoder(estimator='svc',  standardize=True,smoothing_fwhm = 5, mask_strategy = 'whole-brain-template',
                  screening_percentile=2, scoring='accuracy', cv=10)
# Compute the prediction accuracy for the different folds (i.e. session)
    decoder.fit(X, y)
    decoder_scores = pd.DataFrame(decoder.cv_scores_)
    decoder_scores['Average'] = decoder_scores.mean(axis=1)
    decoder_scores = decoder_scores.reset_index(drop=True)
    sns.barplot(x='variable', y='value', data=decoder_scores.melt())
    print(decoder.cv_scores_)
    print( "Averageis ", decoder_scores['Average'])
    return decoder
#choice = 'mni'  # can be native, mni or anat
#bac_img = select_background_img(choice=choice)
#dec = anova_masked_pipeline(epi_img_data)
dec = basic_pipeline(epi_img_data, clf_percentile, iterations = iterations)


' Plots '
unique_labels = list(set(labels))
print(unique_labels)

for uniq in unique_labels:
    weight_img = dec.coef_img_[uniq]
    from nilearn.plotting import plot_stat_map, show
    plot_stat_map(weight_img, bg_img=bac_img, title=subject+'_'+session+' SVM weights '+choice +'_' + uniq )
    show()



"""




print(data)



timingspath = 'X:\\CompSci\\ResearchProjects\\EJONeill\\Neuroimaging\\foteini\\subjectsTimings\\'
fmripath = 'X:\\CompSci\\ResearchProjects\\EJONeill\\Neuroimaging\\foteini\\NFTII\\sub-0'+subject+'\\Session'+session+'\\CMRR\\'




#df=pd.read_excel('D://foteinis_study//subjectsTimings//InnerSpeech-fMRI-0014-0'+subject+'-'+session+'-export.xlsx')
df = pd.read_excel(timingspath +'InnerSpeech-fMRI-0014-0'+subject+'-'+session+'-export.xlsx')
# NFTII//NFTII//sub-05//InnerSpeech-fMRI-0014-05-1-export.xlsx')
#for col in df.columns:
#    print(col)
print(min(df['stimulus.OnsetTime'] - df['rest.OnsetTime']))


# keep only relevant columns
data = df[['InnerWord','Condition','stimulus.OnsetTime', 'rest.OnsetTime','fixation.OnsetTime']]
print(data)

# adapt onset times to account for the trigger
data['stimulus.OnsetTime'] -=  data['fixation.OnsetTime']
data['rest.OnsetTime'] -=  data['fixation.OnsetTime']
data['stimulus.Duration'] = data['rest.OnsetTime'] - data['stimulus.OnsetTime']


#print(conditionVersion)
TR_Version = data#conditionVersion
#print(TR_Version)
TR_Version['stimulus.OnsetTime'] /= TR
TR_Version['stimulus.Duration'] /= TR
TR_Version['stimulus.OnsetTime'] = round(TR_Version['stimulus.OnsetTime'])
TR_Version['stimulus.Duration'] = round(TR_Version['stimulus.Duration'])



# convole with HRF to compensate for BOLD lag - 
# Create an example HRF - this function is from brainiak
hrf = hrf_func(temporal_resolution=1)

# Plot the canonical double gamma HRF
f, ax = plt.subplots(1,1, figsize = (10, 5))
ax.plot(range(30), hrf)

ax.set_title("Hemodynamic Response Function (HRF)")
ax.set_xlabel('Time in secs')







#load up all brain volumes
brainpath = fmripath + "auCMRR_sub05_sess01.nii"
print("loading up brain")

epi_img = nib.load(brainpath)
epi_img_data = epi_img.get_fdata()
print("Shappe should be 4 dimensional ", epi_img_data.shape) # should be four dimensions, last dimension should be time ; 





# iterate through each row of df to find corresponding brain volumes, make into a list of lists
chunked_fmri = []
for index, row in TR_Version.iterrows():
    first_vol = int(row['stimulus.OnsetTime']) # get first volume for chunk
    dur = int(row['stimulus.Duration'])
    print(first_vol, first_vol+dur)
    chunk = epi_img_data[first_vol:first_vol+dur]
    print("shape of chunk is ", chunk.shape)
    chunked_fmri.append(chunk)
    
TR_Version['data'] = chunked_fmri


from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.2)
X_train = train['data']
Y_train = train['Condition']
x_test = test['data']
y_test = test['Condition']


from sklearn.model_selection import cross_val_predict
from sklearn import svm
clf = svm.SVC(kernel='linear', C=3)
cvs=cross_val_score(clf,x,y,scoring='accuracy',cv=10)
print('cross_val_scores=  ',cvs.mean())
y_pred=cross_val_predict(clf,x,y,cv=1)
conf_mat=confusion_matrix(y_pred,y)
conf_mat
"""