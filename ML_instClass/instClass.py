# Music Tribe: Machine Learning Test
'''
tasks1 - data loading
- write a utility to load dataset into array of features X and one array of classification labels y 

task2 - classification
- train and evaluate a classifier. Consider classification methods and methods to evaluate them

task 3 - report
- How did the classifier perform, and how did you evaluate the classifier’s performance?
- What were the pros & cons of the classifier you chose? 
- Did you notice anything interesting about the dataset?
- What would be required to scale this up for use in a product?
'''

# Load JSON files from 
# c:\Users\phill\Dropbox (Personal)\db_Docs\jobs\18_musicTribe\ML_test\music_group_ml_test\music_group_ml_test_data\

import json

pathA_folder = r"c:\Users\phill\Dropbox (Personal)\db_Docs\jobs\18_musicTribe\ML_test\music_group_ml_test\music_group_ml_test_data"
pathR_json = r'\bass\ml_test_data_bass_1.json'
pathA_data = pathA_folder+pathR_json    # load one test json file for a reference
jjs0 = json.load(open(pathA_data)) # JSON~Jameson (R), John jameson Son Limited = jjs
#%%
import os

subDirs = [x for x in os.walk(pathA_folder)]
nInst = len(subDirs[0][1])  # number of possible outputs for classification
listInst = subDirs[0][1]    # names of outputs
listFeat = jjs0.get('featTypes')[0]
numFeat = len(listFeat)
print("Top directory contains {} instruments: \n{}".format(nInst, listInst))
print("-----------------------------------------------------------")
#%%
# initialize
valJsons = [[] for i in range(nInst)]
nsampleAll = [[] for i in range(nInst)]
fsAll,jjs = [], []
#centroidAll,energyAll,flatnessAll,fluxAll,gfccAll,lpcAll,mfccAll,specCompAll,zeroCrossAll = [],[],[],[],[],[],[],[],[]
X = [[] for i in range(numFeat)]
y,y_num = [],[]
for i,val in enumerate(subDirs[1:]): # iterate over subfolders (each instrument)
    for j in val[2]:    # iterate over file names in subfolder (~json files)
        jsonExist = 0;  # clear flag json file occurance (in case it is not in a folder)
        if j.endswith('.json'):     # proceed for all json files
            jsonExist = 1
            valJsons[i].append(j)   # 
            # save all json files into jss list of dictionaries
            pathA_json = os.path.join(val[0], j)
            jjsTmp = json.load(open(pathA_json))
            jjs.append(jjsTmp)
            # get number of samples in each json file
            nsampleTmp = jjsTmp.get('numSamples')[0]
            nsampleAll[i].append(nsampleTmp)
            ## get sampling frequency in each json file
            #fsAll.append(jjsTmp.get('sampleRate')[0]) # all the same

            # create feature vector X and corresponding label vector y
            # loop through # of samples in each json file
            for k in range(nsampleTmp):
                for l in range(numFeat):
                    X[l].append(jjsTmp.get('sample_'+str(k+1)).get(listFeat[l])[0])
#                    y[l].append(i)
#                    y_num[l].append(i)
                # centroidAll.append(jjsTmp.get('sample_'+str(k+1)).get('centroid')[0])
                # centroidAll.append(jjsTmp.get('sample_'+str(k+1)).get('centroid')[0])
                # energyAll.append(jjsTmp.get('sample_'+str(k+1)).get('energy')[0])
                # flatnessAll.append(jjsTmp.get('sample_'+str(k+1)).get('flatness')[0])
                # fluxAll.append(jjsTmp.get('sample_'+str(k+1)).get('flux')[0])
                # gfccAll.append(jjsTmp.get('sample_'+str(k+1)).get('gfcc')[0])
                # lpcAll.append(jjsTmp.get('sample_'+str(k+1)).get('lpc')[0])
                # mfccAll.append(jjsTmp.get('sample_'+str(k+1)).get('mfcc')[0])
                # specCompAll.append(jjsTmp.get('sample_'+str(k+1)).get('spectralComplexity')[0])
                # zeroCrossAll.append(jjsTmp.get('sample_'+str(k+1)).get('zeroCrossingRate')[0])
            

        
    if not jsonExist:
        print("Warning: folder {} does not contain a .json file".format(val[0]))
    print("Instrument \"{}\" contains {} json files and {} snippets ".format(listInst[i], len(valJsons[i]), sum(nsampleAll[i])))
print("Thus we have {} of training examples and {} features".format( sum(list(map(sum, nsampleAll))),numFeat ))
print("-----------------------------------------------------------")
print("Features types: {}".format(listFeat))
print("-----------------------------------------------------------")
#%%


#%% get data from directory test
#b = subDirs[1:]
#b = os.walk(pathA_folder)
#
#def list_files(dir):
#    r = []
#    for root, dirs, files in os.walk(dir):
#        for name in files:
#            r.append(os.path.join(root, name))
#    return r
#
#r = list_files(path_data)
#
#[print(x) for x in dataTree]

#%% analyse reference json file
## check start idexes
#[print(jjs0['sample_'+str(i)]['start']) for i in range(1,jjs0['numSamples'][0])]
#jjs0.get('numSamples')
## length of each sample
#[print( a.get('sample_'+str(i)).get('end')[0] - a.get('sample_'+str(i)).get('start')[0] ) for i in range(1,a.get('numSamples')[0]) ]
## or the same
#[print( a['sample_'+str(i)]['end'][0]-a['sample_'+str(i)]['start'][0] ) for i in range(1,a.get('numSamples')[0]) ]
#
#len(jjs0['sample_1'])
#len(jjs0['featTypes'][0])
#
## check features
#feat1 = []
#[feat1.append(x) for x in jjs0]
## or
#feat = list(jjs0)
#
#for featName, featVal in jjs0.items():
#    print(jjs0[featName])
#    