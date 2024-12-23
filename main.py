## Code to Read Payload from user request

# Import necessary libraries
import os
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, mean_squared_error, roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression

# Payload Filename
payloadNm='algoparams_from_us'

# Read JSON payload
with open(payloadNm+'.json','r') as file:
    paramsData=json.load(file)

# Extracting the data from the payload
# Read session name and IDs
sessionName=paramsData['session_name']
sessionDesp=paramsData['session_description']
projID=paramsData['design_state_data']['session_info']['project_id']
expID=paramsData['design_state_data']['session_info']['experiment_id']
# Read the dataset
filenm=paramsData['design_state_data']['session_info']['dataset']
data = pd.read_csv(filenm)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
# Target and model type
targetColNm=paramsData['design_state_data']['target']['target']
modelTyp=paramsData['design_state_data']['target']['type']

# Feature handling and Preprocessing
featureCol_in_handler=list(paramsData['design_state_data']['feature_handling'].keys())
featureIdx=[paramsData['design_state_data']['feature_handling'][feat]['is_selected'] for feat in featureCol_in_handler]

for feat,isselected in zip(featureCol_in_handler,featureIdx):
    if isselected:
        if paramsData['design_state_data']['feature_handling'][feat]['feature_variable_type']=="numerical":
            if paramsData['design_state_data']['feature_handling'][feat]['feature_details']['impute_with']=="Average of values" :
                data[feat]=data[feat].fillna(data[feat].mean())
            else:
                data[feat]=data[feat].fillna(value=paramsData['design_state_data']['feature_handling'][feat]['feature_details']['impute_value'])
featureData=data.loc[:,featureIdx]

if modelTyp=="regression":
    targetData=featureData.loc[:,targetColNm].to_numpy()
elif modelTyp=="classification":
    targetData=featureData.loc[:,targetColNm].astype('category').astype('category')
featureData=featureData.drop(targetColNm,axis=1)

# Feature Reduction
whichfeatureReduction=paramsData['design_state_data']['feature_reduction']['feature_reduction_method']
featureReductionparams=paramsData['design_state_data']['feature_reduction'][whichfeatureReduction]
if whichfeatureReduction=="Principal Component Analysis":
    pca=PCA(n_components=featureReductionparams['num_of_features_to_keep'])
    reductionfeatureData=pca.fit_transform(featureData)
elif whichfeatureReduction=="No Reduction":
    reductionfeatureData=featureData.to_numpy()
    
# Model Selection and Hyperparameter tuning
hyperParameters=paramsData['design_state_data']['hyperparameters']
models=list(paramsData['design_state_data']['algorithms'].keys())
selectedModels=[mdl for mdl in models if paramsData['design_state_data']['algorithms'][mdl]['is_selected']==True]

# Update results function
def updateResults(finalresults,results,sessionName,sessionDesp,projID,expID):
    if isinstance(finalresults.get('Model Results'),dict):
        finalresults['Model Results'].update(results)
    else:
        finalresults={'Session Name':sessionName,'Session Description':sessionDesp,'Project ID':projID,'Experiment ID':expID,'Model Results':results}
    return finalresults

finalresults={}
for uniModel in selectedModels:
    if modelTyp=="regression":
        match uniModel:
            case 'RandomForestRegressor':
                modelParams=paramsData['design_state_data']['algorithms'][uniModel]
                model = RandomForestRegressor(min_samples_leaf=modelParams['min_trees'])    
                scoreMSE = make_scorer(mean_squared_error)
                modelTuned=GridSearchCV(estimator=model,param_grid={'max_features':[1,2,3]},scoring=scoreMSE,cv=TimeSeriesSplit(n_splits=hyperParameters['num_of_folds']),n_jobs=hyperParameters['parallelism'])
                modelTuned.fit(reductionfeatureData,targetData)
                results={'best_score':modelTuned.best_score_,'best_params':modelTuned.best_params_,'cv_results':modelTuned.cv_results_}
                results={modelParams['model_name']:results}
                finalresults=updateResults(finalresults,results,sessionName,sessionDesp,projID,expID)
            case 'GBTRegressor':
                modelParams=paramsData['design_state_data']['algorithms'][uniModel]
                model = GradientBoostingRegressor(max_depth=modelParams['max_depth'])    
                scoreMSE = make_scorer(mean_squared_error)
                modelTuned=GridSearchCV(estimator=model,param_grid={'max_features':[1,2,3]},scoring=scoreMSE,cv=TimeSeriesSplit(n_splits=hyperParameters['num_of_folds']),n_jobs=hyperParameters['parallelism'])
                modelTuned.fit(reductionfeatureData,targetData)
                results={'best_score':modelTuned.best_score_,'best_params':modelTuned.best_params_,'cv_results':modelTuned.cv_results_}
                results={modelParams['model_name']:results}
                finalresults=updateResults(finalresults,results,sessionName,sessionDesp,projID,expID)
            case 'extra_random_trees':
                modelParams=paramsData['design_state_data']['algorithms'][uniModel]
                model = ExtraTreesRegressor()    
                scoreMSE = make_scorer(mean_squared_error)
                modelTuned=GridSearchCV(estimator=model,param_grid={'max_features':[1,2,3]},scoring=scoreMSE,cv=TimeSeriesSplit(n_splits=hyperParameters['num_of_folds']),n_jobs=hyperParameters['parallelism'])
                modelTuned.fit(reductionfeatureData,targetData)
                results={'best_score':modelTuned.best_score_,'best_params':modelTuned.best_params_,'cv_results':modelTuned.cv_results_}
                results={modelParams['model_name']:results}
                finalresults=updateResults(finalresults,results,sessionName,sessionDesp,projID,expID)
            case 'LinearRegression':
                modelParams=paramsData['design_state_data']['algorithms'][uniModel]
                model = LinearRegression()    
                scoreMSE = make_scorer(mean_squared_error)
                modelTuned=model
                modelTuned.fit(reductionfeatureData,targetData)
                results={'best_score':modelTuned.score(reductionfeatureData,targetData),'best_params':modelTuned.coef_}
                results={modelParams['model_name']:results}
                finalresults=updateResults(finalresults,results,sessionName,sessionDesp,projID,expID)
            case 'RidgeRegression':
                modelParams=paramsData['design_state_data']['algorithms'][uniModel]
                model = Ridge()    
                scoreMSE = make_scorer(mean_squared_error)
                modelTuned=GridSearchCV(estimator=model,param_grid={'alpha':[1,2,3]},scoring=scoreMSE,cv=TimeSeriesSplit(n_splits=hyperParameters['num_of_folds']),n_jobs=hyperParameters['parallelism'])
                modelTuned.fit(reductionfeatureData,targetData)
                results={'best_score':modelTuned.best_score_,'best_params':modelTuned.best_params_,'cv_results':modelTuned.cv_results_}
                results={modelParams['model_name']:results}
                finalresults=updateResults(finalresults,results,sessionName,sessionDesp,projID,expID)
            case 'LassoRegression':
                modelParams=paramsData['design_state_data']['algorithms'][uniModel]
                model = Lasso()    
                scoreMSE = make_scorer(mean_squared_error)
                modelTuned=GridSearchCV(estimator=model,param_grid={'alpha':[1,2,3]},scoring=scoreMSE,cv=TimeSeriesSplit(n_splits=hyperParameters['num_of_folds']),n_jobs=hyperParameters['parallelism'])
                modelTuned.fit(reductionfeatureData,targetData)
                results={'best_score':modelTuned.best_score_,'best_params':modelTuned.best_params_,'cv_results':modelTuned.cv_results_}
                results={modelParams['model_name']:results}
                finalresults=updateResults(finalresults,results,sessionName,sessionDesp,projID,expID)
            case 'ElasticNetRegression':
                modelParams=paramsData['design_state_data']['algorithms'][uniModel]
                model = ElasticNet()    
                scoreMSE = make_scorer(mean_squared_error)
                modelTuned=GridSearchCV(estimator=model,param_grid={'alpha':[1,2,3]},scoring=scoreMSE,cv=TimeSeriesSplit(n_splits=hyperParameters['num_of_folds']),n_jobs=hyperParameters['parallelism'])
                modelTuned.fit(reductionfeatureData,targetData)
                results={'best_score':modelTuned.best_score_,'best_params':modelTuned.best_params_,'cv_results':modelTuned.cv_results_}
                results={modelParams['model_name']:results}
                finalresults=updateResults(finalresults,results,sessionName,sessionDesp,projID,expID)
                   
    elif modelTyp=="classification":
        match uniModel:
            case 'RandomForestClassifier':
                modelParams=paramsData['design_state_data']['algorithms'][uniModel]
                model = RandomForestClassifier(min_samples_leaf=modelParams['min_samples_per_leaf_min_value'])    
                modelTuned=GridSearchCV(estimator=model,param_grid={'n_estimators': [50, 100]},scoring='accuracy',cv=TimeSeriesSplit(n_splits=hyperParameters['num_of_folds']),n_jobs=hyperParameters['parallelism'])
                modelTuned.fit(reductionfeatureData,targetData)
                results={'best_score':modelTuned.best_score_,'best_params':modelTuned.best_params_,'cv_results':modelTuned.cv_results_}
                results={modelParams['model_name']:results}
                finalresults=updateResults(finalresults,results,sessionName,sessionDesp,projID,expID)
            case 'GBTClassifier':
                modelParams=paramsData['design_state_data']['algorithms'][uniModel]
                model = GradientBoostingClassifier()    
                modelTuned=GridSearchCV(estimator=model,param_grid={'n_estimators': [50, 100]},scoring='accuracy',cv=TimeSeriesSplit(n_splits=hyperParameters['num_of_folds']),n_jobs=hyperParameters['parallelism'])
                modelTuned.fit(reductionfeatureData,targetData)
                results={'best_score':modelTuned.best_score_,'best_params':modelTuned.best_params_,'cv_results':modelTuned.cv_results_}
                results={modelParams['model_name']:results}
                finalresults=updateResults(finalresults,results,sessionName,sessionDesp,projID,expID)
            case 'LogisticRegression':
                modelParams=paramsData['design_state_data']['algorithms'][uniModel]
                model = LogisticRegression(n_jobs=hyperParameters['parallelism'])    
                modelTuned=model
                modelTuned.fit(reductionfeatureData,targetData)
                results={'best_score':accuracy_score(targetData,modelTuned.predict(reductionfeatureData)),'best_params':modelTuned.coef_}
                results={modelParams['model_name']:results} 
                finalresults=updateResults(finalresults,results,sessionName,sessionDesp,projID,expID)

# Create Response JSON
# Encode to convert numpy arrays to list
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
# Write File to Json
with open(payloadNm+'_output.json',"w") as file:
    json.dump(finalresults,file,cls=NumpyEncoder)
