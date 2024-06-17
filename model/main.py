import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle

def create_model(dataOri,data):
    y = dataOri['class']
    
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    X_train, X_test, y_train, y_test = train_test_split(data,y,random_state=0,test_size=0.5)
    
    classifier=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.4, gamma=0.0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.25, max_delta_step=0, max_depth=5,
              min_child_weight=1, monotone_constraints='()',
              n_estimators=100, n_jobs=2, num_parallel_tree=1,
              objective='binary:logistic', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', use_label_encoder=True,
              validate_parameters=1, verbosity=None)
    
    classifier.fit(X_train,y_train)
    
    # test model
    y_pred=classifier.predict(X_test)
    print('Accuracy of our model : ',accuracy_score(y_test, y_pred))
    
    return classifier, scaler
    
    
    

def main():
    # load the numpy zip
    file_np = np.load('D:\HK2_2023_2024\Hoc May\Project\App_GUI_Chronic_Kidney_Disease\data\dataframe_chronic_kidney_disease.npz',allow_pickle=True)
    file_np.files
    
    fileO_np = np.load('D:\HK2_2023_2024\Hoc May\Project\App_GUI_Chronic_Kidney_Disease\data\dataOri_chronic_kidney_disease.npz',allow_pickle=True)
    fileO_np.files
    
    # get data 
    data = pd.DataFrame(file_np['arr_0'],columns=file_np['arr_1'])
    dataOri =  pd.DataFrame(fileO_np['arr_0'],columns=fileO_np['arr_1'])
    
    model, scaler = create_model(dataOri,data)
    
    with open('model/model.pkl', 'wb') as model_file, open('model/scaler.pkl', 'wb') as scaler_file:
        pickle.dump(model, model_file)
        pickle.dump(scaler, scaler_file)
    
if __name__ == '__main__':
    main()