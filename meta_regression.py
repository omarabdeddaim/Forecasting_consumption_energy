# -*- coding: utf-8 -*-
"""
Created on Mon Dec  02 03:05:57 2019

@author: ABDEDDAIM OMAR & NOUNA ASSIA
 Studies : Engineering Student of mathematics and computer Sciences
 College : Faculty of Sciences and Technology 
 Project : Free_Lance for Asharea company
"""

import os, gc
from datetime import datetime
import numpy as np
import pandas as pd
import category_encoders 
from datetime import timedelta
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from mlxtend.regressor import StackingRegressor
from pandas.api.types import is_categorical_dtype
from pandas.api.types import is_datetime64_any_dtype as is_datetime

# Memory optimization

# Code reduce-memory-usage by Abdeddaim Omar

def reduce_of_memory(data, use_float16=False) -> pd.DataFrame:
    start_mem = data.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in data.columns:
        if is_datetime(data[col]) or is_categorical_dtype(data[col]):
            continue
        col_type = data[col].dtype

        if col_type != object:
            c_min = data[col].min()
            c_max = data[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    data[col] = data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    data[col] = data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data[col] = data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    data[col] = data[col].astype(np.int64)
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    data[col] = data[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
                else:
                    data[col] = data[col].astype(np.float64)
        else:
            data[col] = data[col].astype('category')

    end_mem = data.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.2f}%'.format(
        100 * (start_mem - end_mem) / start_mem))

    return data

# Import train data
    
   
train = pd.read_csv('Train.csv')

weather_train = pd.read_csv('weatherTrain.csv')

# Import metadata
metadata = pd.read_csv('BuildingsData.csv')


# Remove outliers in train data in case we want make some cleaning of data.
train = train[train['building_id'] != 1289]
train = train.query('not (building_id <= 76 & meter == 2 & timestamp <= "2015-11-03")')

# Function for weather data processing
def weather_filtrer_data(weather_data) -> pd.DataFrame:
    time_format = '%m/%d/%Y %H:%M:%S'
    
    start_date = datetime.strptime(weather_data['timestamp'].min(), time_format)
    end_date = datetime.strptime(weather_data['timestamp'].max(), time_format)
    total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)
    hours_list = [(end_date -timedelta(hours=x)).strftime(time_format) for x in range(total_hours)]
          #-----------Code Analysis point----------------
    for site_id in range(16):
        site_hours = np.array(weather_data[weather_data['site_id'] == site_id]['timestamp'])
        new_rows = pd.DataFrame(np.setdiff1d(hours_list, site_hours), columns=['timestamp'])
        new_rows['site_id'] = site_id
        weather_data = pd.concat([weather_data, new_rows], sort=True)
        weather_data = weather_data.reset_index(drop=True)           
​
    weather_data['datetime'] = pd.to_datetime(weather_data['timestamp'])
    weather_data['day'] = weather_data['datetime'].dt.day
    weather_data['week'] = weather_data['datetime'].dt.week
    weather_data['month'] = weather_data['datetime'].dt.month
​
    weather_data = weather_data.set_index(['site_id', 'day', 'month'])
​
    air_temperature_filler = pd.DataFrame(weather_data.groupby(['site_id','day','month'])['air_temperature'].mean(), columns=['air_temperature'])
    weather_data.update(air_temperature_filler, overwrite = False)
​
    cloud_coverage_filler = weather_data.groupby(['site_id', 'day', 'month'])['cloud_coverage'].mean()
    cloud_coverage_filler = pd.DataFrame(cloud_coverage_filler.fillna(method='ffill'), columns=['cloud_coverage'])
​
    weather_data.update(cloud_coverage_filler, overwrite=False)
​
    due_temperature_filler = pd.DataFrame(weather_data.groupby(['site_id','day','month'])['dew_temperature'].mean(), columns=['dew_temperature'])
    weather_data.update(due_temperature_filler, overwrite=False)
​
    sea_level_filler = weather_data.groupby(['site_id','day','month'])['sea_level_pressure'].mean()
    sea_level_filler = pd.DataFrame(sea_level_filler.fillna(method='ffill'), columns=['sea_level_pressure'])
​
    weather_data.update(sea_level_filler, overwrite=False)
​
    wind_direction_filler =  pd.DataFrame(weather_data.groupby(['site_id','day','month'])['wind_direction'].mean(), columns=['wind_direction'])
    weather_data.update(wind_direction_filler, overwrite=False)
​
    wind_speed_filler =  pd.DataFrame(weather_data.groupby(['site_id','day','month'])['wind_speed'].mean(), columns=['wind_speed'])
    weather_data.update(wind_speed_filler, overwrite=False)
​
    precip_depth_filler = weather_data.groupby(['site_id','day','month'])['precip_depth_1_hr'].mean()
    precip_depth_filler = pd.DataFrame(precip_depth_filler.fillna(method='ffill'), columns=['precip_depth_1_hr'])
​
    weather_data.update(precip_depth_filler, overwrite=False)
​
    weather_data = weather_data.reset_index()
    weather_data = weather_data.drop(['datetime','day','week','month'], axis=1)
​
    return weather_data

# Train weather data processing
weather_train = weather_filtrer_data(weather_train)




# Memory optimization
train = reduce_of_memory(train, use_float16=True)
weather_train = reduce_of_memory(weather_train, use_float16=True)
metadata = reduce_of_memory(metadata, use_float16=True)


# Merge train data 
train = train.merge(metadata, on='building_id', how='right')
train = train.merge(weather_train, on=['site_id', 'timestamp'], how='right')

del weather_train; gc.collect()

# Function for train and test data processing
def filter_data(data) -> pd.DataFrame:
    data.sort_values('timestamp')
    data.reset_index(drop=True)
    
    data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y/%m/%d %H:%M:%S')
    data['weekday'] = data['timestamp'].dt.weekday
    data['hour'] = data['timestamp'].dt.hour
    
    data['square_feet'] =  np.log1p(data['square_feet']) 
    
    data = data.drop(['timestamp', 'sea_level_pressure',
        'wind_direction', 'wind_speed', 'year_built', 'floor_count'], axis=1)
    
    gc.collect()
    
    encoder = LabelEncoder()
    data['primary_use'] = encoder.fit_transform(data['primary_use'])
    
    return data


# Train data processing
train = filter_data(train)


# Define target and predictors
target = np.log1p(train['meter_reading'])
features = train.drop(['meter_reading'], axis = 1) 

del train; gc.collect()



# Process categorical features
categorical_features = ['building_id', 'site_id', 'meter', 'primary_use']

encoder = category_encoders.CountEncoder(cols=categorical_features)
encoder.fit(features)
features = encoder.transform(features)

features_size = features.shape[0]
for feature in categorical_features:
    features[feature] = features[feature] / features_size
    


# Missing data imputation
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(features)
features = imputer.transform(features)

# ...........................Modal Training data.............................................

kfold = KFold(n_splits=3)

models = []

for idx, (train_idx, val_idx) in enumerate(kfold.split(features)):
    
    train_features, train_target = features[train_idx], target[train_idx]
    val_features, val_target = features[val_idx], target[val_idx]
    
    model = StackingRegressor(regressors=(LinearRegression(), LGBMRegressor()),
        meta_regressor=LGBMRegressor(), use_features_in_secondary=True)

    model.fit(np.array(train_features), np.array(train_target))
    models.append(model)

    print('RMSE: {:.4f} of fold: {}'.format(
        np.sqrt(mean_squared_error(val_target, model.predict(np.array(val_features)))), idx))

    del train_features, train_target, val_features, val_target; gc.collect()

del features, target; gc.collect()



#.......................Test data import and processing...........................

# Import test data
test = pd.read_csv(f'test.csv')
weather_test = pd.read_csv('weatherTest.csv')

row_ids = test['row_id']
test.drop('row_id', axis=1, inplace=True)

# Test weather data processing
weather_test = weather_filtrer_data(weather_test)

# Memory optimization
test = reduce_of_memory(test, use_float16=True)
weather_test = reduce_of_memory(weather_test, use_float16=True)

# Merge test data
test = test.merge(metadata, on='building_id', how='left')
test = test.merge(weather_test, on=['site_id', 'timestamp'], how='left')

del metadata; gc.collect()

# Test data processing
test = filter_data(test)

test = encoder.transform(test)
for feature in categorical_features:
    test[feature] = test[feature] / features_size

test = imputer.transform(test)



#..................Make predictions and create submission file........................

# Make predictions for all types of energy it depends on your wishes.
predictions = 1
for model in models:
    predictions += np.expm1(model.predict(np.array(test))) / len(models)
    del model; gc.collect()

del test, models; gc.collect()

# Create submission_All files
submission = pd.DataFrame({
    'row_id': row_ids,
    'meter_reading': np.clip(predictions, 0, a_max=None)
})
submission.to_csv('submission1.csv', index=False, float_format='%.4f')
