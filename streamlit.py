#import libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
import tensorflow
from tensorflow.keras.models import load_model
#write heading
st.title("Machine Learning and Deep Learning Based Nitrate Leaching Prediction in Irrigated Agriculture")
#slider to select numeric columns
# soil_pH = st.slider('Select the soil pH', 0, 14)
# soc_per = st.slider('Select the soil organic carbon (%)', 0, 100)
# som_per = st.slider('Select the soil organic matter (%)', 0, 100)
# soil_total_N = st.slider('Select the soil total N', 0, 100)
# mean_rainfal_mm = st.slider('Select the mean rainfall (mm)', 0, 1000)
# measurement_depth_m = st.slider('Select the measurement depth (m)', 0, 10)
# fertilizer_rate_kg = st.slider('Select the fertilizer rate (kg N/ha)', 0, 100)

#enter numerical values
soil_pH = st.number_input('Enter the soil pH 3.9 ~ 8.5')
soc_per = st.number_input('Enter the soil organic carbon (%) 0 ~ 13.5')
som_per = st.number_input('Enter the soil organic matter (%) 0 ~ 23.22')
soil_total_N = st.number_input('Enter the soil total N 0 ~ 0.95')
mean_rainfal_mm = st.number_input('Enter the mean rainfall (mm) 170 ~ 2500')
measurement_depth_m = st.number_input('Enter the measurement depth (m) 0.25 ~ 2.40')
fertilizer_rate_kg = st.number_input('Enter the fertilizer rate (kg N/ha) 0 ~ 1800')



#select the categorical columns
crop_type = st.selectbox('Select the crop type', ('Cereals', 'Others', 'Grasses', 'Vegetables', 'Legumes'))
#convert categorcal input to numeric
if crop_type == 'Cereals':
    crop_type = 0
elif crop_type == 'Others':
    crop_type = 1
elif crop_type == 'Grasses':
    crop_type = 2
elif crop_type == 'Vegetables':
    crop_type = 3
elif crop_type == 'Legumes':
    crop_type = 4

experiment_type = st.selectbox('Select the experiment type', ('Field', 'Pot'))

#convert categorcal input to numeric
if experiment_type == 'Field':
    experiment_type = 0
elif experiment_type == 'Pot':
    experiment_type = 1

ph_class = st.selectbox('Select the pH class', ('Acidic', 'Neutral', 'Alkaline'))
#convert categorcal input to numeric
if ph_class == 'Acidic':
    ph_class = 0
elif ph_class == 'Neutral':
    ph_class = 1
elif ph_class == 'Alkaline':
    ph_class = 2
soil_texture = st.selectbox('Select the soil texture', ('Coarse', 'Unknown', 'Fine', 'Medium'))
#convert categorcal input to numeric
if soil_texture == 'Coarse':
    soil_texture = 0
elif soil_texture == 'Unknown':
    soil_texture = 1
elif soil_texture == 'Fine':
    soil_texture = 2
elif soil_texture == 'Medium':
    soil_texture = 3
fertiisation_method = st.selectbox('Select the fertiisation method', ('Split dose', 'Full dose', 'Others'))
#convert categorcal input to numeric
if fertiisation_method == 'Split dose':
    fertiisation_method = 0
elif fertiisation_method == 'Full dose':
    fertiisation_method = 1
elif fertiisation_method == 'Others':
    fertiisation_method = 2
fertilizer_name = st.selectbox('Select the fertilizer_name',('CAN', 'Urea', 'Manure', 'Others', 'Complex', 'Cattle slurry', 'Ammonium based',
 'Compost', 'Biogas slurry', 'Pig slurry', 'Animal urine'))

#convert categorcal input to numeric
if fertilizer_name == 'CAN':
    fertilizer_name = 0
elif fertilizer_name == 'Urea':
    fertilizer_name = 1
elif fertilizer_name == 'Manure':
    fertilizer_name = 2
elif fertilizer_name == 'Others':
    fertilizer_name = 3
elif fertilizer_name == 'Complex':
    fertilizer_name = 4
elif fertilizer_name == 'Cattle slurry':

    fertilizer_name = 5
elif fertilizer_name == 'Ammonium based':
    fertilizer_name = 6
elif fertilizer_name == 'Compost':

    fertilizer_name = 7
elif fertilizer_name == 'Biogas slurry':
    fertilizer_name = 8
elif fertilizer_name == 'Pig slurry':
    fertilizer_name = 9
elif fertilizer_name == 'Animal urine':
    fertilizer_name = 10

elif fertilizer_name == 'Biogas slurry':
    fertilizer_name = 8
elif fertilizer_name == 'Pig slurry':
    fertilizer_name = 9

measuring_method = st.selectbox('Select the measuring method', ('Porous cups', 'Lysimeter', 'Others'))
#convert categorcal input to numeric
if measuring_method == 'Porous cups':
    measuring_method = 0
elif measuring_method == 'Lysimeter':
    measuring_method = 1
elif measuring_method == 'Others':
    measuring_method = 2

fertilizer_type = st.selectbox('Select the fertilizer type', ('Synthetic', 'Organic'))
#convert categorcal input to numeric
if fertilizer_type == 'Synthetic':
    fertilizer_type = 0
elif fertilizer_type == 'Organic':
    fertilizer_type = 1

#side paned to select the model
st.sidebar.title("Select the model")
model = st.sidebar.selectbox('Select models', ('Bayesian Ridge','Linear Regression','Extra Tree Regrssor','Random Forest','Neural Network'))
if model == 'Bayesian Ridge':
    #select pickle file
    model = pickle.load(open('bayesian_ridge.pkl','rb'))
elif model == 'Linear Regression':
    #select pickle file
    model = pickle.load(open('RandomForestRegressor().pkl','rb'))
elif model == 'Random Forest':
    #select pickle file
    model = pickle.load(open('LinearRegression().pkl','rb'))
elif model == 'Extra Trees Regressor':
    #select pickle file
    model = pickle.load(open('ExtraTreesRegressor().pkl','rb'))
elif model == 'Neural Network':
    #select h5 file
    model = load_model('model.h5')
soil_pH = int(soil_pH)
soc_per = int(soc_per)
som_per = int(som_per)
soil_total_N = int(soil_total_N)
mean_rainfal_mm = int(mean_rainfal_mm)
measurement_depth_m = int(measurement_depth_m)
fertilizer_rate_kg = int(fertilizer_rate_kg)
crop_type = int(crop_type)
experiment_type = int(experiment_type)
ph_class = int(ph_class)
soil_texture = int(soil_texture)
fertiisation_method = int(fertiisation_method)
fertilizer_name = int(fertilizer_name)
measuring_method = int(measuring_method)
fertilizer_type = int(fertilizer_type)
#predict the output
if st.button('Predict'):
    output = model.predict([[soil_pH, soc_per, som_per, soil_total_N, mean_rainfal_mm, measurement_depth_m, fertilizer_rate_kg, crop_type, experiment_type, ph_class, soil_texture, fertiisation_method, fertilizer_name, measuring_method, fertilizer_type]])
    st.success('The predicted nitrate leaching is {}'.format(output))




