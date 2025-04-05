# Project Description
You work for the oil extraction company OilyGiant. Your task is to find the best locations to drill 200 new oil wells.

To complete this task, you will need to perform the following steps:

Read the files with parameters collected from oil wells in the selected region: crude oil quality and reserve volume.
Create a model to predict the reserve volume in new wells.
Select the oil wells with the highest estimated values.
Select the region with the highest total profit for the selected oil wells.
You have data on crude oil samples from three regions. The parameters for each oil well in the region are already known. Create a model to help you select the region with the highest profit margin. Analyze the potential benefits and risks using the bootstrapping technique.

## Conditions:
Only linear regression should be used for model training.
When exploring the region, a 500-point survey is conducted, with the best 200 points selected for profit calculation.
The budget for developing 200 oil wells is $100 million.
One barrel of raw material generates $4.50 in revenue. The revenue from one unit of product is $4,500 (reserve volume is expressed in thousands of barrels).
After risk assessment, only retain regions with a loss risk of less than 2.5%. Of those that meet the criteria, the region with the highest average profit should be selected.
The data is synthetic: contract details and well characteristics are not published.

## Data Description
Geological exploration data for the three regions are stored in files:

geo_data_0.csv. Download the dataset
geo_data_1.csv. Download the dataset
geo_data_2.csv. Download the dataset
id — unique oil well identifier
f0, f1, f2 — three characteristics of the points (their specific meaning is not important, but the characteristics themselves are significant)
product — volume of reserves in the oil well (thousands of barrels).

## Project Instructions

- Download and prepare the data.
- Train and test the model for each region in geo_data_0.csv:
- Split the data into a training set and a validation set in a 75:25 ratio.
- Train the model and make predictions for the validation set.
- Save the predictions and correct answers for the validation set.
- Display the predicted average reserve volume and RMSE of the model.
- Analyze the results.
- Put all the previous steps into functions and run them for the other datasets.

Prepare for the profit calculation:
- Store all the values ​​needed for the calculations in separate variables.
- Given an investment of $100 million for 200 oil wells, on average, an oil well must produce at least $500,000 worth of units to avoid losses (this is equivalent to 111.1 units). Compare this amount with the average reserves in each region.
- Present conclusions on how to prepare for the profit calculation step.
- Write a function to calculate the profit for a set of selected oil wells and model the predictions:
- Select the 200 wells with the highest prediction values ​​from each of the three regions (i.e., 'csv' files).
- Summarize the target reserve volume based on these predictions. Store the predictions for the 200 wells in each region. de las 3 regiones.
- Calcula la ganancia potencial de los 200 pozos principales por región. Presenta tus conclusiones: propón una región para el desarrollo de pozos petrolíferos y justifica tu elección.
- Calcula riesgos y ganancias para cada región:
- Utilizando las predicciones que almacenaste, emplea la técnica del bootstrapping con 1000 muestras para hallar la distribución de los beneficios.
- Encuentra el beneficio promedio, el intervalo de confianza del 95% y el riesgo de pérdidas. La pérdida es una ganancia negativa, calcúlala como una probabilidad y luego exprésala como un porcentaje.
- Presenta tus conclusiones: propón una región para el desarrollo de pozos petrolíferos y justifica tu elección. ¿Coincide tu elección con la elección anterior en el punto 4.3?
