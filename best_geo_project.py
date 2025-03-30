# # Find the best places to open 200 new oil wells for the company OilyGiant

# # Introduction
# 
# Oil drilling company OilyGiant is asking to find the best locations to drill 
# 200 new oil wells. 
# You have data on crude oil samples from three regions. The parameters of each
# oil well in the region are already known. Create a model that helps choose 
# the region with the highest profit margin. Analyze potential benefits and 
# risks using the bootstrapping technique.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# In[2]:


# Import data

df_0 = pd.read_csv('geo_data_0.csv')
df_1 = pd.read_csv('geo_data_1.csv')
df_2 = pd.read_csv('geo_data_2.csv')


# In[3]:


# The data frame (df_0) information and a sample of the data are printed

print(df_0.head())
df_0.info()
print(df_0.isnull().sum())
df_0.describe()


# In[4]:


# The data frame (df_1) information and a sample of the data are printed

print(df_1.head())
df_1.info()
print(df_1.isnull().sum())
df_1.describe()


# In[5]:


# The data frame (df_2) information and a sample of the data are printed

print(df_2.head())
df_2.info()
print(df_2.isnull().sum())
df_2.describe()

# In[6]:


# Verify duplicated data

print('Duplicated values in df_0:')
print(df_0[df_0.duplicated()])
print("")
print('Duplicated values in df_1:')
print(df_1[df_1.duplicated()])
print("")
print('Duplicated values in df_2:')
print(df_2[df_2.duplicated()])

# In[7]:


def train_test(df,scale):
        
    # Tranform object variables to categorical variables avoinding the dummy problem
    features = df.drop(['id', 'product'], axis=1)
    target = df['product']
    
    # Split the dataset in train, validation and test set (75% train, 25% validation)
    X_train, X_valid, y_train, y_valid = train_test_split(features, target, test_size=0.25, random_state=12345)
    
    if scale == 1:
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_valid = sc.fit_transform(X_valid)

    # Train a LinearRegression model and make predictions
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred = model.predict(X_valid)
    
    # Predicted average reserve volume and RMSE of the model
    
    y_predict_mean = y_pred.mean()
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    
    print(f"Predicted average volume: {y_predict_mean}")
    print(f"RMSE: {rmse}")
    
    return y_valid, y_pred
    


# ### Firts results.
# In[8]:


# Calculation of the model for df_0

print('Region 0')
y_valid_df_0, y_pred_df_0 = train_test(df_0, 1)


# In[9]:


# Calculation of the model for df_1

print('Region 1')
y_valid_df_1, y_pred_df_1 = train_test(df_1, 1)


# In[10]:


# Calculation of the model for df_2

print('Region 2')
# Correction
y_valid_df_2, y_pred_df_2 = train_test(df_2, 1)

# In[11]:


# Set the principal variables 

budget = 100e6
n_wells = 200
cost_per_well = budget/n_wells
min_units = 111.1
cost_per_barrel = 500000/min_units
print(cost_per_barrel)


# In[12]:


# Calculate the average amount of reservations predicted for each region

mean_reserves_0 = y_pred_df_0.mean()
mean_reserves_1 = y_pred_df_1.mean()
mean_reserves_2 = y_pred_df_2.mean()

# Compare quantities with the minimum threshold of 111.1 units

print(f"Cantidad mínima requerida por pozo para evitar pérdidas: {min_units:.2f} mil barriles")
print(f"Cantidad media de reservas predicha en la Región 0: {mean_reserves_0:.2f} mil barriles")
print(f"Cantidad media de reservas predicha en la Región 1: {mean_reserves_1:.2f} mil barriles")
print(f"Cantidad media de reservas predicha en la Región 2: {mean_reserves_2:.2f} mil barriles")

# Preliminary conclusions
if mean_reserves_0 > min_units:
    print("La Región 0 es rentable en promedio.")
else:
    print("La Región 0 no es rentable en promedio.")

if mean_reserves_1 > min_units:
    print("La Región 1 es rentable en promedio.")
else:
    print("La Región 1 no es rentable en promedio.")

if mean_reserves_2 > min_units:
    print("La Región 2 es rentable en promedio.")
else:
    print("La Región 2 no es rentable en promedio.")

# In[13]:


# Function to calculate the profit of a set of selected oil wells

def profits(predict, num_wells, cost_per_barrel, budget):
    best_wells = predict.sort_values(ascending=False).head(200)
    
    # Calculate total volume and profits
    total_volume = best_wells.sum()
    revenue = total_volume * cost_per_barrel
    
    # Calculate profit
    profit = revenue - budget
    
    return best_wells, profit


# In[14]:


# Calculate expected profits from the top 200 wells by region

best_wells_df_0, profit_df_0 = profits(pd.Series(y_pred_df_0), n_wells, cost_per_barrel, budget)
best_wells_df_1, profit_df_1 = profits(pd.Series(y_pred_df_1), n_wells, cost_per_barrel, budget)
best_wells_df_2, profit_df_2 = profits(pd.Series(y_pred_df_2), n_wells, cost_per_barrel, budget)

print(f"Expected profit for Region 0: ${profit_df_0:.2f}")
print(f"Expected profit for Region 1: ${profit_df_1:.2f}")
print(f"Expected profit for Region 2: ${profit_df_2:.2f}")

# Determine the most profitable region

if (profit_df_0 > profit_df_1 and profit_df_0 != profit_df_1) and (profit_df_0 > profit_df_2 and profit_df_0 != profit_df_2):
    print("Region 0 has the highest potential benefit and is recommended for well development.")
elif (profit_df_1 > profit_df_0 and profit_df_1 != profit_df_0) and (profit_df_1 > profit_df_2 and profit_df_1 != profit_df_2):
    print("Region 1 has the highest potential benefit and is recommended for well development.")
elif (profit_df_2 > profit_df_0 and profit_df_2 != profit_df_0) and (profit_df_2 > profit_df_1 and profit_df_2 != profit_df_1):
    print("Region 2 has the highest potential benefit and is recommended for well development.")
elif profit_df_0 == profit_df_1 and profit_df_0 > profit_df_2: 
    print("Region 0 and Region 1 have the greatest potential benefit and are recommended for well development.")
elif profit_df_0 == profit_df_2 and profit_df_0 > profit_df_1:
    print("Region 0 and Region 2 have the greatest potential benefit and are recommended for well development.")
else:
    print("Region 1 and Region 2 have the greatest potential benefit and are recommended for well development.")



# In[15]:


# Function to calculate risk and profit using bootstrapping

state = np.random.RandomState(12345)
def bootstrap_profit(predictions, n_bootstrap=1000):
    state = np.random.RandomState(12345)
    values = []
    
    for _ in range(n_bootstrap):
        # Perform a random sample with replacement
        subsample = predictions.sample(n=n_wells, replace=True, random_state=state)
        
        # Calculate the total volume and profit for the sample
        total_volume = subsample.sum()
        revenue = total_volume * cost_per_barrel
        profit = revenue - budget
        values.append(profit)
    
    
    values = pd.Series(values)
    
    mean_profit = values.mean()
    lower = values.quantile(0.025)
    upper = values.quantile(0.975)
    
    # Calculate risk of loss
    risk_of_loss = (values < 0).mean()
    
    return mean_profit, lower, upper, risk_of_loss


# In[16]:


# Apply bootstrapping for each region

print("\nRegion 0:")
mean_profit_df_0, lower_df_0, upper_df_0, risk_df_0 = bootstrap_profit(pd.Series(best_wells_df_0))
print(f"Average profit: ${mean_profit_df_0:.2f}, Confidence interval: [{lower_df_0:.2f}, {upper_df_0:.2f}]")
print(f"Risk of loss: {risk_df_0 * 100:.2f}%")

print("\nRegion 1:")
mean_profit_df_1, lower_df_1, upper_df_1, risk_df_1 = bootstrap_profit(pd.Series(best_wells_df_1))
print(f"Average profit: ${mean_profit_df_1:.2f}, Confidence interval: [{lower_df_1:.2f}, {upper_df_1:.2f}]")
print(f"Risk of loss: {risk_df_1 * 100:.2f}%")

print("\nRegion 2:")
mean_profit_df_2, lower_df_2, upper_df_2, risk_df_2 = bootstrap_profit(pd.Series(best_wells_df_2))
print(f"Average profit: ${mean_profit_df_2:.2f}, Confidence interval: [{lower_df_2:.2f}, {upper_df_2:.2f}]")
print(f"Risk of loss: {risk_df_2 * 100:.2f}%")

# In[17]:

# Comparar y determinar la región más adecuada para el desarrollo
if (mean_profit_df_0 > mean_profit_df_1 and mean_profit_df_0 != mean_profit_df_1) and (mean_profit_df_0 > mean_profit_df_2 and mean_profit_df_0 != mean_profit_df_2) and risk_df_0 < 2.5:
    print("Region 0 is the most profitable and has an acceptable risk of loss.")
elif (mean_profit_df_1 > mean_profit_df_0 and mean_profit_df_1 != mean_profit_df_0) and (mean_profit_df_1 > mean_profit_df_2 and mean_profit_df_1 != mean_profit_df_2) and risk_df_1 < 2.5:
    print("Region 1 is the most profitable and has an acceptable risk of loss.")
elif (mean_profit_df_2 > mean_profit_df_0 and mean_profit_df_2 != mean_profit_df_0) and (mean_profit_df_2 > mean_profit_df_1 and mean_profit_df_2 != mean_profit_df_1) and risk_df_2 < 2.5:
    print("Region 2 is the most profitable and has an acceptable risk of loss.")
elif (mean_profit_df_0 == mean_profit_df_1 and mean_profit_df_0 > mean_profit_df_2) and risk_df_0 < 2.5:
    print("Region 0 and Region 1 have the greatest potential benefit and are recommended for well development.")
elif (mean_profit_df_0 == mean_profit_df_2 and mean_profit_df_0 > mean_profit_df_1) and risk_df_0 < 2.5:
    print("Region 0 and Region 2 have the greatest potential benefit and are recommended for well development.")
elif (mean_profit_df_1 == mean_profit_df_2 and mean_profit_df_1 > mean_profit_df_0) and risk_df_1 < 2.5:
    print("Region 1 and Region 2 have the greatest potential benefit and are recommended for well development.")
else:
    print("No region fits the risk and gain criteria..")