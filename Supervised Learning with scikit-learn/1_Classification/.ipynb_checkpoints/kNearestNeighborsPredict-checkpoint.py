"""
X_new = np.array([[30.0, 17.5],
                  [107.0, 24.1],
                  [213.0, 10.9]])
                  
                  
churn_df = 
   account_length  total_day_charge  total_eve_charge  total_night_charge  total_intl_charge  customer_service_calls  churn
             101             45.85             17.65                9.64               1.22                       3      1
              73             22.30              9.05                9.98               2.75                       2      0
              86             24.62             17.53               11.49               3.13                       4      0
              59             34.73             21.02                9.66               3.24                       1      0
             129             27.42             18.75               10.11               2.59                       1      0
"""


# Import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier 

# Create arrays for the features and the target variable
y = churn_df["churn"].values
X = churn_df[["account_length", "customer_service_calls"]].values

# Create a KNN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors = 6)

# Fit the classifier to the data
knn.fit(X,y)

# Predict the labels for the X_new
y_pred = knn.predict(X_new)

# Print the predictions for X_new
print("Predictions: {}".format(y_pred)) 
