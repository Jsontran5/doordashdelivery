# Predicting DoorDash Delivery
### by Dylan Chan, Michael Co, Max Fukuhara, Thomas McConnell, and Jason Tran
## 1.1 Data Set
Our data set ([link](https://www.kaggle.com/datasets/dharun4772/doordash-eta-prediction)) was downloaded from Kaggle under the MIT license by Dharun Suryaa. As a quick overview, each row entry represents an order through DoorDash and each column represents feature variables that we examined throughout this project. The self-explanatory variables are listed as the following:

- **Time Features:** `market_id` (city/region), `created_at` (UTC), `actual_delivery_time` (UTC)
- **Store Features:** `store_id`, `store_primary_category` (cuisine), `order_protocol` (modes)
- **Order Details:** `total_items`, `subtotal` (cents), `num_distinct_items`, `min_item_price` (cents), `max_item_price` (cents)
- **Market Features:** 
  - `total_onshift_dashers` (onshift who are currently working on order) 
  - `total_busy_dashers` 
  - `total_outstanding_orders` (number of orders currently being processed)
- **Other Models’ Measurements:** 
  - `estimated_order_place_duration` (estimated time in secs for the restaurant to receive the order from DoorDash) 
  - `estimated_store_to_consumer_driving_duration` (estimated travel time in secs between store and consumer)

The target value to predict is how long the orders take, denoted by `deliver_time_seconds`, which is `actual_delivery_time - created_at`.

---

## 1.2 Overview of Problem
For many students living on their own, cooking is sometimes not an option. From balancing school, life, work, and extracurricular activities, we may need to rely on food delivery services from time to time. However, the given estimated delivery times are not the best or the most reliable, which is detrimental when needing a fast delivery service. Thus, our goal with this project is to see if we can better predict how long the food delivery will take based on your given situation so that you can better assess your options and manage your time.

---

## 1.3 Key Effective Methodology and Why
The first important thing we did was to pre-process the data. This was necessary and it ensured that the data is clean, consistent, and suitable for modeling. Through pre-processing, we were able to improve the quality of the dataset and enhance the accuracy, efficiency, and reliability of the models we built on it.

Some of the key methodologies we used to try to interpret and make predictions based on our data include: linear regression, regularization, logistic regression, K-nearest neighbors, PCA clustering, and neural networks. Using multiple different models allowed us to look at the data through multiple lenses and try to determine what the most optimal model was for our dataset so that we could provide the best predictions.

Based on the numerical results mentioned below as well as combined domain knowledge, our key effective methodology was neural networks. This will be explained after going over the results in the next section.

---

## 1.4 Results
The different models we chose provided different kinds of results, some being a regressor and some being a classifier, we ultimately decided that the neural network model provided the best overall utility for our end goal of this project.

Our worst performing models were the classifier models: KNN clustering and PCA clustering. Of all the cross validation folds, PCA clustering was only able to achieve a silhouette score as high as 0.09, indicating poor clustering performance. Additionally, while KNN clustering had a great True Negative (TN) score, it had a poor True Positive (TP) and a poor AUC score of only 0.67. Not only were these two modelling methodologies not effective, they deviated from our original goal.

Additionally, the two regression models did not perform as well as we would like to have. The linear regression model, even after performing regularization, we were still reaching an average error about 17.5 minutes. As users of food delivery services, we would be extremely dissatisfied with a predicted arrival time off by nearly 20 minutes. Seeing how poorly the linear regression model did, we opted to use logistic regression as a classifier for “quick” orders and “not quick” orders (under or over 30 minutes). Key results include a TN rate of 0.77, a TN rate of 0.7, and an accuracy of 0.71. By adjusting the class weights we were able to significantly improve the detection of quick deliveries while maintaining a balanced trade-off between identifying both classes. The model was also able to achieve an Area Under the ROC Curve (AUC) of 0.81, indicating moderate discriminatory power.

Lastly, we decided that neural networks performed the best overall and brought us the closest to our goals. While the neural network does not cater exactly to the original goal of predicting the precise duration of a delivery, it is still able to provide very helpful and useful information that any DoorDash user can appreciate: if their order will be “quick” or “slow”. With an achieved accuracy of 87% and an AUC-ROC score of 0.81, this model reflects strong discriminatory power between quick and non-quick deliveries. Although the neural network achieved similar accuracy to logistic regression, it excelled at capturing complex, non-linear relationships, as reflected in its higher AUC-ROC score. Overall, the neural network showcased strong performance and highlighted the potential of deep learning for predicting delivery times.

---

## 1.5 How to Use Code
The code for the project is organized in a Google Colab notebook, with each section conveniently aligned with the topics covered in the appendix. To run it:

1. Download the `Predicting_DoorDash_Delivery.ipynb` file from the [GitHub link](https://github.com/maxfukuh4ra/doordashdelivery) and upload it to Google Colab or any Jupyter Notebook Reader.
   - The notebook already contains code to directly download and load the dataset into the environment.
2. Ensure all dependencies are installed. Required libraries like `pandas`, `numpy`, `matplotlib`, and `sklearn` are already imported into the notebook.
   - Just run the cell.
3. Run the notebook cells.
   - **Check-In #1:** Data Cleaning and Preprocessing & Exploratory Data Analysis
   - **Check-In #2:** Linear Regression Model
   - **Check-In #3:** Logistic Regression
   - **Check-In #4:** KNN, Decision Trees, and Random Forest
   - **Check-In #5:** PCA and Clustering
   - **Check-In #6:** Neural Networks

Outputs for each section (e.g., metrics, plots, and model summaries) are displayed below the corresponding cells. Users can additionally modify the notebook to adjust thresholds or change specific parameters.
