#from utils import db_connect
#engine = db_connect()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.feature_selection import f_classif, SelectKBest




# your code here
df = pd.read_csv("/workspaces/machine-learning-python-template/data/raw/AB_NYC_2019.csv")
print(df.head())
print(df.shape)
print(df.info())
print(df.describe(include=np.number).T)
print(df.describe(include=["O"]).T) #or "object"
print("There are " + str(df.drop("id", axis = 1).duplicated().sum()) + " duplicates")

if df.drop("id", axis = 1).duplicated().sum() != 0:
    df.drop_duplicates(inplace=True)

"""Clean the data"""

df.drop(["id", "name", "host_id", "host_name"], axis = 1, inplace = True)
df_num = df.select_dtypes(include=np.number)
df_cat = df.select_dtypes(include=["object"])
print(df.info())


"""Categorical Data"""
fig, axis = plt.subplots(2, 2, figsize = (10, 7))

index = 0
for i in range(2):
    if index > 4:
        break
    for j in range(2):
        c = df_cat.columns[index]
        s = sns.countplot(ax = axis[i,j],data = df_cat, x = c)
        if c in ["neighbourhood", "neighbourhood_group", "last_review", "room_type"]:
            s.set_xticks([])
        index  +=1
# Adjust the layout
plt.xticks(rotation = 90)
plt.tight_layout()

# Show the plot
plt.savefig("df_cat.jpg")


"""Numerical Data"""
fig, axis = plt.subplots(4, 4, figsize = (10, 7))

index = 0
for i in range(2):
    if index > 5:
        break
    for j in range(4):
        c = df_num.columns[index]
        sns.histplot(ax = axis[i,j],data = df_num, x = c)
        sns.boxplot(ax = axis[i+2,j],data = df_num, x = c)
        index  +=1

# Adjust the layout
plt.tight_layout()

# Show the plots
plt.savefig("df_num.jpg")

plt.clf()
sns.countplot(data = df, x="room_type", hue="price")
plt.savefig("corr_price_roomType.jpg")

plt.clf()
sns.regplot(data=df, x="minimum_nights", y="price")
plt.savefig("reg_min_nights_price.jpg")

plt.clf()
sns.regplot(data=df, x="reviews_per_month", y="number_of_reviews")
plt.savefig("reg_reviews.jpg")

"""Heatmaps & Analysis of multivariate variables"""

plt.clf()
sns.heatmap(df_num.corr(), annot = True)
plt.tight_layout()
plt.savefig("num_heatmap.jpg")

plt.clf()
sns.pairplot(data = df)
plt.tight_layout()
plt.savefig("pairplot_total.jpg")
print("Day 1 Complete")


"""DAY 2 - Feature Engineering"""
summary  =  df.describe().T
print(summary)

plt.clf()
fig, axis = plt.subplots(3, 3, figsize = (15, 10))

sns.boxplot(ax = axis[0, 0], data = df, y = "latitude")
sns.boxplot(ax = axis[0, 1], data = df, y = "longitude")
sns.boxplot(ax = axis[0, 2], data = df, y = "price")
sns.boxplot(ax = axis[1, 0], data = df, y = "minimum_nights")
sns.boxplot(ax = axis[1, 1], data = df, y = "number_of_reviews")
sns.boxplot(ax = axis[1, 2], data = df, y = "reviews_per_month")
sns.boxplot(ax = axis[2, 0], data = df, y = "calculated_host_listings_count")
sns.boxplot(ax = axis[2, 1], data = df, y = "availability_365")

plt.tight_layout()
plt.savefig("boxplots.jpg")

price_stats = df["price"].describe()
print(price_stats)

price_iqr = price_stats["75%"]-price_stats["25%"]
price_upper_limit = price_stats["75%"] + 1.5 * price_iqr
price_lower_limit = price_stats["25%"] - 1.5 * price_iqr

print(f"The upper and lower limits for finding outliers are {round(price_upper_limit, 2)} and {round(price_lower_limit, 2)}, with an interquartile range of {round(price_iqr, 2)}")

print(df[df["price"]>5000])

print(df.isnull().sum().sort_values(ascending=False)/len(df))
#convert last_review column to pandas datetime type.
df['last_review'] = pd.to_datetime(df['last_review'])
"""
# Fill missing dates with the previous valid date
df['last_review'] = df['last_review'].fillna(method='ffill')
#fill missing data with mean number of reviews per month
df["reviews_per_month"].fillna(df["reviews_per_month"].mean(), inplace = True)
"""
# Identify columns with numerical data for mean imputation
numeric_cols = df.select_dtypes(include=np.number).columns

# Fill missing dates with the previous valid date
df['last_review'] = df['last_review'].fillna(method='ffill')


# Fill missing numerical data with column means
for col in numeric_cols:
    df_num[col].fillna(df_num[col].mean(), inplace=True)

# Check for remaining NaN values
print(df_num.isna().sum().sort_values(ascending=False)/len(df))

"""we want to predict the price of the houses. Split train set y var = price. """
num_variables = df_num.columns
num_var = ["latitude", "longitude","minimum_nights","number_of_reviews","reviews_per_month","calculated_host_listings_count","availability_365"]

# We divide the dataset into training and test samples
X = df_num.drop("price", axis = 1)

y = df_num["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(X_train.info())


#Min Max Scaling
scaler = MinMaxScaler()
scaler.fit(X_train)


# Transform training data
X_train_scal = scaler.transform(X_train)
X_train_scal = pd.DataFrame(X_train_scal, index = X_train.index, columns = num_var)

# Transform testing data
X_test_scal = scaler.transform(X_test)
X_test_scal = pd.DataFrame(X_test_scal, index = X_test.index, columns = num_var)

#print(X_train_scal.head())


print(X_train_scal.isna().sum())
#print(X_train_scal.info())
#print(X_test_scal.info())
"""Day 3"""

# With a value of k = 5 we implicitly mean that we want to remove 2 features from the dataset
selection_model = SelectKBest(f_classif, k = 5)
selection_model.fit(X_train_scal, y_train)
ix = selection_model.get_support()
X_train_sel = pd.DataFrame(selection_model.transform(X_train), columns = X_train.columns.values[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test), columns = X_test.columns.values[ix])

print(X_train_sel.head())
print(X_test_sel.head())

X_train_sel["price"] = list(y_train)
X_test_sel["price"] = list(y_test)

X_train_sel.to_csv("/workspaces/machine-learning-python-template/assets/clean_NYC_realestate_train.csv", index=False)
X_test_sel.to_csv("/workspaces/machine-learning-python-template/assets/clean_NYC_realestate_test.csv", index=False)

print("Day 3 Complete")
