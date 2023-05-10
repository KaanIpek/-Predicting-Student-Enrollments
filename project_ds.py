import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Function to process data (impute missing values, remove outliers)
def process_data(X, y=None):
    if y is not None:
        data = pd.concat([X, y], axis=1)
    else:
        data = X.copy()

    # Impute missing values
    data.fillna(data.select_dtypes(include=[np.number]).mean(), inplace=True)

    if y is not None:
        # Outlier detection and removal (1.5 * IQR method)
        Q1 = data.quantile(0.25, numeric_only=True)
        Q3 = data.quantile(0.75, numeric_only=True)
        IQR = Q3 - Q1

        data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        return X, y
    else:
        return data

# Function to select important features and remove less important ones
def select_important_features(X, y, n_features):
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X, y)
    feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

    return feature_importances[:n_features].index
# Also function to select important features and remove less important ones but improved!
def select_and_visualize_features_by_correlation(X, y, num_features=8):
    correlations = X.corrwith(y).abs()
    important_features = correlations.sort_values(ascending=False).head(num_features)

    if "nearest_campus_distance" not in important_features.index:
        important_features = pd.concat([important_features, pd.Series({"nearest_campus_distance": correlations["nearest_campus_distance"]})])

    highest_corr_feature = important_features.index[0]
    scatter_plot(X[highest_corr_feature], y)
    regression_plot(X[[highest_corr_feature]], y, highest_corr_feature)

    return important_features.index

# Function to read and merge data
def read_and_merge_data():
    census_data = pd.read_csv("2022_01_28_Census_Data_by_Geoid.csv")
    gray_data = pd.read_csv("2022_01_28_V2_Gray_Data.csv")
    definition_data = pd.read_csv("2022_01_28_Census_Data_Definition.csv")
    X_train = pd.read_csv("X_train.csv")
    y_train = pd.read_csv("y_train.csv")
    X_test = pd.read_csv("X_test.csv")

    train_data = X_train.merge(y_train, left_index=True, right_index=True)

    train_data = train_data.merge(gray_data, left_on="nearest_campus_hashed_geoid", right_on="geoid_hashed")
    train_data = train_data.merge(census_data, left_on="student_geoid_hashed", right_on="geoid_hashed")

    test_data = X_test.merge(gray_data, left_on="nearest_campus_hashed_geoid", right_on="geoid_hashed")
    test_data = test_data.merge(census_data, left_on="student_geoid_hashed", right_on="geoid_hashed")

    train_data['student_geoid_hashed'] = train_data['student_geoid_hashed'].str.lstrip('a').astype(int)
    train_data['nearest_campus_hashed_geoid'] = train_data['nearest_campus_hashed_geoid'].str.lstrip('a').astype(int)

    test_data['student_geoid_hashed'] = test_data['student_geoid_hashed'].str.lstrip('a').astype(int)
    test_data['nearest_campus_hashed_geoid'] = test_data['nearest_campus_hashed_geoid'].str.lstrip('a').astype(int)

    return train_data, test_data


# Function to build and compare models
def build_and_compare_models(X_train, y_train, X_val, y_val):
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "RandomForestRegressor": RandomForestRegressor(random_state=42)
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        mse, rmse, r2 = calculate_metrics(y_val, y_val_pred)
        results.append({"model": name, "mse": mse, "rmse": rmse, "r2": r2})

    return pd.DataFrame(results)


# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return mse, rmse, r2


# Function to create scatter plot
def scatter_plot(x, y):
    sns.scatterplot(x=x, y=y)
    plt.title("Scatter Plot of Nearest Campus Distance vs Starts")
    plt.xlabel("Nearest Campus Distance")
    plt.ylabel("Starts")
    plt.show()


# Function to create regression plot
def regression_plot(x, y, highest_corr_feature):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=x.squeeze(), y=y, label="Original data")

    lin_reg = LinearRegression()
    lin_reg.fit(x, y)
    y_pred = lin_reg.predict(x)
    plot_data = pd.DataFrame(
        {highest_corr_feature: x.squeeze(), 'y_pred': y_pred.squeeze()})
    sns.lineplot(data=plot_data, x=highest_corr_feature, y="y_pred", color='m', label='Linear regression line')

    ax.set_xlabel(highest_corr_feature)
    ax.set_ylabel("Starts")
    ax.legend(loc="upper left")
    plt.show()


# Function to train and predict using the best model
def train_and_predict(X_train, y_train, X_val):
    rf_model = RandomForestRegressor(random_state=42)

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    best_rf_model = grid_search.best_estimator_

    y_val_pred = best_rf_model.predict(X_val)

    return y_val_pred, best_rf_model


# Function to plot histogram of residuals
def plot_histogram_of_residuals(residuals):
    sns.histplot(residuals, kde=True)
    plt.title("Histogram of Residuals")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.show()


# Main function
if __name__ == "__main__":
    # Read and merge the data
    train_data, test_data = read_and_merge_data()

    y = train_data["starts"]

    selected_features = ["nearest_campus_hashed_geoid", "student_geoid_hashed", "nearest_campus_id",
                         "nearest_campus_distance", "INCCYMEDHH", "INCCYPCAP", "EDUCYBACH", "EDUCYGRAD", "LBFCYEMPL",
                         "LBFCYUNEM",
                         "DWLCYOWNED", "DWLCYRENT", "DADI", "DACI", "DAGI", "DAEI", "Google", "Inquiries",
                         "BLS Job Openings", "BLS Current Employment"]
    X = train_data[selected_features]

    # Apply data processing steps
    X, y = process_data(X, y)
    test_data = process_data(test_data)

    # Select important features
    # important_features = select_important_features(X, y, 10)
    important_features = select_and_visualize_features_by_correlation(X, y)

    X = X[important_features]
    test_data = test_data[important_features]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model_comparison = build_and_compare_models(X_train, y_train, X_val, y_val)
    print(model_comparison)
    y_val_pred, best_rf_model = train_and_predict(X_train, y_train, X_val)

    mse, rmse, r2 = calculate_metrics(y_val, y_val_pred)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("R-squared:", r2)

    y_pred = best_rf_model.predict(test_data)

    output = pd.DataFrame({"Index": test_data.index, "Predicted_Enrollment": y_pred})
    output.to_csv("output.csv", index=False)
    scatter_plot(X["nearest_campus_distance"], y)
    plot_histogram_of_residuals(y_val - y_val_pred)


