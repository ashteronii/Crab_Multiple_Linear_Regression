import numpy as np

class LinearRegression:
    """
    A simple implementation of Ordinary Least Squares Linear Regression.
    """

    def __init__(self):
        self.coef_ = None  # Will hold the coefficients (weights) for the features.
        self.intercept_ = None  # Will hold the intercept (bias term).

    def fit(self, X, y):
        """
        Trains the model by calculating the coefficients and intercepts based on the data.

        Parameters:
        X (ndarray): Feature matrix.
        y (ndarray): Target vector.

        Returns:
        self: Fitted LinearRegression model instance.
        """
        M = np.column_stack((np.ones(X.shape[0]), X)) # M is the feature matrix X with an additional column of ones added for the intercept term (bias).
        beta = np.linalg.inv(M.T @ M) @ M.T @ y # beta is the vector of coefficients calculated using the Normal Equation for linear regression.
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]
        return self

    def predict(self, X):
        """
        Predicts the target values using the fitted model.

        Parameters:
        X (ndarray): Feature matrix.

        Returns:
        predicted_values (ndarray): Predicted target values:
        """
        predicted_values = X @ self.coef_ + self.intercept_
        return predicted_values

    def score(self, X, y):
        """
        Computes the R^2 score.

        Parameters:
        X (ndarray): Feature matrix.
        y (ndarray): True target values.

        Returns:
        score (float): R^2 score.
        """
        residual_sum_of_squares = np.sum((self.predict(X) - y) ** 2.0)
        total_sum_of_squares = np.sum((y - np.average(y)) ** 2.0)
        score = 1.0 - residual_sum_of_squares / total_sum_of_squares
        return score

    def RMSE(self, X, y):
        """
        Computes the Root Mean Squared Error (RMSE).

        Parameters:
        X (ndarray): Feature matrix.
        y (ndarray): True target values.

        Returns:
        rmse (float): RMSE value.
        """
        rmse = np.sqrt(np.average((self.predict(X) - y) ** 2.0))
        return rmse