import numpy as np

class Linear_Regression:
    """
    A simple implementation of Linear Regression using gradient descent.
    
    Parameters
    ----------
    lr : float, optional (default=0.001)
        Learning rate for gradient descent.
    n_iters : int, optional (default=1000)
        Number of iterations for training.
    """

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the linear regression model to the training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training input samples.
        y : ndarray of shape (n_samples,)
            Target values.
        """
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        for iteration in range(self.n_iters):
            y_pred = np.dot(X, self.weight) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weight -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """
        Predict using the linear regression model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        y_pred = np.dot(X, self.weight) + self.bias
        return y_pred

    def mse(self, y_test, y_pred):
        """
        Calculate the Mean Squared Error (MSE).

        Parameters
        ----------
        y_test : ndarray of shape (n_samples,)
            Ground truth (correct) target values.
        y_pred : ndarray of shape (n_samples,)
            Estimated target values.

        Returns
        -------
        float
            Mean squared error between y_test and y_pred.
        """
        mean_square_error = np.mean((y_test - y_pred) ** 2)
        return mean_square_error



if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    
    X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Linear_Regression(lr=0.001, n_iters=1000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("MSE on test set:", model.mse(y_test, predictions))


    import matplotlib.pyplot as plt

    fig,ax=plt.subplots()

    ax.scatter(X_test,y_test,color='b',label="Our Training Data")

    

    ax.plot(X_test,predictions,color='r',label="Prediction Line")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax.set_title("LR Fitting")

    plt.show()