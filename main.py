import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class GaussianNB:

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes, n_features = len(self.classes_), X.shape[1]

        self.means_ = np.zeros((n_classes, n_features))
        self.vairances_ = np.zeros((n_classes, n_features))
        self.priors_ = np.zeros(n_classes)

        for idx, k in enumerate(self.classes_):
            Xk = X[y == k]

            self.means_[idx] = Xk.mean(axis=0)
            self.vairances_[idx] = Xk.var(axis=0)
            self.priors_[idx] = Xk.shape[0] / X.shape[0]

        return self

    def _log_gaussian(self, X):
        num = -0.5 * (X[:, None, :] - self.means_) ** 2 / self.vairances_
        log_prob = num - 0.5 * np.log(2 * np.pi * self.vairances_)

        return log_prob.sum(axis=2)

    def predict(self, X):
        X = np.asarray(X)

        log_likelihood = self._log_gaussian(X)
        log_prior = np.log(self.priors_)

        return self.classes_[np.argmax(log_likelihood + log_prior, axis=1)]


X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = GaussianNB().fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_pred, y_test) * 100

print(
    f"Accuracy of the Gaussian Naive Bayes classifier algorithm : {acc:.2f}%")
