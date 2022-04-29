import numpy as np

class DecisionStump:

    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.weight = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]

        predictions = np.ones(n_samples)
        predictions[X_column < self.threshold] = -1
        return predictions

class AdaBoost:
    def __init__(self, boostingRounds=5):
        self.boostingRounds = boostingRounds
        self.stumps = []

    def fit(self, X, y):
        #calculates the number of samples and features
        n_samples, n_features = X.shape

        probabilities = np.full(n_samples, (1/n_samples))

        for _ in range(self.boostingRounds):
            stump = DecisionStump()

            min_error = float('inf')
            for feature_i in range(n_features):
                #returns the column of the corresponding feature
                X_column = X[:, feature_i]
                threshold = np.sum(X_column)/len(X_column)

                # generates an array of n_samples dimension composed by 1s
                predictions = np.ones(n_samples)
                predictions[X_column < threshold] = -1

                # missclassified takes from probabilites all those instances in which the prediction is different from y
                missclassified = probabilities[y != predictions]

                # calculates the error by summing all the values in missclassified
                error = sum(missclassified)

                if error == 0 or error == 0.5 or error == 1:
                    break

                if error < min_error:
                    min_error = error
                    stump.threshold = threshold
                    stump.feature_idx = feature_i

            stump.weight = 0.5 * np.log((1-error) / error)

            predictions = stump.predict(X)

            probabilities *= np.exp(-stump.weight * y * predictions)
            probabilities /= np.sum(probabilities)

            self.stumps.append(stump)

    def predict(self, X):
        stump_preds = [stump.weight * stump.predict(X) for stump in self.stumps]
        y_pred = np.sum(stump_preds, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred
