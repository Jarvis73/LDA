import numpy as np
import tensorflow as tf
from loss import linear_discriminative_eigvals


class LDA(object):
    def __init__(self, model, train_loader, test_laoder, test_step,
                 lambda_val=1e-3, n_components=None, verbose=False, logger=None):
        super(LDA, self).__init__(name="acc,val_acc")
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_laoder
        self.test_step = test_step

        self.lambda_val = lambda_val
        self.n_components = n_components
        self.verbose = verbose
        self.logger = logger.info if logger is not None else print

    def fit(self, X, y):
        X = tf.convert_to_tensor(X, tf.float32)
        y = tf.convert_to_tensor(y, tf.int32)
        classes = tf.sort(tf.unique(y).y)

        if self.n_components is None:
            self.n_components = classes.shape[0] - 1

        means = []
        for i in classes:
            Xg = X[y == i]
            means.append(tf.reduce_mean(Xg, axis=0))
        self.means = tf.stack(means, axis=0)                                        # [cls, d]

        eigvals, eigvecs = linear_discriminative_eigvals(y, X, self.lambda_val, ret_vecs=True)
        eigvecs = tf.reverse(eigvecs, axis=[1])                                     # [d, cls]
        eigvecs = eigvecs / tf.linalg.norm(eigvecs, axis=0, keepdims=True)          # [d, cls]
        self.scaling = eigvecs.numpy()
        self.coef = tf.matmul(
            tf.matmul(self.means, eigvecs), tf.transpose(eigvecs, (1, 0)))           # [cls, d]
        self.intercept = -0.5 * tf.linalg.diag_part(
            tf.matmul(self.means, tf.transpose(self.coef, (1, 0))))                  # [cls]
        self.coef = self.coef.numpy()
        self.intercept = self.intercept.numpy()

        eigvals = eigvals.numpy()
        if self.verbose:
            top_k_evals = eigvals[-self.n_components + 1:]
            self.logger("\nLDA-Eigenvalues:", np.array_str(top_k_evals, precision=2, suppress_small=True))
            self.logger("Eigenvalues Ratio min/max: %.3f, Mean: %.3f" % (
                top_k_evals.min() / top_k_evals.max(), top_k_evals.mean()))

        return eigvals

    def prob(self, X):
        prob = np.dot(X, self.coef.T) + self.intercept                           	# [N, cls]
        # prob_sigmoid = 1. / (np.exp(prob) + 1)                                    # [N, cls]
        # sigmoid = prob_sigmoid / np.sum(prob_sigmoid, axis=1, keepdims=True)      # [N, cls]
        # return sigmoid
        return prob

    def pred(self, X):
        return np.argmax(self.prob(X), axis=1)                                      # [N]

    def test(self, X, y):
        pred = self.pred(X)
        return np.sum(pred == y) / len(pred)

    def map(self, X):
        X_new = np.dot(X, self.scaling)                                             # [N, cls]
        return X_new[:, :self.n_components]                                         # [N, cls - 1]
