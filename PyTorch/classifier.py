import torch
import numpy as np
from loss import linear_discriminative_eigvals


class LDA(object):
    def __init__(self, lambda_val=1e-3, n_components=None, verbose=False, logger=None):
        self.lambda_val = lambda_val
        self.n_components = n_components
        self.verbose = verbose
        self.logger = logger.info if logger is not None else print

    def fit(self, X, y):
        # Assert X is float32, y is int32
        classes = torch.unique(y)

        if self.n_components is None:
            self.n_components = classes.shape[0] - 1

        means = []
        for i in classes:
            Xg = X[y == i]
            means.append(torch.mean(Xg, dim=0))
        self.means = torch.stack(means, dim=0)                                          # [cls, d]

        eigvals, eigvecs = linear_discriminative_eigvals(y, X, self.lambda_val, ret_vecs=True)
        eigvecs = eigvecs.flip(dims=(1,))                                               # [d, cls]
        eigvecs = eigvecs / torch.norm(eigvecs, dim=0, keepdim=True)                    # [d, cls]
        self.scaling = eigvecs.detach().data.cpu().numpy()
        self.coef = torch.matmul(
            torch.matmul(self.means, eigvecs), torch.transpose(eigvecs, 0, 1))          # [cls, d]
        self.intercept = -0.5 * torch.diag(
            torch.matmul(self.means, torch.transpose(self.coef, 0, 1)))                 # [cls]
        self.coef = self.coef.detach().data.cpu().numpy()
        self.intercept = self.intercept.detach().data.cpu().numpy()

        eigvals = eigvals.detach().data.cpu().numpy()
        if self.verbose:
            top_k_evals = eigvals[-self.n_components + 1:]
            self.logger("\nLDA-Eigenvalues:", np.array_str(top_k_evals, precision=2, suppress_small=True))
            self.logger("Eigenvalues Ratio min/max: %.3f, Mean: %.3f" % (
                top_k_evals.min() / top_k_evals.max(), top_k_evals.mean()))

        return self

    def prob(self, X):
        prob = np.dot(X, self.coef.T) + self.intercept                                  # [N, cls]
        # prob_sigmoid = 1. / (np.exp(prob) + 1)                                        # [N, cls]
        # sigmoid = prob_sigmoid / np.sum(prob_sigmoid, axis=1, keepdims=True)          # [N, cls]
        # return sigmoid
        return prob

    def pred(self, Xt):
        return np.argmax(self.prob(Xt), axis=1)                                          # [N]

    def test(self, X, y):
        pred = self.pred(X)
        return np.sum(pred == y) / len(pred)

    def map(self, X):
        X_new = np.dot(X, self.scaling)                                                 # [N, cls]
        return X_new[:, :self.n_components]                                             # [N, cls - 1]

