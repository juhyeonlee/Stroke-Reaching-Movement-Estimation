import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, r2_score
from scipy.integrate import trapz
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import itertools
from sklearn.svm import SVC
import os
from collections import namedtuple
from scipy.special import fdtrc


def magnitude(v, axis=1):
    return np.sqrt(np.sum(np.square(v), axis=axis))

def magnitude_xy(v, axis=1):
    return np.sqrt(np.sum(np.square(v)[:, 0:2], axis=axis))


def rms(v):
    return np.sqrt(np.mean(np.square(v)))


def regression_performance(y, y_hat, ax=None, pedi=None, color=None):

    if pedi is None:
        pedi = np.zeros(y.shape, dtype=bool)

    if ax is not None:
        max_score = max(y.max(), y_hat.max())
        ax.scatter(y, y_hat, c=color)

        ax.plot([0, max_score], [0, max_score])
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Estimations')
        ax.axis('square')
        # ax.legend()
    plt.show()
    r2 = r2_score(y, y_hat)
    rmse = np.sqrt(np.mean(np.square(y_hat - y)))
    nrmse = rmse / (y.max() - y.min()) * 100
    return r2, rmse, nrmse


def plot_confusion_matrix(ax, y_true, y_pred, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    im = ax.matshow(cm, interpolation='nearest', cmap=cmap)
    ax.grid(False)
    plt.colorbar(im)
    # tick_marks = np.arange(-0.5, 1.5)
    # ax.set_xticks(tick_marks)
    # ax.set_yticks(tick_marks)
    ax.xaxis.tick_bottom()
    ax.set_xticklabels([''] + classes)
    ax.set_yticklabels([''] + classes)

    fmt = '0.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt) + ('%' if normalize else ''),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=22)

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')



# class CFS:
#     def __init__(self, rfunc=pearsonr):
#         self.rfunc = rfunc
#
#     def cfs_score(self, X, y):
#         """See https://en.wikipedia.org/wiki/Feature_selection#Correlation_feature_selection"""
#         k = X.shape[1]
#         class_corr_sum = sum([abs(self.rfunc(X[:, i], y)[0]) for i in range(k)])
#         if k == 1:
#             return class_corr_sum
#         else:
#             feat_corr_sum = 0.
#             for i in range(k):
#                 for j in range(i+1, k):
#                     feat_corr_sum += abs(self.rfunc(X[:, i], X[:, j])[0])
#             return class_corr_sum / np.sqrt(k + 2 * feat_corr_sum)
#
#     def fit(self, X, y, k):
#         selected = []
#         # best_score_selected = []
#         for i in range(k):
#             best_score = -np.inf
#             best_j = -1
#             for j in range(X.shape[1]):
#                 if j not in selected:
#                     score = self.cfs_score(X[:, selected + [j]], y)
#                     if score > best_score:
#                         best_score = score
#                         best_j = j
#             selected += [best_j]
#         #     best_score_selected += [best_score]
#         self.selected_features = selected
#         # self.best_score_selected = best_score_selected
#         return self
#
#     def transform(self, X):
#         return X[:, self.selected_features]
#
#     def get_features_name(self, features_name):
#         return features_name[self.selected_features]

class CFS():
    """Single threaded correlation feature selection, with forwards and backwards search methods."""

    def __init__(self, rfunc=pearsonr):
        self.rfunc = rfunc

    def _merit(self, x):
        """Calculate the merit of the feature set determine by the boolean vector x"""
        assert x.dtype == np.bool, "x is not a boolean vector!"
        b_mask = np.triu(x[:, np.newaxis] @ x[np.newaxis, :],
                         1)  # Denotes which elements of B should be included in the cross correlation sum
        return (self.A @ x) / np.sqrt(x.sum() + 2 * np.sum(self.B * b_mask))

    def fit(self, X, y, method='forward', max_features=None, k=None):
        n_features = X.shape[1]
        self.n_features = n_features
        max_features = n_features if max_features is None else max_features

        # Create a vector of absolute correlations (A) and matrix of absolute cross-correlations (B)
        self.A = np.zeros((n_features,))
        self.B = np.zeros((n_features, n_features))
        for i in range(n_features):
            self.A[i] = self.rfunc(X[:, i], y)[0]
            for j in range(i + 1, n_features):
                self.B[i, j] = self.rfunc(X[:, i], X[:, j])[0]
                self.B[j, i] = self.B[i, j]
            self.B[i, i] = 1
        self.A = np.absolute(self.A)
        self.B = np.absolute(self.B)

        # Using the specified method, search through the feature space and select the set with the most merit
        if method == 'forward':
            features = self._forward_selection(max_features, k)
        elif method == 'backward':
            features = self._backward_selection(max_features, k)
        else:
            raise ValueError('An unknown CFS method was specified: {}'.format(method))

        self.selected_features = np.arange(n_features)[features]

        return self  # For chaining

    def _forward_selection(self, max_features, k):
        max_features = max_features if k is None else k
        merits, xs = np.zeros(max_features, ), np.zeros((max_features, self.n_features), dtype=np.bool)

        x = np.zeros((self.n_features,), dtype=np.bool)
        for f in range(max_features):
            best_merit = -np.inf
            best_x = x
            for i in range(self.n_features):
                if x[i] == 0:
                    x_ = np.copy(x)
                    x_[i] = 1
                    merit = self._merit(x_)
                    if merit > best_merit:
                        best_merit = merit
                        best_x = x_
            merits[f] = best_merit
            xs[f, :] = best_x
            x = best_x

        if k is not None:
            return xs[k - 1, :]
        else:
            return xs[np.argmax(merits), :]

    def _backward_selection(self, max_features, k):
        merits, xs = np.zeros(self.n_features - 1, ), np.zeros((self.n_features - 1, self.n_features), dtype=np.bool)

        x = np.ones((self.n_features,), dtype=np.bool)
        for f in reversed(range(self.n_features - 1)):
            best_merit = -np.inf
            best_x = x
            for i in range(self.n_features):
                if x[i] == 1:
                    x_ = np.copy(x)
                    x_[i] = 0
                    merit = self._merit(x_)
                    if merit > best_merit:
                        best_merit = merit
                        best_x = x_
            merits[f] = best_merit
            xs[f, :] = best_x
            x = best_x

        if k is not None:
            return xs[k - 1, :]
        else:
            return xs[np.argmax(merits[:max_features]), :]

    def transform(self, X):
        return X[:, self.selected_features]


class ExhaustiveForwardSelect:
    def __init__(self, score_func):
        self.score_func = score_func

    def fit(self, X, y):
        best_selected = []
        best_score = -np.inf

        selected = []
        while len(selected) < X.shape[1]:
            loop_score = -np.inf
            loop_selection = -1
            for i in range(X.shape[1]):
                if i not in selected:
                    score = self.score_func(X[:, selected + [i]], y)
                    if score > loop_score:
                        loop_score = score
                        loop_selection = i
            selected += [loop_selection]
            if loop_score > best_score:
                best_score = loop_score
                best_selected = selected.copy()

        self.selected_features = best_selected
        return self

    def transform(self, X):
        return X[:, self.selected_features]


def efs_score(X, y):
    # model = RandomForestClassifier(n_estimators=100, random_state=0, class_weight='balanced').fit(X, y)
    model = SVC(C=5, random_state=0, class_weight='balanced', gamma='auto').fit(X, y)
    y_hat = model.predict(X)
    return f1_score(y, y_hat, average='macro')


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def extract_primary_mag(vmag, threshold=np.sqrt(2) * 0.01, min_len=10):

    # Threshold vmag then find zeros, add start and end as potential bounds
    tvmag = vmag.copy()
    tvmag[tvmag < threshold] = 0  # Apply threshold
    zidx = np.where(tvmag == 0)  # Find indices of zeros
    zidx = np.insert(zidx, 0, 0)  # Add start index
    zidx = np.append(zidx, len(vmag) - 1)  # Add end index
    zidx = np.unique(zidx)  # Remove duplicates (at start or end)

    # Extract candidate bounds from segments longer than the minimum length
    bounds = []  # Index bounds of candidate primary movements
    for si, sf in pairwise(zidx):
        if (sf - si) >= min_len:
            bounds.append((si, sf + 1))
    if len(bounds) == 0:
        return [(0, len(vmag))]

    # Merge adjacent bounds where the gap is smaller then the minimum length
    merged_bounds = []
    start = None
    end = None
    for i in range(len(bounds) - 1):
        si1, sf1 = bounds[i]
        si2, sf2 = bounds[i + 1]
        if start is None:
            if (si2 - sf1) <= min_len:
                start = si1
                end = sf2
            else:
                merged_bounds.append(bounds[i])
        else:
            if (si2 - sf1) <= min_len:
                end = sf2
            else:
                merged_bounds.append((start, end))
                start = None
                end = None
    if start is None:
        merged_bounds.append(bounds[-1])
    else:
        merged_bounds.append((start, end))

    # Pick the highest distance segment as the primary movement
    distances = []
    for si, sf in merged_bounds:
        distances.append(trapz(vmag[si:sf], dx=0.01))
    si, sf = merged_bounds[np.argmax(distances)]
    return [(si, sf)]


def extract_full_mvmt(vel):
    return [(0, vel.shape[0])]


def detect_zero_crossings(v):
    zc = [0, v.shape[0]-1]
    for i in range(v.shape[0]-1):
        if (v[i] < 0 and v[i+1] >= 0) or (v[i] > 0 and v[i+1] <= 0):
            zc.append(i+1)
    zc = np.unique(zc)
    sI, sF = zc[:-1], zc[1:]
    return sI, sF


def extract_first_me(vmag):
    si, sf = detect_zero_crossings(vmag)
    if len(si) > 1:
        print('more than one me')
    return [(si[0], sf[0])]


def plot_save_xyz(sub, dist, angle, score_reach, score_retract, data, divide, save_path, nofilt_data=None):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_type = save_path.split('_')[1]

    fig = plt.figure(figsize=[18, 8])

    ax_x = fig.add_subplot(3, 1, 1)
    ax_y = fig.add_subplot(3, 1, 2)
    ax_z = fig.add_subplot(3, 1, 3)

    ax_x.set_title('{} : sID {} / dist: {} / angle: {} / label: {} / {}'.format(
        data_type, sub, dist, angle, score_reach, score_retract))

    if nofilt_data is not None:
        ax_x.plot(nofilt_data[:, 0], label=data_type + ' x before filtering')
    ax_x.plot(data[:, 0], label=data_type + ' x')
    ax_x.legend()
    ax_x.set_ylabel(data_type)
    ax_x.axhline(0, color='k')
    ax_x.axvline(divide, color='r', linestyle='--')

    if nofilt_data is not None:
        ax_y.plot(nofilt_data[:, 1], label=data_type + ' y before filtering')
    ax_y.plot(data[:, 1], label=data_type + ' y')
    ax_y.legend()
    ax_y.set_ylabel(data_type)
    ax_y.axhline(0, color='k')
    ax_y.axvline(divide, color='r', linestyle='--')

    if nofilt_data is not None:
        ax_z.plot(nofilt_data[:, 2], label=data_type + ' z before filtering')
    ax_z.plot(data[:, 2], label=data_type + ' z')
    ax_z.legend()
    ax_z.set_ylabel(data_type)
    ax_z.axhline(0, color='k')
    ax_z.axvline(divide, color='r', linestyle='--')

    fig.tight_layout()
    fig.savefig(os.path.join(save_path, '{}_{}_{}'.format(sub, dist, angle)))
    plt.close()


def welch_anova_np(*args, var_equal=False):
    # https://svn.r-project.org/R/trunk/src/library/stats/R/oneway.test.R
    # translated from R Welch ANOVA (not assuming equal variance)

    F_onewayResult = namedtuple('F_onewayResult', ('statistic', 'pvalue'))

    args = [np.asarray(arg, dtype=float) for arg in args]
    k = len(args)
    ni =np.array([len(arg) for arg in args])
    mi =np.array([np.mean(arg) for arg in args])
    vi =np.array([np.var(arg,ddof=1) for arg in args])
    wi = ni/vi

    tmp =sum((1-wi/sum(wi))**2 / (ni-1))
    tmp /= (k**2 -1)

    dfbn = k - 1
    dfwn = 1 / (3 * tmp)

    m = sum(mi*wi) / sum(wi)
    f = sum(wi * (mi - m)**2) /((dfbn) * (1 + 2 * (dfbn - 1) * tmp))
    prob = fdtrc(dfbn, dfwn, f)   # equivalent to stats.f.sf
    return F_onewayResult(f, prob)

def plot_correlation_matrix(matrix,
                            target_names_x, target_namses_y,
                            title='feature correlation matrix',
                            cmap=None,
                            normalize=False):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    matrix: shape is (n, n)

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(abs(matrix), interpolation='nearest', cmap=cmap, vmax=1, vmin=0)
    plt.title(title)
    plt.colorbar(im, ax=ax)
    plt.grid(False)
    ax.set_aspect('auto')
    plt.ylim([-0.5, max(len(target_names_x), len(target_namses_y))+0.5])
    if target_names_x is not None:
        # ax.set_xticklabels([''] + list(target_names), fontsize='xx-small', rotation=45)
        # ax.set_yticklabels([''] + list(target_names), fontsize='xx-small')

        plt.xticks(np.arange(0, len(target_namses_y)), target_namses_y, rotation=45, fontsize='xx-small')
        plt.yticks(np.arange(0, len(target_names_x)), target_names_x, fontsize='xx-small')

    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]


    thresh = matrix.max() / 1.5 if normalize else matrix.max() / 2
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.3f}".format(matrix[i, j]),
                     horizontalalignment="center",
                     color="white" if abs(matrix[i, j]) > thresh else "black", fontsize='small')
        else:
            plt.text(j, i, "{:0.3f}".format(matrix[i, j]),
                     horizontalalignment="center",
                     color="white" if abs(matrix[i, j]) > thresh else "black", fontsize='small')


    plt.tight_layout()
    plt.show()