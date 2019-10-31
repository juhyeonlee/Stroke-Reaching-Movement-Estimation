import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from scipy.integrate import trapz
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import itertools
from sklearn.svm import SVC


def magnitude(v, axis=1):
    return np.sqrt(np.sum(np.square(v), axis=axis))

def magnitude_xy(v, axis=1):
    return np.sqrt(np.sum(np.square(v)[:, 0:2], axis=axis))


def rms(v):
    return np.sqrt(np.mean(np.square(v)))


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

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.grid(False)
    ax.set_title(title)
    plt.colorbar(im)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    plt.ylim(-0.5, 2.5)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)

    fmt = '0.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt) + ('%' if normalize else ''),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')


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
    model = SVC(random_state=0, class_weight='balanced', gamma='scale').fit(X, y)
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
