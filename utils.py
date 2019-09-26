import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
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
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)

    fmt = '3.0f' if normalize else 'd'
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
    model = SVC(random_state=0, class_weight='balanced', gamma='scale').fit(X, y)
    y_hat = model.predict(X)
    return f1_score(y, y_hat, average='macro')