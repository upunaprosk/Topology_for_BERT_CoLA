import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# features' names
topological_titles = {
    's': 'Connected strong components',
    'w': 'Connected weak components',
    'e': 'Edge number',
    'v': 'Avg. vertix degree',
    'c': 'Simple cycles',
    'b0': 'Betty 0',
    'b1': 'Betty 1'
}
barcode_feature_names = [
    'h0_s',
    'h0_e',
    'h0_t_d',
    'h0_n_d_m_t0.75',
    'h0_n_d_m_t0.5',
    'h0_n_d_l_t0.25',
    'h1_t_b',
    'h1_n_b_m_t0.25',
    'h1_n_b_l_t0.95',
    'h1_n_b_l_t0.70',
    'h1_s',
    'h1_e',
    'h1_v',
    'h1_nb'
]
barcode_titles = {
    'h0_s': 'h0, sum of lengths',
    'h0_e': 'h0, entropy',
    'h0_t_d': 'h0, death time',
    'h0_n_d_m_t0.75': '#h0, death time > t_0.75',
    'h0_n_d_m_t0.5':'#h0, death time > t_0.5',
    'h0_n_d_l_t0.25':'#h0, death time < t_0.25',
    'h1_t_b':'h1, birth time',
    'h1_n_b_m_t0.25':'#h1, birth time > t_0.25',
    'h1_n_b_l_t0.95':'#h1, birth time < t_0.95',
    'h1_n_b_l_t0.70':'#h1, birth time < t_0.7',
    'h1_s':'h1, sum of lengths',
    'h1_e':'h1, entropy',
    'h1_v':'h1, variance of lengths',
    'h1_nb':'number of barcodes in h1' }


template_feature_names = [
    'self',
    'beginning',
    'prev',
    'next',
    'comma',
    'dot']

# functions for getting feature's values w.r.t. layer, head given
def topological_get_layer_head(features, layer, head, topological_feature_names):
    df = features[layer, head, :, :, :]
    df = np.moveaxis(df, 0, -1)
    df = np.moveaxis(df, 2, 1)
    nfeat = df.shape[1]
    nthrs = df.shape[2]
    # 7 topological features x 6 thresholds = 42 features
    df = df.reshape((df.shape[0], nfeat * nthrs))
    return pd.DataFrame(df, columns=topological_feature_names)


def barcode_get_layer_head(features, layer, head, barcode_feature_names = barcode_titles):
    return pd.DataFrame(features[layer, head, :, :], columns=barcode_feature_names)


def template_get_layer_head(features, layer, head, template_feature_names = template_feature_names):
    df = features[layer, head, :, :]
    return pd.DataFrame(df.T, columns=template_feature_names)


def corr(X):
    """Return an array of correlation coefs between features and targets
    :param X: np.array of the shape(y.shape[0], number of features + 1 (targets))
    :rtype: np.array of corr.coefs
    """
    C = np.corrcoef(X, rowvar=False)
    np.fill_diagonal(C, 0)
    C[np.isnan(C)] = 0
    return C[:, -1]


def stat(X):
    """Return an array of p-values

    :param X: np.array of the shape(y.shape[0], number of features + 1 (targets))
    :rtype: np.array of p-values
    """
    y = X['y']
    values_1 = np.argwhere(y.values == 1)  # correct sents
    values_0 = np.argwhere(y.values == 0)  # incorrect sents
    ps = []
    for col in range(X.shape[1] - 1):
        top_values_0 = X.to_numpy()[:, col][values_0]
        top_values_1 = X.to_numpy()[:, col][values_1]
        try:
            _, pval = mannwhitneyu(top_values_0, top_values_1, alternative='two-sided')
            ps.append(pval)
        except:
            # the case of the 1 unique value in the feature values
            # print(f'For the feature {names[col]} all values are the same, layer {layer}, head {head}')
            ps.append(0.9)
    ps = np.array(ps)
    return ps


def get_matrices(features, func, y, names, NUM_heads=12, NUM_layers=12, type_=corr):
    """Return a matrix filled with correlation coefficients, p-values, or roc auc scores

    :param features: np.array of features, of the shape (NUM_layers, NUM_heads)
    :param func: function object for getting features[l, h] out of the features
    :param y: pd.array/np.array of target.values
    :param names: list of features' names
    :param type_: 'corr' or 'stat' or 'auc' - types of coefficients (function's name),
                    default corr

    :rtype: matrix of the shape (NUM_layers, NUM_heads)
    """
    m = {}
    for name in names:
        if name not in m:
            m[name] = np.zeros((NUM_heads, NUM_layers))
            m[name][:, :] = -10
    function = type_
    for layer in range(NUM_layers):
        for head in range(NUM_heads):
            X = func(features, layer, head, names)
            X['y'] = y
            values = function(X)
            for i, name in enumerate(names):
                m[name][layer, head] = values[i]
    max_or_min = False if type_ == stat else True
    show_max_matrices(m, max_or_min)
    return m


def show_max_matrices(mtrs, max_val=True):
    """Print max/min values of matrices and its indices:
    feature: value  layer head
    """
    critical_values = []
    critical_strings = []
    for f in mtrs:
        m = mtrs[f]
        coord = np.argmin(np.abs(m), axis=None)
        if max_val:
            coord = np.argmax(np.abs(m), axis=None)
        i, j = np.unravel_index(coord, m.shape)
        v_critical = m[i, j]
        critical_values.append(v_critical)
        critical_strings.append(f'{f}: {v_critical}  {i} {j}')
    critical_abs = [abs(i) for i in critical_values]
    critical_args = sorted(range(len(critical_abs)), key=critical_abs.__getitem__)
    if max_val:
        critical_args = critical_args[::-1]


    for _ in critical_args: print(critical_strings[_]);
    return


def feat_construction(f, t, topological):
    """
    Return a feat

    :param f: feature name if topological
    :param t: threshold
    :param topological: True if feat for topological features (with thr) else False

    :rtype: string of feature with a threshold if topological
    """
    if topological:
        return f"{f}_t{t}"
    else:
        return f"{t}"


def dict_init(features_class, type_, dictionary, matrices):
    # for features tables construction
    if not dictionary.get(features_class, 0):
        dictionary[features_class] = {}
    dictionary[features_class][type_] = matrices


def make_comparison_data(m_tr, m_nt, feature, names, topological=[]):
    """
    Return list of coefficient (corr|p_value) matrices for 2 models (trained/non-trained)

    :param m_tr: matrix of values for trained model
    :param m_nt: matrix of values for non-trained model
    :param feature: feature name

    :param names: list of names (used for barcodes and template)
    :param topological: thrs list if called for topological else []

    :rtype: data  -> list(Dict)
            len(data) == len(names) || len(topological) if topological
            data[i].keys() == ['trained', 'non-trained']
            data[i][j].shape == (NUM_layers, NUM_heads)
    """
    data = []

    def find_feature_val(t_or_f):
        feat = feat_construction(feature, t_or_f, topological)
        data.append({'trained': m_tr[feat], 'non-trained': m_nt[feat]})

    if topological:
        for _, t in enumerate(topological):
            find_feature_val(t)
    else:
        for f in names:
            find_feature_val(f)
    return data


def find_values(feature, dictionary):
    triplet = []
    for type_ in ['corr', 'stat']:
        triplet.append(dictionary[feature][type_][0])
    return triplet