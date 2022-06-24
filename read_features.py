import pandas as pd
import numpy as np

from pathlib import Path
import itertools
import tqdm


topological_names = 's_e_v_c_b0_b1'.split('_')

topological_thresholds = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75]

topological_titles = {'b0': 'betty 0',
                      'b1': 'betty 1',
                      'c': 'simple cycles',
                      'e': 'edge number',
                      's': 'strongly connected components',
                      'v': 'average vertex degree'}

topological_feature_names = [f'{n}_t{t}' for n in topological_names for t in topological_thresholds]

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
    'h1_nb']

template_feature_names = [
    'self',
    'beginning',
    'prev',
    'next',
    'comma',
    'dot']


def topological_get_layer_head(features, layer, head, topological_feature_names):
    df = features[layer, head, :, :, :]
    df = np.moveaxis(df, 0, -1)
    df = np.moveaxis(df, 2, 1)
    nfeat = df.shape[1]
    nthrs = df.shape[2]
    df = df.reshape((df.shape[0], nfeat * nthrs))
    try:
        return pd.DataFrame(df, columns=topological_feature_names)
    except ValueError as err:
        errmsg = 'Check topological_feature_names length, it is incompatible with the df.shape (possibly redundant feature `w`)'
        raise Exception(errmsg) from err


def barcode_get_layer_head(features, layer, head, barcode_names_passed=barcode_feature_names):
    return pd.DataFrame(features[layer, head, :, :], columns=barcode_names_passed)


def template_get_layer_head(features, layer, head, template_names_passed=template_feature_names):
    df = features[layer, head, :, :]
    return pd.DataFrame(df.T, columns=template_names_passed)


def load_features(dataset_name, model_name, features_dir="./features", heads="all", layers=12,
                  max_len=64, topological_thr=6,
                  topological_features="s_e_v_c_b0b1"):
    """
    Load topological, barcodes and template features from features_dir for dataset_name
    Returns:
        features_frame (pandas.DataFrame)
            Frame of concatenated features; column names: feature_layer_head
    """
    features = pd.DataFrame()
    path_joined = '_'.join([dataset_name, str(heads), 'heads', str(layers), 'layers'])
    topological_thrs = '_'.join([topological_features, 'lists_array', str(topological_thr), 'thrs'])
    topological_file = '_'.join([path_joined, topological_thrs]) + "_"
    len_model = '_'.join(['MAX_LEN', str(max_len), model_name])
    path_joined = '_'.join([path_joined, 'MAX_LEN', str(max_len), model_name])
    topological_file += len_model
    barcode_file = path_joined + "_ripser"
    template_file = path_joined + "_template"
    all_features_dict = {
        'topological': (topological_get_layer_head, topological_feature_names, topological_file),
        'barcode': (barcode_get_layer_head, barcode_feature_names, barcode_file),
        'template': (template_get_layer_head, template_feature_names, template_file)}
    pbar = tqdm.tqdm(total=len(all_features_dict) * 12 * layers, desc="Loading {} features...".format(dataset_name),
                     position=0, leave=True)
    for feature_type, read_args in all_features_dict.items():
        function_, feature_names, path = read_args
        features_type_i_path = Path(features_dir).joinpath(*[model_name, path + '.npy'])
        features_table = np.load(features_type_i_path)
        for (layer, head) in itertools.product(range(12), range(layers)):
            X = function_(features_table, layer, head, feature_names)
            columns = X.columns
            columns_names = {i: i + '_' + str(layer) + '_' + str(head) for i in columns}
            X = X.rename(columns=columns_names)
            features = pd.concat([features, X], axis=1)
            pbar.update(1)
    pbar.close()
    return features
