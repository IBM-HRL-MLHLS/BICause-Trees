from sklearn.linear_model import LogisticRegression, LinearRegression
from causallib.estimation import IPW, MarginalOutcomeEstimator, Matching
import numpy as np
import itertools
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter, MultipleLocator
import pandas as pd
from sklearn.model_selection import train_test_split
import itertools
import seaborn as sns
import seaborn.objects as so
import numpy as np
from causallib.estimation import IPW, MarginalOutcomeEstimator
from bicause_tree import BecauseTree, PropensityImbalanceStratification, \
    PropensityStartaficationPropensity, crump, prevalence_symmetric
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import jaccard_score, brier_score_loss, roc_auc_score, mean_squared_error,\
    r2_score, roc_curve
from sklearn.linear_model import LogisticRegression
from copy import deepcopy
from multiprocessing import Pool
import statsmodels.api as sm
from scipy.interpolate import interp1d
from sklearn.calibration import calibration_curve
import statsmodels.api as sm
from scipy.interpolate import interp1d
import seaborn.objects as so
from causallib.evaluation.weight_evaluator import calculate_covariate_balance


def get_data(dataset):
    if dataset == 'twins':
        data = pd.read_csv('./data/twins/ztwins_sample0.csv')
        x = data.drop(columns=['y0', 'y1', 'ite', 'y', 't'])
        t = data['t']
        y = data['y']

    elif dataset == 'acic':
        data = pd.read_csv('./data/acic/zymu_174570858.csv')
        x = pd.read_csv('./data/acic/x.csv')
        t = data['z']
        y = data['y0']
        idx_to_change = data.loc[data['z'] == 1].index.to_list()
        for idx in idx_to_change:
            y.loc[idx] = data['y1'].loc[idx]
        y = y.rename('y')
        one_hot = OneHotEncoder(drop='first').fit(x[['x_2', 'x_21', 'x_24']])
        new_data = pd.DataFrame(one_hot.transform(x[['x_2', 'x_21', 'x_24']]).toarray())
        x = x.drop(columns=['x_2', 'x_21', 'x_24'])
        x = pd.concat([x, new_data], axis=1)

    elif dataset == 'natural_exp':
        data = pd.read_csv('./data/natural_exp/dataset.csv')
        x = data.drop(columns=['T', 'D', 'y1', 'y0'])
        t = data['T']
        y = data['D']

    elif dataset == 'positivity_violations':
        data = pd.read_csv('./data/positivity_violations/data.csv')
        x = data.drop(columns=['T', 'D', 'y1', 'y0'])
        t = data['T']
        y = data['D']

    return data, x, t, y


def _effect(outcomes):
    return outcomes[1] - outcomes[0]

def _train_test_effect(train_test_data, model, ground_truth=False):
    x_train, a_train, y_train = train_test_data[0], train_test_data[2], train_test_data[4]
    x_test, a_test, y_test = train_test_data[1], train_test_data[3], train_test_data[5]
    if ground_truth:
        truey1, truey0 = model[0], model[1]
        effects = [truey1[x_train.index].mean()-truey0[x_train.index].mean(),
                   truey1[x_test.index].mean()-truey0[x_test.index].mean()]
    else:
        effects = [
            _effect(model.estimate_population_outcome(x_train, a_train, y_train)),
            _effect(model.estimate_population_outcome(x_test, a_test, y_test))]
    return effects


def compare_fit_effect(models_to_compare:dict, train_test_data):
    train_data = train_test_data[::2]
    effects=[]
    fitted_models = {}
    for name, model in models_to_compare.items():
        is_ground_truth = (name == 'Ground truth')
        if not is_ground_truth:
            model_copy = deepcopy(model)
            model_copy.fit(*train_data)
            if isinstance(model_copy, BecauseTree):
                propensity_model = PropensityStartaficationPropensity()
                propensity_model.tree = model_copy.tree
                propensity_model.fit_treatment_models(train_data[0], train_data[1])
                setattr(model_copy, 'propensity_model', propensity_model)
            fitted_models[name] = model_copy
        else:
            model_copy=model
        effects.extend(_train_test_effect(train_test_data, model_copy, is_ground_truth))
    return effects, fitted_models

def bootstrap_fit_effect(models_to_compare: dict, X: pd.DataFrame, a: pd.Series, y: pd.Series, tree_names,
                         n_bootstraps=500, test_size=0.5, run_multiprocessing=True):
    # this function does the following n_bootstraps times:
    #      - split the data into train and test sets according to test_size
    #      - record the indices of samples sent to test into bootstrap_matrix
    #      - fit the models in models_to_compare on the train set
    #      - compute the effect on the train and test set into a nested list bootstrap_effects

    bootstrap_effects, bootstrap_fitted_models = [], {}
    bootstrap_matrix = pd.DataFrame(np.zeros((len(X.index), n_bootstraps)), index=X.index)
    list_of_splits = []

    for i in range(n_bootstraps):
        np.random.seed(i)
        train_test_data = train_test_split(X, a, y, test_size=test_size, random_state=i, stratify=a)
        X_test = train_test_data[1]
        for idx in X_test.index:
            bootstrap_matrix.loc[idx, i] = 1
        list_of_splits.append(train_test_data)

    if run_multiprocessing:
        with Pool() as pool:
            zipped_input = zip(
                itertools.repeat(models_to_compare),
                list_of_splits)
            results = pool.starmap(compare_fit_effect, zipped_input)  # returns a list of results
            pool.close()
            pool.join()
    else:
        results = []
        for split in list_of_splits:
            split_result = compare_fit_effect(models_to_compare, split)
            results.append(split_result)

    bootstrap_effects = [res[0] for res in results]
    for idx, element in enumerate(results):
        model_dictionary = element[1]
        for name, model in model_dictionary.items():
            model_id = (name, idx)
            bootstrap_fitted_models[model_id] = model
    # bootstrap_fitted_models is a dict with model_id: model
    # where model_id is a tuple (name, #bootstrap)

    return bootstrap_effects, bootstrap_fitted_models, bootstrap_matrix


def generate_violating_matrix(screening_models: dict, X):
    keys0 = [key[0] for key in screening_models.keys()]
    model_name = keys0[0]
    positivity_matrix = pd.DataFrame(np.zeros((len(X), len(screening_models))), index=X.index)

    for bootstrap_number in range(len(screening_models)):
        model = screening_models[(model_name, bootstrap_number)]
        leaf_summary = model.tree.generate_leaf_summary()
        node_assignment = pd.DataFrame(model.apply(X)).rename(columns={0: 'node_index'})
        individual_positivity = node_assignment.merge(leaf_summary, left_on="node_index",
                                                      right_on="node_index", how="left")
        individual_positivity.index = node_assignment.index
        X_pos_filtered = X.loc[individual_positivity['positivity_violation'] == False]
        for index in list(X_pos_filtered.index):
            positivity_matrix.loc[index, bootstrap_number] = 1

    return positivity_matrix


def filter_results_by_positivity_violation(positivity_boot_mask, bootstrap_results):
    model_names = bootstrap_results['model'].unique()
    filtered_bootstraps = pd.DataFrame(columns=bootstrap_results.columns)

    for model_name in model_names:
        for bootstrap_number in bootstrap_results['bootstrap_number'].unique():
            copy_bootstrap_results = bootstrap_results.loc[(bootstrap_results['model'] == model_name) &
                                                           (bootstrap_results['bootstrap_number'] == bootstrap_number)]
            copy_positivity_boot_mask = positivity_boot_mask.iloc[:, bootstrap_number]
            indices_pos_boot = copy_positivity_boot_mask.loc[copy_positivity_boot_mask == 1].index.to_list()
            copy_bootstrap_results = copy_bootstrap_results[copy_bootstrap_results['X_index'].isin(indices_pos_boot)]
            filtered_bootstraps = pd.concat([filtered_bootstraps, copy_bootstrap_results], axis=0)

    return filtered_bootstraps

def compute_pscore(X, model):

    if isinstance(model, BecauseTree):
        probas = model.propensity_model.predict_proba(X)[:,1]
    if isinstance(model, IPW):
        probas = model.compute_propensity(X, a=None, treatment_values=1)

    return probas

def get_pscores(X, bootstrap_matrix, fitted_models:dict):
    column_names = ["model", "bootstrap_number", "X_index", "propensity_score"]
    bootstrap_pscores = []
    for model_id, model in fitted_models.items():
        name, bootstrap_number = model_id[0], model_id[1]
        bootstrap_mask=bootstrap_matrix.iloc[:,bootstrap_number]
        indices = bootstrap_mask.loc[bootstrap_mask==1].index
        X_test=X.loc[indices]
        pscore_xtest = compute_pscore(X_test, model)
        next_block_df= pd.DataFrame(
            {
                "model": name,
                "bootstrap_number": bootstrap_number,
                "X_index": X_test.index,
                "propensity_score": list(pscore_xtest)
            }
        )
        bootstrap_pscores.append(next_block_df)
    bootstrap_pscores = pd.concat(bootstrap_pscores, ignore_index=True)
    return bootstrap_pscores

def compute_potential_outcomes(X, a, y, model):
    if callable(model.estimate_individual_outcome):
        outcomes = model.estimate_individual_outcome(X, a, y)
    else:
        outcomes = [None, None]
    return outcomes[0], outcomes[1]

def get_outcomes(X, a, bootstrap_matrix, fitted_models: dict, y=None):
    column_names = ["model", "bootstrap_number", "X_index"]
    bootstrap_outcomes = pd.DataFrame(columns=column_names)

    for model_id, model in fitted_models.items():
        name, bootstrap_number = model_id[0], model_id[1]
        bootstrap_mask = bootstrap_matrix.iloc[:, bootstrap_number]
        indices = bootstrap_mask.loc[bootstrap_mask == 1].index
        X_test = X.loc[indices]
        a_test = a.loc[indices]
        y0, y1 = compute_potential_outcomes(X_test, a_test, y, model)
        next_block_df = pd.DataFrame(columns=column_names)
        next_block_df["model"] = np.repeat(name, len(X_test))
        next_block_df["bootstrap_number"] = np.repeat(bootstrap_number, len(X_test))
        next_block_df["X_index"] = X_test.index
        next_block_df = next_block_df.merge(y0, how='left', left_on='X_index', right_index=True)
        next_block_df = next_block_df.merge(y1, how='left', left_on='X_index', right_index=True)
        bootstrap_outcomes = pd.concat([bootstrap_outcomes, next_block_df])

    return bootstrap_outcomes.rename(columns={0: 'y0', 1: 'y1'})


def compute_filtered_effect(X, a, y, bootstrap_matrix, positivity_matrix, fitted_models, models_to_compare):
    n_samples, n_bootstrap = np.shape(bootstrap_matrix)[0], np.shape(bootstrap_matrix)[1]
    filtered_effects = []
    model_names = [key for key in models_to_compare.keys()]
    # in bootstrap_matrix 1 is for test samples 0 for train samples
    # in positivity_matrix 1 is for non-violating samples 0 for train samples

    for bootstrap_number in range(n_bootstrap):
        effects = []
        test_mask = bootstrap_matrix.iloc[:, bootstrap_number]
        positivity_mask = positivity_matrix.iloc[:, bootstrap_number]
        x_train, a_train, y_train = X.loc[test_mask == 0], a[test_mask == 0], y[test_mask == 0]
        x_test, a_test, y_test = X.loc[test_mask == 1], a[test_mask == 1], y[test_mask == 1]
        x_train_filtered, a_train_filtered, y_train_filtered = x_train.loc[positivity_mask == 1], a_train[
            positivity_mask == 1], y_train[positivity_mask == 1]
        x_test_filtered, a_test_filtered, y_test_filtered = x_test.loc[positivity_mask == 1], a_test[
            positivity_mask == 1], y_test[positivity_mask == 1]
        for model_name in model_names:
            is_ground_truth = (model_name == 'Ground truth')
            if not is_ground_truth:
                model = fitted_models[(model_name, bootstrap_number)]
            else:
                model = models_to_compare['Ground truth']
            train_test_data = [x_train_filtered, x_test_filtered, a_train_filtered, a_test_filtered, y_train_filtered,
                               y_test_filtered]
            # train_test_data = [x_test, x_train, a_test, a_train, y_test, y_train]
            effects.extend(_train_test_effect(train_test_data, model, is_ground_truth))
        filtered_effects.extend([effects])

    return filtered_effects


def box_plot_effect_difference(models_to_compare: dict, effects: list, plot_matching=True, plot_causal_tree=True,
                               plot_test=True, path=None):
    ground_truth_index = list(models_to_compare.keys()).index("Ground truth")
    ground_truth_bootstrap_effects = []
    differences = effects

    for sublist_index, sublist in enumerate(differences):
        train_test_ground_truth = sublist[ground_truth_index * 2:(ground_truth_index + 1) * 2]
        ground_truth_bootstrap_effects.append(train_test_ground_truth)
        differences[sublist_index].pop(ground_truth_index * 2)  # delete train_ground_truth_effect for that bootstrap
        differences[sublist_index].pop(ground_truth_index * 2)  # delete test_ground_truth_effect for that bootstrap
        subtracted_array = np.absolute(np.array(sublist) - np.array(train_test_ground_truth * (len(sublist) // 2)))
        differences[sublist_index] = list(subtracted_array)
    model_names = list(models_to_compare.keys())
    model_names.remove("Ground truth")
    columns_tuples = list(itertools.product(model_names, ["Train", "Test"]))
    fig, ax = plt.subplots()
    effects_tp = pd.DataFrame(differences, columns=pd.MultiIndex.from_tuples(columns_tuples, names=["Model", "Phase"]))
    data = effects_tp.stack().stack().reset_index(level=[1, 2]).rename(columns={0: "Estimated ATE"})

    if not plot_test:
        data = data[data['Phase'] != "Test"]
    ax = sns.boxplot(data=data, y="Model", x="Estimated ATE", hue="Phase", ax=ax)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='both', labelsize=12)
    models_label = ax.yaxis.get_ticklabels()
    ax.yaxis.set_ticklabels(models_label)
    ax.set_ylabel(None);
    ax.set_xlabel(r"$ \vert ATE - \widehat{ATE} \vert $", fontsize=13);
    fig.tight_layout()

    if path is not None:
        fig.savefig(fname=path, dpi=600)
    fig.show()

    return fig

def compute_jaccard_rand_score(X, bootstrap_matrix, tree_fitted_models, path=None):

    all_rand_scores, all_jacc_scores = {}, {}
    n_bootstraps = np.shape(bootstrap_matrix)[1]
    keys0 = [key[0] for key in tree_fitted_models.keys()]

    for model_name in np.unique(keys0):
        model_dict={key:value for key,value in tree_fitted_models.items() if key[0]==model_name}
        rand_scores, jacc_scores = [], []
        for model_id, model in model_dict.items():
            name, i = model_id[0], model_id[1]
            for j in range(i+1, n_bootstraps):
                model_j=tree_fitted_models[(name, j)]
                mask_i = bootstrap_matrix.iloc[:,i]
                indices_i = pd.DataFrame(mask_i.loc[mask_i == 0].index)
                mask_j = bootstrap_matrix.iloc[:, j]
                indices_j = pd.DataFrame(mask_j.loc[mask_j == 0].index)
                common_indices = indices_i.merge(indices_j)
                X_test_common= X.loc[common_indices[0]]
                nodes_i = model.apply(pd.DataFrame(X_test_common))
                nodes_j = model_j.apply(pd.DataFrame(X_test_common))
                rand_scores.append(adjusted_rand_score(nodes_i,nodes_j))
                jacc_scores.append(jaccard_score(nodes_i,nodes_j, average='macro'))
            all_rand_scores[model_name] = [np.mean(rand_scores), np.std(rand_scores)]
            all_jacc_scores[model_name] = [np.mean(jacc_scores), np.std(jacc_scores)]

    if path is not None:
        pd.DataFrame(all_rand_scores).to_csv(path_or_buf= path + 'rand_scores.csv')
        pd.DataFrame(all_jacc_scores).to_csv(path_or_buf= path + 'jacc_scores.csv')

    return all_rand_scores, all_jacc_scores

def generate_bias(X, t, y, y1, y0, bootstrap_matrix, tree_model, on_train=True):
    n_bootstraps = np.shape(bootstrap_matrix)[1]
    biases = {}
    for bootstrap_number in range(n_bootstraps):
        test_mask = bootstrap_matrix.iloc[:, bootstrap_number]
        x_train, t_train, y_train = X.loc[test_mask == 0], t[test_mask == 0], y[test_mask == 0]
        tree_model_copy = deepcopy(tree_model)
        tree_model_copy.fit(x_train, t_train, y_train)

        if not on_train:
            x_test, t_test, y_test = X.loc[test_mask == 1], t[test_mask == 1], y[test_mask == 1]
            y1_test, y0_test = y1.loc[test_mask == 1], y0.loc[test_mask == 1]
            ate_hat = tree_model_copy.estimate_population_outcome(x_test, t_test, y_test)
            true_ate = np.mean(y1_test) - np.mean(y0_test)
        else:
            ate_hat = tree_model_copy.estimate_population_outcome(x_train, t_train, y_train)
            y1_train, y0_train = y1.loc[test_mask == 0], y0.loc[test_mask == 0]
            true_ate = np.mean(y1_train)-np.mean(y0_train)
        ate_hat = ate_hat[1] - ate_hat[0]
        bias = true_ate - ate_hat
        biases[bootstrap_number] = bias
    biases = pd.Series(biases).to_frame("Bias").reset_index().rename(columns={"index": "bootstrap"})
    return biases


def generate_bias_across_depth(X, t, y, y1, y0, bootstrap_matrix, model_depths, n_feat_to_plot=10, on_train=True, path=None):
    bias = {}
    for depth in [0] + model_depths:
        tree_model = BecauseTree(
            max_depth=depth, min_treat_group_size=1,
            asmd_threshold_split=0,
            multiple_hypothesis_test_alpha=0.05,
            multiple_hypothesis_test_method='holm',
            positivity_filtering_method=crump,
            positivity_filtering_kwargs={}
        )
        depth_bias = generate_bias(X, t, y, y1, y0, bootstrap_matrix, tree_model, on_train)
        bias[depth] = depth_bias
    bias = pd.concat(bias, names=["Depth"]).reset_index().drop(columns=["level_1"])
    return bias

def bias_plot_across_depth2(bias_data, ipw_data, path):

    bias_data = bias_data.apply(abs)
    ipw_data = ipw_data.apply(abs)

    plot_data = pd.concat({"Tree": bias_data, "IPW": ipw_data}, names=["Model"]).reset_index(level=0).reset_index(
        drop=True)
    plot_data
    fig, ax = plt.subplots()

    depth_data = plot_data.query("(Model == 'Tree') & (Depth != 0)").copy()  # save the `SettingWithCopyWarning`
    depth_data["Depth"] = depth_data["Depth"].astype(int)  # StrMethodFormatter returns trash otherwise

    sns.boxplot(
        depth_data,
        x="Depth",
        y="Bias",
        # hue="depth",
        ax=ax,
        color=".9",
    )
    sns.swarmplot(
        depth_data,
        x="Depth",
        y="Bias",
        hue="Depth",
        ax=ax,
        legend=False,
        size=4.2,
    )

    ipw_bias = plot_data.query("Model == 'IPW'")["Bias"].agg(["mean", "std"])
    ipw_mean_bias = ipw_bias.loc["mean"]
    ipw_bias_width = ipw_bias["std"] * 1.96
    ax.fill_between(
        [-0.5, 9.5],
        [ipw_mean_bias - ipw_bias_width],
        [ipw_mean_bias + ipw_bias_width],
        color="black", alpha=0.2,
        interpolate=True,
    )
    ax.text(
        0, ipw_mean_bias - ipw_bias_width,
        "95% confidence interval",
        color="black", alpha=0.5, fontsize=11,
        va="bottom", ha="left",
    )

    ax.axhline(ipw_mean_bias, linestyle="--", color="black")
    ax.text(-0.3, ipw_mean_bias, "IPW", color="black", va="bottom", ha="left", fontsize=13)

    # ax.legend(fontsize=13)
    # locator = MultipleLocator(1)
    # ax.xaxis.set_major_locator(locator)
    # ax.set_ylim(0, ax.get_ylim()[1])
    # ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    ax.tick_params(axis='both', which='both', labelsize=12)
    ax.set_xlabel("Tree maximum depth", fontsize=13)
    ax.set_ylabel(r"$ \vert ATE - \widehat{ATE} \vert $", fontsize=13)

    fig.tight_layout()
    if path is not None:
        fig.savefig(fname=path + "bias_across_depths.pdf", dpi=600)
    return fig


def generate_weighted_asmds_across_depths(X, t, bootstrap_matrix, model_depths, n_feat_to_plot=10, on_train=True,
                                          path=None):
    weighted_asmds = {}
    for depth in [0] + model_depths:
        pscore_model = IPW(PropensityStartaficationPropensity(max_depth=depth, min_treat_group_size=2,
                                                              asmd_threshold_split=0,
                                                              multiple_hypothesis_test_alpha=0.05,
                                                              multiple_hypothesis_test_method='holm',
                                                              positivity_filtering_method=crump,
                                                              positivity_filtering_kwargs={}))
        weight_asmds = generate_weighted_asmds_leaf(X, t, bootstrap_matrix, pscore_model, on_train)
        weight_asmds.rename(columns={'covariate': 'Covariate', 'weighted': 'Weighted ASMD'}, inplace=True)
        weighted_asmds[depth] = weight_asmds
    weighted_asmds = pd.concat(weighted_asmds, names=["Depth"]).reset_index().drop(columns="level_1")
    weighted_asmds["Model"] = "Tree"

    weighted_asmds = weighted_asmds.groupby(["Covariate", "Depth"])["Weighted ASMD"].mean().reset_index()
    features_to_plot = \
    weighted_asmds.query("Depth == 0").sort_values(by="Weighted ASMD", ascending=False).iloc[:n_feat_to_plot][
        "Covariate"]
    weighted_asmds = weighted_asmds.loc[weighted_asmds["Covariate"].isin(features_to_plot)]

    return weighted_asmds, features_to_plot


def generate_weighted_asmds_leaf(X, t, bootstrap_matrix, pscore_model, on_train=True):
    n_bootstraps = np.shape(bootstrap_matrix)[1]
    weighted_asmds = {}

    for bootstrap_number in range(n_bootstraps):
        test_mask = bootstrap_matrix.iloc[:, bootstrap_number]
        x_train, t_train = X.loc[test_mask == 0], t[test_mask == 0]
        pscore_model_copy = deepcopy(pscore_model)
        pscore_model_copy.fit(x_train, t_train)

        if not on_train:
            x_test, t_test = X.loc[test_mask == 1], t[test_mask == 1]
            weights = pscore_model_copy.compute_weights(x_test, t_test)
            next_weighted_asmds = calculate_covariate_balance(x_test, t_test, weights)['weighted']
        else:
            weights = pscore_model_copy.compute_weights(x_train, t_train)
            next_weighted_asmds = calculate_covariate_balance(x_train, t_train, weights)['weighted']
        weighted_asmds[bootstrap_number] = next_weighted_asmds
    weighted_asmds = pd.concat(weighted_asmds, names=["bootstrap"], axis=0).reset_index()

    return weighted_asmds

def love_plot_across_depths(X, t, bootstrap_matrix, model_depths, weighted_asmds, features_to_plot, n_feat_to_plot=10,
                            on_train=True, path=None):
    ipw_asmds = generate_weighted_asmds_leaf(
        X, t, bootstrap_matrix,
        IPW(LogisticRegression(penalty="none", solver='saga', max_iter=500)),
        on_train
    )
    ipw_asmds.rename(columns={'depth': 'Depth', 'covariate': 'Covariate', 'weighted': 'Weighted ASMD'},
                          inplace=True)
    ipw_asmds = ipw_asmds.groupby(["Covariate"])["Weighted ASMD"].mean().reset_index()
    ipw_asmds = ipw_asmds.loc[ipw_asmds["Covariate"].isin(features_to_plot)]

    plot_data = pd.concat({"Tree": weighted_asmds, "IPW": ipw_asmds}, names=["Model"]).reset_index()

    fig, ax = plt.subplots()

    ax = sns.lineplot(plot_data, x="Depth", y="Weighted ASMD", hue="Covariate", legend=False, style="Covariate")  #, style="Covariate")
    ax.tick_params(axis='both', which='both', labelsize=12)
    ax.set_xlabel(xlabel='Tree maximum depth', fontsize=13)
    ax.set_ylabel(ylabel='Weighted ASMD', fontsize=13)

    fig.show()
    fig.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=600)
    return fig


def generate_calibration_data2(bootstrap_results, truth, mode, t=None, n_bins=10):
    truth_test = truth.reset_index()
    truth_test.rename(columns={'index': 'X_index'}, inplace=True)

    if mode == 'y':

        factual = []
        for cols, data in bootstrap_results.groupby(["model", "bootstrap_number"]):
            data = data.set_index("X_index")
            tt = t.loc[data.index].astype(int)
            data = data.drop(columns=["model", "bootstrap_number"])
            data = data.rename(columns={"y0": 0, "y1": 1})
            datanp = data.values
            datanp = datanp[np.arange(datanp.shape[0]), tt]
            data = pd.DataFrame(datanp, index=data.index).reset_index()
            data = data.rename(columns={0: "y"})
            data["model"] = cols[0]
            data["bootstrap_number"] = cols[1]
            factual.append(data)
        factual = pd.concat(factual)
        bootstrap_results = factual
    bootstrap_results_xtest = pd.merge(truth_test, bootstrap_results, on='X_index', how="inner")

    if mode == 'propensity_score':
        column_true_name, column_pred_name = 't', mode
    else:
        column_true_name, column_pred_name = 'y_x', 'y_y'  # 'D', 'y'

    calibration_curves = []
    for name, d in bootstrap_results_xtest.groupby(["model", "bootstrap_number"]):
        prob_true, prob_pred = calibration_curve(
            d[column_true_name], d[column_pred_name],
            n_bins=n_bins,
            strategy="uniform",
        )
        cur_res = pd.DataFrame({
            "model": name[0],
            "bootstrap": name[1],
            "pred": prob_pred,
            "true": prob_true,
        })
        calibration_curves.append(cur_res)
    calibration_curves = pd.concat(calibration_curves)
    calibration_curves

    smooth_calibration_curves = []
    use_lowess = False
    for name, d in bootstrap_results_xtest.groupby(["model", "bootstrap_number"]):
        if use_lowess:
            n_bins = 50
            prob_pred_new = np.linspace(0, 1 + 1e-8, num=n_bins)
            prob_true_new = sm.nonparametric.lowess(
                d[column_true_name], d[column_pred_name],
                frac=0.5,
                xvals=prob_pred_new,
            )

        else:
            n_bins = 10
            prob_true, prob_pred = calibration_curve(
                d[column_true_name], d[column_pred_name],
                n_bins=n_bins,
                strategy="uniform",
            )
            f_interp = interp1d(
                prob_pred, prob_true,
                bounds_error=False,
                fill_value="extrapolate",
            )
            m_prop = bootstrap_results_xtest.loc[
                bootstrap_results_xtest["model"] == name[0], column_pred_name].quantile([0.001, 0.999])
            prob_pred_new = np.linspace(
                m_prop.iloc[0], m_prop.iloc[1] + 1e-6,
                num=n_bins
            )
            prob_true_new = f_interp(prob_pred_new)

        cur_res = pd.DataFrame({
            "model": name[0],
            "bootstrap": name[1],
            "pred": prob_pred_new,
            "true": prob_true_new,
        })
        smooth_calibration_curves.append(cur_res)
    smooth_calibration_curves = pd.concat(smooth_calibration_curves)

    avg_smoothed_calibration_curves = smooth_calibration_curves.groupby(["model", "pred"])["true"].mean()
    avg_smoothed_calibration_curves = avg_smoothed_calibration_curves.reset_index()
    avg_smoothed_calibration_curves

    return avg_smoothed_calibration_curves, calibration_curves


def plot_calibration_curve_2(avg_smoothed_calibration_curves, calibration_curves, path=None, use_legend=False):
    n_models = len(calibration_curves["model"].unique())
    n_rows = int(np.ceil(n_models / 2))
    fig, axes = plt.subplots(n_rows, 2, figsize=(8, 4), sharey=True)
    use_legend = False

    for i, model in enumerate(calibration_curves["model"].unique()):
        row = int(np.floor(i / 2))
        col = int(np.ceil(i / 2) - np.floor(i / 2))
        if n_rows == 1:
            axes[col].plot(
                [0, 1], [0, 1],
                linestyle="--", color="dimgrey",
                label="Optimal",
            )
        else:
            axes[row, col].plot(
                [0, 1], [0, 1],
                linestyle="--", color="dimgrey",
                label="Optimal",
            )
        data_model = calibration_curves.query("model==@model")
        for boot, d in data_model.groupby("bootstrap"):
            if n_rows == 1:
                axes[col].plot(
                    d["pred"], d["true"],
                    linewidth=0.75,
                    marker=".", markersize=2,
                    alpha=0.15, color="#919090",
                    label="Subsamples" if boot == 0 else None,
                )
            else:
                axes[row, col].plot(
                    d["pred"], d["true"],
                    linewidth=0.75,
                    marker=".", markersize=2,
                    alpha=0.15, color="#919090",
                    label="Subsamples" if boot == 0 else None,
                )

        data_model = avg_smoothed_calibration_curves.query("model==@model")
        if n_rows == 1:
            axes[col].plot(
                data_model["pred"], data_model["true"],
                linewidth=2, color="#062f80",
                label="Average",
            )
        else:
            axes[row, col].plot(
                data_model["pred"], data_model["true"],
                linewidth=2, color="#062f80",
                label="Average",
            )

        title = model
        if n_rows == 1:
            axes[col].set_title(title)
            axes[col].set_xlabel("Predicted probability")
        else:
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel("Predicted probability")
    if n_rows == 1:
        axes[0].set_ylabel("True probability")
        axes[0].legend(loc="best", fontsize=8)
        axes[1].legend(loc="best", fontsize=8)
    else:
        for row_i in range(n_rows):
            axes[row_i, 0].set_ylabel("True probability")
            for col_i in range(2):
                axes[row_i, col_i].legend(loc="best", fontsize=8)

    if path is not None:
        fig.savefig(path)
    fig.subplots_adjust(hspace=0.8)
    fig.show()

    return fig


def save_tree_explains(X, a, fitted_models, path=None, verbose=False):
    for model_id, model in fitted_models.items():
        model_name, bootstrap_number = model_id[0], model_id[1]
        if bootstrap_number == 0:
            tree_list = model.explain(X, a)
            if verbose:
                print(tree_list)
            tree_df = pd.DataFrame(tree_list)
            if path is not None:
                tree_df.to_csv(path_or_buf=path + '_' + str(model_name) + '_tree_paths_df.csv')

    return tree_df

