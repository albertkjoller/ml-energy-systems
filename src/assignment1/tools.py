from typing import Optional
from collections import defaultdict, Counter
from pprint import  pprint

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression, f_regression

from .linear_regression import ClosedFormLinearRegression, LocallyWeightedLinearRegression

import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import timedelta
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# feature selection
def select_features(Xtrain_, ytrain_, Xtest_, n_features):
	# configure to select a subset of features
	fs = SelectKBest(score_func=mutual_info_regression, k=n_features)
	# learn relationship from training data
	fs.fit(Xtrain_, ytrain_)
	# transform train input data
	Xtrain_fs = fs.transform(Xtrain_)
	# transform test input data
	Xtest_fs = fs.transform(Xtest_)
	return Xtrain_fs, Xtest_fs, fs

def min_max_normalization(Xtrain, Xtest, cont_attr, a=0, b=1):
    # Compute min and max of train split 
    min_train, max_train        = Xtrain[cont_attr].min(axis=0), Xtrain[cont_attr].max(axis=0)
    # Normalize train and test splits
    Xtrain[cont_attr]           = a + (Xtrain[cont_attr] - min_train) * (b-a) / (max_train - min_train)
    Xtest[cont_attr]            = a + (Xtest[cont_attr] - min_train) * (b-a) / (max_train - min_train)
    return Xtrain, Xtest

def get_train_test_split(X, y, timestamps, split_type: str = 'fraction', test_size: Optional[float] = None, fold: Optional[int] = None):
    assert test_size != fold, 'Either test_size or fold must be specified'

    # Split data into train and test by using last x days as test set
    if split_type == 'fraction':
        assert test_size is not None, 'test_size must be specified'
        
        N                                   = len(X)
        Xtrain, Xtest                       = X[:int(N * ( 1- test_size ))].reset_index(drop=True), X[int(N * ( 1- test_size )):].reset_index(drop=True)
        ytrain, ytest                       = y[:int(N * ( 1- test_size ))].reset_index(drop=True), y[int(N * ( 1- test_size )):].reset_index(drop=True)
        timestamps_train                    = timestamps[:int(N * ( 1- test_size))]

    elif split_type == 'inner_fold':
        assert fold is not None, 'fold must be specified'
        X = X.reset_index(drop=True); y = y.reset_index(drop=True); # timestamps = timestamps.reset_index(drop=True)

        test_idxs                           = np.logical_and(timestamps > timestamps.max() - timedelta(days=fold), timestamps <= timestamps.max() - timedelta(days=(fold-1)))
        Xtrain, Xtest                       = X[:test_idxs[test_idxs != False].index[0]], X[test_idxs]
        ytrain, ytest                       = y[:test_idxs[test_idxs != False].index[0]], y[test_idxs]
        timestamps_train                    = timestamps[:test_idxs[test_idxs != False].index[0]]
    
    elif split_type == 'outer_fold':
        assert fold is not None, 'fold must be specified'
        X = X.reset_index(drop=True); y = y.reset_index(drop=True); timestamps = timestamps.reset_index(drop=True)

        test_idxs                           = np.logical_and(timestamps > timestamps.max() - timedelta(days=fold), timestamps <= timestamps.max() - timedelta(days=(fold-1)))
        Xtrain, Xtest                       = X[:test_idxs[test_idxs != 0].index[0]], X[test_idxs]
        ytrain, ytest                       = y[:test_idxs[test_idxs != 0].index[0]], y[test_idxs]
        timestamps_train                    = timestamps[:test_idxs[test_idxs != 0].index[0]]

    return Xtrain.reset_index(drop=True), ytrain.reset_index(drop=True), Xtest.reset_index(drop=True), ytest.reset_index(drop=True), timestamps_train.reset_index(drop=True)

class CrossValidation:

    def __init__(self, X, y, timestamps, cont_attr, a=-1, b=1, lambda_=0.0, use_num_features=None, base_model=ClosedFormLinearRegression, CV_range_inner=range(1, 52), CV_range_outer=range(1, 52), store_ytest=False):
        # Initialize class-wide variables
        self.X                  = X
        self.y                  = y
        self.timestamps         = timestamps
        self.cont_attr          = cont_attr
        self.base_model         = base_model
        self.CV_range_inner     = CV_range_inner
        self.CV_range_outer     = CV_range_outer
        self.a, self.b          = a, b
        self.lambda_            = lambda_
        self.use_num_features   = use_num_features
        self.results_inner      = defaultdict(lambda: defaultdict(list))
        self.results_outer      = defaultdict(lambda: defaultdict(list))
        self.predictions        = defaultdict(lambda: defaultdict(dict))
        self.results_latex      = pd.DataFrame([])
        self.store_ytest        = store_ytest

    def store_results(self, param_name, ytest, ypred, mode='inner'):
        if mode == 'inner':
            # Compute MAE and RMSE
            self.results_inner['MAE'][param_name].append(mean_absolute_error(ytest, ypred))
            self.results_inner['MSE'][param_name].append(mean_squared_error(ytest, ypred))
            self.results_inner['RMSE'][param_name].append(np.sqrt(mean_squared_error(ytest, ypred)))
            self.results_inner['R2'][param_name].append(r2_score(ytest, ypred))
        
        elif mode == 'outer':
            # Compute MAE and RMSE
            self.results_outer['MAE'][param_name].append(mean_absolute_error(ytest, ypred))
            self.results_outer['MSE'][param_name].append(mean_squared_error(ytest, ypred))
            self.results_outer['RMSE'][param_name].append(np.sqrt(mean_squared_error(ytest, ypred)))
            self.results_outer['R2'][param_name].append(r2_score(ytest, ypred))

    def no_hyperparameters(self, X_, y_, timestamps_, outer_fold, seed=0):
        # Loop through various fractions
        np.random.seed(seed)

        for fold in tqdm(self.CV_range_inner, desc=f'OUTER FOLD {outer_fold}/{max(self.CV_range_outer)}) INNER CV for no hyperparameter...'):
            # Get split for current CV fold
            Xtrain_, ytrain_, Xtest_, ytest_, _ = get_train_test_split(X_, y_, timestamps_, split_type='inner_fold', fold=fold)

            # Normalize continuous attributes
            Xtrain_, Xtest_   = min_max_normalization(Xtrain_, Xtest_, self.cont_attr, a=self.a, b=self.b)

            # Fit model and get predictions
            model = self.base_model(regularization='ridge', lambda_=0.0)
            model.fit(Xtrain_.reset_index(drop=True), ytrain_.reset_index(drop=True))
            ypred_ = model.predict(Xtest_.reset_index(drop=True))
            
            # Compute and store metrics
            self.store_results(f'Hyperparam=None', ytest_, ypred_, mode='inner')

        return [pd.DataFrame.from_dict({name_: res_ for (name_, res_) in self.results_inner[score_].items() if 'Hyperparam' in name_}) for score_ in ['MAE', 'MSE', 'RMSE', 'R2']]

    def dataset_size(self, fractions, X_, y_, timestamps_, outer_fold, seed=0):
        # Loop through various fractions
        for j, frac in enumerate(fractions):
            np.random.seed(seed)

            for fold in tqdm(self.CV_range_inner, desc=f'OUTER FOLD {outer_fold}/{max(self.CV_range_outer)} - {j+1}/{len(fractions)}) INNER CV for fraction == {frac}...'):
                # Get split for current CV fold
                Xtrain_, ytrain_, Xtest_, ytest_, _ = get_train_test_split(X_, y_, timestamps_, split_type='inner_fold', fold=fold)

                # Subsample training set
                temp_               = pd.concat([Xtrain_, ytrain_], axis=1).sample(frac=frac).reset_index(drop=True)
                Xtrain_, ytrain_    = temp_.iloc[:, :-1], temp_.iloc[:, -1]

                # Normalize continuous attributes
                Xtrain_, Xtest_   = min_max_normalization(Xtrain_, Xtest_, self.cont_attr, a=self.a, b=self.b)

                # Fit model and get predictions
                model = self.base_model(regularization='ridge', lambda_=self.lambda_)
                model.fit(Xtrain_.reset_index(drop=True), ytrain_.reset_index(drop=True))
                ypred_ = model.predict(Xtest_.reset_index(drop=True))
                
                # Compute and store metrics
                self.store_results(f'Fraction={frac}', ytest_, ypred_, mode='inner')

        return [pd.DataFrame.from_dict({name_: res_ for (name_, res_) in self.results_inner[score_].items() if 'Fraction' in name_}) for score_ in ['MAE', 'MSE', 'RMSE', 'R2']]

    def feature_selection(self, n_features_list, X_, y_, timestamps_, outer_fold, seed=0):
        # Loop through various fractions
        for j, n_features in enumerate(n_features_list):
            np.random.seed(seed)

            for fold in tqdm(self.CV_range_inner, desc=f'OUTER FOLD {outer_fold}/{max(self.CV_range_outer)} - {j+1}/{len(n_features_list)}) Running CV for n_features == {n_features}...'):
                # Get split for current CV fold
                Xtrain_, ytrain_, Xtest_, ytest_, _ = get_train_test_split(X_, y_, timestamps_, split_type='inner_fold', fold=fold)

                # Run feature selection algorithm
                _, _, fs                    = select_features(Xtrain_, ytrain_, Xtest_, n_features=n_features)
                current_features            = Xtrain_.columns[fs.get_support()]
                Xtrain_, Xtest_             = Xtrain_[current_features], Xtest_[current_features]

                # Normalize continuous attributes
                cont_attr_                  = np.intersect1d(list(current_features), self.cont_attr)
                Xtrain_, Xtest_             = min_max_normalization(Xtrain_, Xtest_, cont_attr_, a=self.a, b=self.b)

                # Fit model and get predictions
                model = self.base_model(regularization='ridge', lambda_=self.lambda_)
                model.fit(Xtrain_, ytrain_)
                ypred_ = model.predict(Xtest_)
                
                # Compute and store metrics
                self.store_results(f'N_features={n_features}', ytest_, ypred_)

        return [pd.DataFrame.from_dict({name_: res_ for (name_, res_) in self.results_inner[score_].items() if 'N_features' in name_}) for score_ in ['MAE', 'MSE', 'RMSE', 'R2']]
    
    def l2_regularization(self, lambdas, X_, y_, timestamps_, outer_fold, seed=0):
        # Loop through various fractions
        for j, lambda_ in enumerate(lambdas):
            np.random.seed(seed)

            for fold in tqdm(self.CV_range_inner, desc=f'OUTER FOLD {outer_fold}/{max(self.CV_range_outer)} - {j+1}/{len(lambdas)}) INNER CV for lambda == {lambda_}...'):
                # Get split for current CV fold
                Xtrain_, ytrain_, Xtest_, ytest_, _  = get_train_test_split(X_, y_, timestamps_, split_type='inner_fold', fold=fold)

                # Normalize continuous attributes
                Xtrain_, Xtest_   = min_max_normalization(Xtrain_, Xtest_, self.cont_attr, a=self.a, b=self.b)

                # Fit model and get predictions
                model = self.base_model(regularization='ridge', lambda_=lambda_)
                model.fit(Xtrain_, ytrain_)
                ypred_ = model.predict(Xtest_)
                
                # Compute and store metrics
                self.store_results(f'Lambda={lambda_}', ytest_, ypred_, mode='inner')

        return [pd.DataFrame.from_dict({name_: res_ for (name_, res_) in self.results_inner[score_].items() if 'Lambda' in name_}) for score_ in ['MAE', 'MSE', 'RMSE', 'R2']]

    def lengthscale_locally_weighted(self, X_, y_, timestamps_, lengthscales, outer_fold, seed=0):
        # Loop through various fractions
        for j, sigma in enumerate(lengthscales):
            np.random.seed(seed)
        
            for fold in tqdm(self.CV_range_inner, desc=f'OUTER FOLD {outer_fold}/{max(self.CV_range_outer)} - {j+1}/{len(lengthscales)}) INNER CV for sigma == {sigma:.3f}...'):
                # Get split for current CV fold
                Xtrain_, ytrain_, Xtest_, ytest_, _ = get_train_test_split(X_, y_, timestamps_, split_type='inner_fold', fold=fold)

                # Normalize continuous attributes
                Xtrain_, Xtest_   = min_max_normalization(Xtrain_, Xtest_, self.cont_attr, a=self.a, b=self.b)

                # Fit model and get predictions
                model = LocallyWeightedLinearRegression(sigma=sigma, lambda_=self.lambda_, kernel='Gaussian')
                model.fit(Xtrain_, ytrain_)
                ypred_ = model.predict(Xtest_)
                
                # Compute and store metrics
                self.store_results(f'Sigma={sigma}', ytest_, ypred_, mode='inner')
        
        return [pd.DataFrame.from_dict({name_: res_ for (name_, res_) in self.results_inner[score_].items() if 'Sigma' in name_}) for score_ in ['MAE', 'MSE', 'RMSE', 'R2']]

    def kernel_type(self, kernels, seed=0):
        pass

    def run(self, inner_loop='dataset_size', seed=0, **kwargs):
        np.random.seed(seed)

        self.best_lambda_               = self.lambda_
        self.best_cont_attr_            = self.cont_attr

        train_set_sizes = []
        self.attr_used  = []
        for fold in self.CV_range_outer:

            # Get split for current CV fold
            Xtrain, ytrain, Xtest, ytest, timestamps_   = get_train_test_split(self.X, self.y, self.timestamps, split_type='outer_fold', fold=fold)
            if self.use_num_features is not None:
                _, _, fs                                = select_features(Xtrain, ytrain, Xtest, n_features=self.use_num_features)
                current_features                        = Xtrain.columns[fs.get_support()]
                Xtrain, Xtest                           = Xtrain[current_features], Xtest[current_features]

                # Normalize continuous attributes
                cont_attr_                              = np.intersect1d(list(current_features), self.cont_attr)
                Xtrain, Xtest                           = min_max_normalization(Xtrain, Xtest, cont_attr_, a=self.a, b=self.b)

            train_set_sizes.append(Xtrain.__len__())
            self.attr_used.append(list(Xtrain.columns))

            if inner_loop == 'no_hyperparam':
                param_name = 'no_hyperparam'
                _, _, RMSE, _ = self.results_outer['no_hyperparam'][fold] = self.no_hyperparameters(X_=Xtrain.copy(), y_=ytrain.copy(), timestamps_=timestamps_, outer_fold=fold)
            elif inner_loop == 'dataset_size':
                param_name = 'dataset_size'
                _, _, RMSE, _ = self.results_outer['dataset_size'][fold] = self.dataset_size(X_=Xtrain.copy(), y_=ytrain.copy(), timestamps_=timestamps_, outer_fold=fold, fractions=kwargs['fractions'])
            elif inner_loop == 'feature_selection':
                param_name = 'feature_selection'
                _, _, RMSE, _ = self.results_outer['feature_selection'][fold] = self.feature_selection(X_=Xtrain.copy(), y_=ytrain.copy(), timestamps_=timestamps_, outer_fold=fold, n_features_list=kwargs['n_features_list'])
            elif inner_loop == 'l2_regularization':
                param_name = 'l2_regularization'
                _, _, RMSE, _ = self.results_outer['lambdas'][fold] = self.l2_regularization(X_=Xtrain.copy(), y_=ytrain.copy(), timestamps_=timestamps_, outer_fold=fold, lambdas=kwargs['lambdas'])
            elif inner_loop == 'lengthscale_locally_weighted':
                param_name = 'lengthscale_locally_weighted'
                _, _, RMSE, _ = self.results_outer['lengthscale_locally_weighted'][fold] = self.lengthscale_locally_weighted(X_=Xtrain.copy(), y_=ytrain.copy(), timestamps_=timestamps_, outer_fold=fold, lengthscales=kwargs['lengthscales'])
            elif inner_loop == 'kernel_type':
                self.results_outer[fold] = self.kernel_type(**kwargs)
            else:
                raise ValueError(f'Invalid inner loop: {inner_loop}')

            RMSE                            = RMSE.mean(axis=0)
            if inner_loop == 'dataset_size':        
                best_frac                   = float(RMSE.index[RMSE.argmin()].split("=")[-1])        
                temp_                       = pd.concat([Xtrain, ytrain], axis=1).sample(frac=best_frac)
                Xtrain, ytrain              = temp_.iloc[:, :-1].reset_index(drop=True), temp_.iloc[:, -1].reset_index(drop=True)

            if inner_loop == 'feature_selection':
                best_param                  = int(RMSE.index[RMSE.argmin()].split("=")[-1])
                _, _, fs                    = select_features(Xtrain, ytrain, Xtest, n_features=best_param)
                current_features            = Xtrain.columns[fs.get_support()]
                Xtrain, Xtest               = Xtrain[current_features], Xtest[current_features]
                self.best_cont_attr_        = np.intersect1d(list(current_features), self.cont_attr)
            if inner_loop == 'l2_regularization':
                self.best_lambda_           = float(RMSE.index[RMSE.argmin()].split("=")[-1])

            # Normalize continuous attributes
            Xtrain, Xtest   = min_max_normalization(Xtrain, Xtest, self.best_cont_attr_, a=self.a, b=self.b)

            # Define model
            if inner_loop == 'lengthscale_locally_weighted':
                best_lengthscale            = float(RMSE.index[RMSE.argmin()].split("=")[-1])
                model                       = self.base_model(sigma=best_lengthscale, kernel='Gaussian')
            else:
                model = self.base_model(regularization='ridge', lambda_=self.best_lambda_)

            # Fit model and get predictions
            model.fit(Xtrain, ytrain)
            ypred = model.predict(Xtest)

            # Compute and store metrics
            self.store_results(f'Generalization error', ytest, ypred, mode='outer')
            self.predictions[param_name][fold] = (ytest, ypred) if self.store_ytest else ypred

        print(f"\nTraining set sizes: {np.sort(train_set_sizes)}")
        print(f"Features used:")
        pprint(Counter(np.array(self.attr_used).flatten()))

        res_        = pd.DataFrame.from_dict({score_: self.results_outer[score_]['Generalization error'] for score_ in ['MAE', 'MSE', 'RMSE', 'R2']})
        gen_error   = pd.DataFrame(res_.apply(lambda x: f"{x.mean():.4f} \pm {x.std() / np.sqrt(len(res_)):.4f}"), columns=[param_name]).T
        self.results_latex = gen_error if self.results_latex.empty else pd.concat([self.results_latex, gen_error], axis=0)