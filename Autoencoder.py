#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 06:50:08 2018

@author: henrique

Este arquivo é parte da dissertação de mestrado Autoencoders para criação de inferencias na indústria química.

Necessita o acompanhamento do dataset: COL1 ou 2.npy

"""
#%%
import numpy as np
import pandas as pd
import seaborn as sns
# import statsmodels.api as sm
import matplotlib.pyplot as plt
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from os import makedirs, listdir, sep, path
from itertools import compress
from math import ceil
from scipy.stats import f, t
from time import time, localtime, strftime
from keras.layers import Input, Dense, Dropout
from random import shuffle, sample, choice, uniform
from keras.models import Model, load_model, Sequential
from keras.optimizers import RMSprop, SGD, Adadelta, Nadam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, LassoLarsIC, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#%%
# from numpy.random import seed
# seed(18)
# from tensorflow import set_random_seed
# set_random_seed(18)
#%%



def read_data(dataset, feed_composition_as_input=False):
    """
    Lê e carrega o dataset que deve estar salvo em uma pasta chamada data no mesmo local.
    feed_composition_as_input chama considera a composição de entrada das colunas de destilação como entradas e é utilizada para modelagem do sistema.

    Returns
    -------
    Retorna o dataset em pandas, nomes das colunas de entrada e nomes das colunas de saida
    """
    if dataset == 'COL1':
        dataset = pd.read_csv('data'+sep+'COL1.csv')
        dataset.columns = dataset.columns.str.strip()
        dataset.drop([0, 1], inplace=True)
        if feed_composition_as_input:
            compositions = ['YETANO', 'YPROPENO', 'YPROPANO',
                            'YPES', 'XETANO', 'XPROPENO', 'XPROPANO']
            inputs = ['Z propeno', 'F in', 'D/F', 'RR', 'QREF', 'QCOND',
                      'FTOPO', 'FFUNDO', 'TEMPTOPO', 'TEMPFUND', 'DELTAP']
        else:
            compositions = ['Z propeno', 'YETANO', 'YPROPENO',
                            'YPROPANO', 'YPES', 'XETANO', 'XPROPENO', 'XPROPANO']
            inputs = ['F in', 'D/F', 'RR', 'QREF', 'QCOND',
                      'FTOPO', 'FFUNDO', 'TEMPTOPO', 'TEMPFUND', 'DELTAP']
    elif dataset == 'COL2':
        dataset = pd.read_csv('data'+sep+'COL2.csv')
        dataset.columns = dataset.columns.str.strip()
        if feed_composition_as_input:
            compositions = ['ypropym', 'ypropam',
                            'yetanom', 'xpropym', 'xpropam', 'xetanom']
            inputs = ['FEED', 'ZPROPENO', 'ZPROPANO', 'ZETANO', 'RR', 'F/C', 'TOPOMAS',
                      'FUNDOMAS', 'FPRES', 'FTEMP', 'QCOND', 'QREF', 'DELTAP1', 'DELTAP2']
        else:
            compositions = ['ZPROPENO', 'ZPROPANO', 'ZETANO', 'ypropym',
                            'ypropam', 'yetanom', 'xpropym', 'xpropam', 'xetanom']
            inputs = ['FEED', 'RR', 'F/C', 'TOPOMAS', 'FUNDOMAS',
                      'FPRES', 'FTEMP', 'QCOND', 'QREF', 'DELTAP1', 'DELTAP2']
    elif dataset == 'COL3':
        dataset = pd.read_csv('data'+sep+'COL3.csv')
        dataset = dataset.drop(['Unnamed: 0'], axis=1)
        dataset.columns = dataset.columns.str.strip()
        if feed_composition_as_input:
            compositions = ['C11PROPA', 'C11PROPE', 'C15PROPA', 'C15PROPE']
            inputs = ['Vazão', 'Z Propeno', 'B8', 'B4', 'C11MAS', 'C11TEMP', 'C11PRESS', 'C15MAS', 'C15TEMP', 'C14AMASS',
                      'C16AMASS', 'B2WORK', 'B22WORK', 'B6HEAT', 'B21HEAT',
                      'TEMPTOPO', 'TEMPFUND', 'PRESFUND', 'C9VFRAC', 'RETMASS']
        else:
            compositions = ['Z Propeno', 'C11PROPA',
                            'C11PROPE', 'C15PROPA', 'C15PROPE']
            inputs = ['Vazão', 'B8', 'B4', 'C11MAS', 'C11TEMP', 'C11PRESS', 'C15MAS', 'C15TEMP', 'C14AMASS',
                      'C16AMASS', 'B2WORK', 'B22WORK', 'B6HEAT', 'B21HEAT',
                      'TEMPTOPO', 'TEMPFUND', 'PRESFUND', 'C9VFRAC', 'RETMASS']
    elif dataset[:8] == 'dataset_':
        dataset = pd.read_csv('data'+sep+dataset+'.csv')
        inputs = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
        compositions = ['y']
    else:
        print('not found')
    return dataset, inputs, compositions

def _files_in_workspace(where):
    """
    Função auxiliar para escrita de resultados em csvs

    Returns
    -------
    Retorna lista de pastas e arquivos.csvs na pasta where
    """
    _files = listdir(where)
    folders = []
    csvs = []
    for item in _files:
        if path.isdir(where+sep+item) and item[0] != '.':
            folders += [item]
        if item[-3:] == 'csv':
            csvs += [item]
    return folders, csvs

def read_evaluations_from_csv(work_folder, merge_all=True):
    """
    Concatena todos os resultados já escritos em csvs

    merge_all por padrão retorna um dataframe com todos as metricas observadas nos autoencoders,
    se falso, retorna dicionário separando os resultados por pasta
    """
    _, csvs = _files_in_workspace(work_folder)
    dfs = {}
    if len(csvs) == 0:
        print('no results written here')
        return
    # elif len(csvs) > 1:
    #     print(csvs)
    #     chosen = input('Which one? (1 for the first, 2 for the second...\n  ')
    else:
        # chosen=1
        for files in csvs:
            dfs[files[:-4]] = pd.read_csv(work_folder+sep+files)
    if merge_all:
        if len(dfs)==1:
            print('there is only {}'.format(list(dfs.keys())[0]))
            return dfs[list(dfs.keys())[0]]
        elif len(dfs)>1:
            complete_df = pd.DataFrame()
            for name in dfs:
                complete_df = pd.concat([complete_df, dfs[name]])
                complete_df=complete_df.set_index(['model'])
            return complete_df
        else:
            print('how the hell you got here?')
    else:
       return dfs

def old_read_evaluations_from_csv(work_folder, merge_all=True):
    _files = listdir(work_folder)
    csvs = []
    for item in _files:
        if item[-3:] == 'csv':
            csvs += [item]
    dfs = {}
    if len(csvs) == 0:
        print('no results written here')
        return dfs
    else:
        for files in csvs:
            dfs[files[:-4]] = pd.read_csv(work_folder+sep+files)

    if merge_all == True:
        if len(dfs) == 1:
            print('there is only {}'.format(list(dfs.keys())[0]))
            return dfs[list(dfs.keys())[0]]
        elif len(dfs) > 1:
            complete_df = pd.DataFrame()
            for name in dfs:
                complete_df = pd.concat([complete_df, dfs[name]])
            complete_df = complete_df.set_index(['model'])
            return complete_df
        else:
            print('how the hell you got here?')
    return dfs

def load_aes(work_folder, criterium, architecture='', how_many=1):
    dfs = read_evaluations_from_csv(work_folder)
    if architecture:
        path = work_folder+sep+'{}_models'.format(architecture)+sep
        df = dfs['{}_models'.format(architecture)]
    else:
        path = work_folder+sep+'shallow_models'+sep
        df = dfs['shallow_models']
    
    for met_ in ['loss_train', 'loss_test', 'loss_val', 'loss_all_data']:
        df[met_] = df[met_]**(0.5)
        df.rename(columns={met_: met_.replace('loss', 'rmse')}, inplace=True)
    df.set_index(['model'], inplace=True)

    if 'accuracy' in criterium:
        aes_df = df.sort_values([criterium], ascending=False).head(how_many)
    else:
        aes_df = df.sort_values([criterium], ascending=True).head(how_many)

    for name in aes_df.index:
        aes_df.loc[name, 'autoencoder'] = load_model(path+name+'_AUTOENCODER.hdf5')
        aes_df.loc[name, 'encoder'] = load_model(path+name+'_ENCODER.hdf5')

    return aes_df

def plot_ae_reconstruction(ae_model, data, scatter='', subtracted=''):
    ae_result = pd.DataFrame(ae_model.predict(data), columns=data.columns, index=data.index)
    if subtracted:
        residual = ae_result-data
        for iterator in range(residual.shape[1]):
            ax = plt.subplot(3, 4, iterator+1)
            if scatter:
                plt.scatter(range(len(residual)),abs(residual.values[:,iterator]), color = 'b', s = 7)
                plt.title(residual.columns[iterator])
            else:
                plt.plot(abs(residual.values[:,iterator]))
                plt.title(residual.columns[iterator])
    else:
        for iterator in range(ae_result.shape[1]):
            ax = plt.subplot(3, 4, iterator+1)
            if scatter:
                plt.scatter(range(len(data)),data.values[:,iterator], color = 'k', s = 5)
                plt.scatter(range(len(data)),ae_result.values[:,iterator], color = 'b', s = 7)
                plt.title(data.columns[iterator])
            else:
                plt.plot(data.values[:,iterator], color = 'k')
                plt.plot(ae_result.values[:,iterator], color = 'b')
                plt.title(data.columns[iterator])

    # pylab.savefig(model_to_load+'.png')
    # https://stackoverflow.com/questions/24116318/how-to-show-residual-in-the-bottom-of-a-matplotlib-plot

def plot_shallow_ae_weights(ae_model):
    weights_encoder = pd.DataFrame(ae_model.get_weights()[0],
                                    index=inputs,
                                    columns=['n{}'.format(x+1) for x in range(encoded_size)])
    # bias_encoder = ae_model.get_weights()[1]

    weights_decoder = pd.DataFrame(ae_model.get_weights()[2],
                                    index=['n{}'.format(x+1) for x in range(encoded_size)],
                                    columns=inputs)
    # bias_decoder = ae_model.get_weights()[3]

    sns.heatmap(weights_encoder.transpose(),
                cmap="coolwarm",
                center=0,
                annot=True,
                fmt=".1f",
                linewidth=.5)
    sns.heatmap(weights_decoder,
                cmap="coolwarm",
                center=0,
                annot=True,
                fmt=".1f",
                linewidth=.5)

def _bic(sse, n, k):
    return n*np.log(sse/n)+k*np.log(n)

def _aic(sse, n, k):
    return 2*k - 2*np.log(sse)

def linear_model_from_encoded(encoded, data_treated, identifier, y_name):
    # identifier - train, test or val como string
    # encoder = load_model(mainpath+sep+folder+sep+model+'_ENCODER.hdf5')
    x =  pd.DataFrame(encoded.predict(data_treated['test_x'].values))
    x2 = sm.add_constant(x)

    y_real = list(data_treated['test_y'][y_name])
    log10_y_real = np.log10(y_real)
    ln_y_real = np.log(y_real)
    
    ys = [y_real, log10_y_real, ln_y_real]
    ty = ['normal', 'log10', 'ln']
    models = pd.DataFrame(columns=['None', 'Lasso', 'Ridge'])
    for y, t  in zip(ys, ty):
        model = sm.OLS(y, x)
        result = model.fit()
        print(result.summary())
        result_lasso = model.fit_regularized(L1_wt=1.0)
        print(result_lasso.summary())
        result_ridge = model.fit_regularized(L1_wt=0.0)
        print(result_ridge.summary())
        models.loc[t] = [result, result_lasso, result_ridge]


            # if rest=='LassoLars':
            #     lin_model = linear_model.LassoLarsIC()
            # elif rest=='Ridge':
            #     lin_model = linear_model.Ridge()
            # else:
            #     lin_model = linear_model.LinearRegression()

            # lin_model.fit(encoder_results, y_real)
            # y_hat = lin_model.predict(encoder_results)

            # rss = sum((y_real-y_hat)**2)
            # n_samples = data_treated['{}_x'.format(identifier)].shape[0]
            # n_params = sum(1 for x in lin_model.coef_ if x != 0)
            # R2 = r2_score(y_real, y_hat)
            # MSE = mean_squared_error(y_real, y_hat)
            # RMSE = np.sqrt(mean_squared_error(y_real, y_hat))
            # AIC = _aic(rss, n_samples, n_params)
            # BIC = _bic(rss, n_samples, n_params)

            # print('coef: {}'.format(lin_model.coef_))
            # print('R2: {}'.format(R2))
            # print('mse: {}'.format(MSE))
            # print('RMSE: {}'.format(RMSE))
            # print('AIC: {}'.format(AIC))
            # print('BIC: {}'.format(BIC))

    # if plot_chart:
    # plt.figure()
    # plt.scatter(y_real, y_hat)
    # plt.plot(y_real,y_real)
    # plt.title('Target: {} Data Type: {}  Restriction: {}'.format(y_name, identifier, 'none'))
    # print('Data Type: {} Model Type: {} \n'.format(identifier, rest))
    # return [y_real, y_hat, lin_model, mean_squared_error(y_real, y_hat)]
    # return [BIC, R2, RMSE, rest, lin_model]
    return models

def _plot2(lm_, lm_LL, x, y):
    plt.figure(figsize=(13, 5), dpi=80, facecolor='w', edgecolor='k')

    plt.subplot(1, 2, 1)
    plt.plot(lm_.predict(x), y, 'o')
    plt.plot(y, y, 'k')
    plt.xlabel('Predictions')
    plt.ylabel('Target Values')
    # plt.title('Linear Regression')

    plt.subplot(1, 2, 2)
    plt.plot(lm_LL.predict(x), y, 'o')
    plt.plot(y, y, 'k')
    plt.xlabel('Predictions')
    plt.ylabel('Target Values')
    # plt.title('Lasso-Lars')

def _plots_and_stats(lm_, lm_LL, lm_RG, x, y):
    
    plt.figure(figsize=(18, 5), dpi=80, facecolor='w', edgecolor='k')

    plt.subplot(1, 3, 1)
    plt.plot(lm_.predict(x), y, 'ko')
    plt.plot(y, y, 'k')
    plt.xlabel('Predictions')
    plt.ylabel('Real Values')
    plt.title('Linear Regression')

    plt.subplot(1, 3, 2)
    plt.plot(lm_LL.predict(x), y, 'ko')
    plt.plot(y, y, 'k')
    plt.xlabel('Predictions')
    plt.ylabel('Real Values')
    plt.title('Lasso-Lars')

    plt.subplot(1, 3, 3)
    plt.plot(lm_RG.predict(x), y, 'ko')
    plt.plot(y, y, 'k')
    plt.xlabel('Predictions')
    plt.ylabel('Real Values')
    plt.title('Ridge')

    plt.show()

    for lms in [lm_, lm_LL, lm_RG]:
        _stats_lm(lms, x, y)

def _stats_lm(lm, x_test, y_test):
    params = np.append(lm.intercept_, lm.coef_)
    predictions = lm.predict(x_test)

    newX = pd.DataFrame({"Constant": np.ones(len(x_test))}).join(pd.DataFrame(x_test))
    sse = sum((y_test-predictions)**2)

    MSE = (sse)/(len(newX)-len(newX.columns))
    MSLE = sum((np.log(y_test+1)-np.log(predictions+1))**2)/(len(newX)-len(newX.columns))

    var_b = MSE*(np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params / sd_b

    p_values = [2*(1-t.cdf(np.abs(i), (len(newX)-1))) for i in ts_b]

    sd_b = np.round(sd_b, 3)
    ts_b = np.round(ts_b, 3)
    p_values = np.round(p_values, 3)
    params = np.round(params, 4)

    n_params = sum(1 for x in params if x != 0)
    n_samples = x_test.shape[0]

    BIC = n_samples*np.log(sse/n_samples)+n_params*np.log(n_samples)
    AIC = 2*n_params - 2*np.log(sse)

    myDF3 = pd.DataFrame()
    myDF3["Coefficients"] = params
    myDF3["Std Errors"] = sd_b
    # myDF3["t values"] = ts_b
    myDF3["Probabilites"] = p_values

    print('RMSE: {}'.format(np.sqrt(MSE)))
    print('MSLE {}'.format(MSLE))
    print('R2: {}'.format(lm.score(x_test, y_test)))
    print('AIC: {}'.format(AIC))
    print('BIC: {}'.format(BIC))
    print('_______________________________________________________________')
    print(myDF3)
    # print('coef: {}'.format(params))
    print('_______________________________________________________________')
    # print('np: {}'.format(n_params))

def _linear_model(X, y, rest='', centered=''):
    if centered:
        if rest == 'LassoLars':
            lm = LassoLarsIC(fit_intercept=False)
        elif rest == 'Ridge':
            lm = Ridge(fit_intercept=False)
        else:
            lm = LinearRegression(fit_intercept=False)
    else:
        if rest == 'LassoLars':
            lm = LassoLarsIC()
        elif rest == 'Ridge':
            lm = Ridge()
        else:
            lm = LinearRegression()

    lm.fit(X, y)


    # plt.figure()
    # plt.scatter(y_real, y_hat)
    # plt.plot(y_real,y_real)
    # plt.title('Target: {} Data Type: {}  Restriction: {}'.format(y_name, identifier, 'none'))
    # print('Data Type: {} Model Type: {} \n'.format(identifier, rest))
    # return [y_real, y_hat, lin_model, mean_squared_error(y_real, y_hat)]
    return lm

def _plot_pca(pca):
    plt.figure(figsize=(13, 5), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(range(pca.n_components_),np.cumsum(pca.explained_variance_ratio_), '-o', label = 'variância explicada acumulada')
    plt.bar(range(pca.n_components_),pca.explained_variance_ratio_,width = 0.8, align = 'center', tick_label = ['PC.%s' %i for i in range(1,pca.n_components_+1)])
    #plt.bar(range(pca.n_components_),np.cumsum(pca.explained_variance_ratio_)-pca.explained_variance_ratio_, bottom = pca.explained_variance_ratio_,width = 0.8, align = 'center')
    plt.tick_params(axis='x', which='major', labelsize=8)
    #plt.plot(range(pca.n_components_),np.ones([pca.n_components_])*0.95, c = 'r')
    plt.plot([-0.4,pca.n_components_+1],[0.99, 0.99], c = 'r', ls = '--', label = '99% de variância explicada')
    plt.xlim([-0.4,pca.n_components_+1])
    leg = plt.legend(loc='center right', fontsize = 12)
    leg.get_frame().set_alpha(0.5)
    plt.xlabel('Número de Componentes Principais', size = 13)
    plt.ylabel('Variância Explicada ( % )', size = 13)

class custom_AE(object):

    def plot_history(h):
            plt.plot(h.history['loss'])
            plt.plot(h.history['val_loss'])
            plt.title('AE loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper right')
            plt.show()

            plt.figure()
            plt.plot(h.history['acc'])
            plt.plot(h.history['val_acc'])
            plt.title('AE accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.show()


    def custom_autoencoder(datatreated, architecture, to_plot=False):
        train_x = data_treated['train_x'].values
        test_x = data_treated['test_x'].values
        val_x = data_treated['val_x'].values
        # datatotal = data_treated['datatreated_x'].values

        variables = train_x.shape[1]
        hidden1 = int(architecture[0])
        hidden2 = int(architecture[1])
        activ_ = 'selu'
        wei_inits = 'lecun_uniform'

        autoencoder = Sequential()
        autoencoder.add(Dense(hidden1, input_shape=(variables, ),
                            activation=activ_, kernel_initializer=wei_inits))
        autoencoder.add(Dense(hidden2, activation=activ_,
                            kernel_initializer=wei_inits))
        autoencoder.add(Dense(variables, activation=activ_,
                            kernel_initializer=wei_inits))

        if to_plot:
            autoencoder.summary()

        #declaring encoder as part of the autoencoder
        input_img = Input(shape=(variables,))
        encoder_layer1 = autoencoder.layers[0]
        encoder_layer2 = autoencoder.layers[1]
        encoder = Model(input_img, encoder_layer2(encoder_layer1(input_img)))

        # autoencoder.summary()
        # encoder.summary()

        # stop training criterion
        es = EarlyStopping(monitor='val_loss', min_delta=0.0001,
                        patience=100, verbose=1, mode='auto')
        opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999,
                    epsilon=0.001, schedule_decay=0.004)
        # opt = RMSprop(lr=0.0001, decay=1e-6)

        autoencoder.compile(optimizer=opt,
                            loss='mse',
                            metrics=['accuracy', 'mape', 'msle'])

        h = autoencoder.fit(x=train_x,
                            y=train_x,
                            epochs=10000,
                            batch_size=int(len(train_x)/2),
                            shuffle=True,
                            validation_data=(val_x, val_x),
                            verbose=0,
                            callbacks=[es])
        if to_plot:
            plot_history(h)
        else:
            print(' ')

        results_train = autoencoder.evaluate(
            train_x, train_x, batch_size=len(train_x), verbose=0)
        results_test = autoencoder.evaluate(
            test_x, test_x, batch_size=len(test_x), verbose=0)
        results_val = autoencoder.evaluate(
            val_x, val_x, batch_size=len(val_x), verbose=0)

        # plt.plot()

        # print('input \t\t output')
        # for iter, iter2 in zip(test_x[0], autoencoder.predict(test_x)[0]):
        #     print('{:.4f} / {:.4f}'.format(iter, iter2))

        print('         Loss / Accuracy')
        print(
            'Train: {:.4f} / {:.1f}%'.format(results_train[0], results_train[1]*100))
        print('Val  : {:.4f} / {:.1f}%'.format(results_val[0], results_val[1]*100))
        print(
            'Test : {:.4f} / {:.1f}%'.format(results_test[0], results_test[1]*100))
        model_rmse = results_test[0]**(0.5)
        print('results test rmse: {}'.format(model_rmse))
        print('\n')

        activ0 = activ_
        activ1 = activ_
        loss = 'mse'
        optimizer = 'nadam'
        name = '{}_{}_{}_{}_{}_{}'.format(
            architecture, activ0, activ1, loss, optimizer, wei_inits)

        return [model_rmse, name, autoencoder, encoder, h]


    def multiple_custom_autoencoders(datatreated, architecture, how_many=5):
        results = pd.DataFrame(
            columns=['ae_rmse_test', 'model', 'ae', 'encoder', 'history'])
        for hm in range(how_many):
            ind_result = custom_autoencoder(datatreated, architecture)
            results.loc[hm] = ind_result
        return results


    def multiple_linear_models(datatreated, df_mca='', architecture='757', how_many=5, plotting=False):
        if df_mca:
            pass
        else:
            df_mca = multiple_custom_autoencoders(
                datatreated, architecture, how_many)
        df_mlm = pd.DataFrame(columns=['BIC', 'R2', 'RMSE', 'rest', 'lin_model'])
        for nn in range(df_mca.shape[0]):
            df_mlm.loc[nn] = linear_model_from_encoded(
                df_mca.loc[nn, 'encoder'], datatreated, 'test', 'YPES', rest='None', plot_chart=plotting)
        df = pd.concat([df_mlm, df_mca], axis=1)
        return df

class _dopping(object):
    def manual_dopper(self, df, ratio_of_samples=0.5, number_of_variables=1, error_size=0.1, spec=''):
        noise_index = list(df.sample(frac=ratio_of_samples).index)
        noise_columns = []
        noise = pd.DataFrame(np.ones(df.shape), index=df.index, columns=df.columns)
        for row in noise_index:
            var_ = np.random.choice(df.shape[1], number_of_variables)
            if spec:
                noise.loc[row][spec] = 1+error_size*np.random.choice([-1,1])
            else:
                for col in var_:
                    noise.loc[row][col] = 1+error_size*np.random.choice([-1,1])
            noise_columns += [var_]
        self.noise_index = noise_index
        self.noise_columns = noise_columns
        self.noise = noise
        return noise*df
    
class data_treatment(object):
    def pretreatment(self, dataset, output_columns='', num_clusters=1, hotteling=False, y_rank='', scaler_type='Robust'):
        """
        Pre-treatment handler
        """
        print('---------------------------------')
        print('----------PreTreatment-----------')
        print('---------------------------------')
        print('----------------')
        print('data shape      : {}'.format(dataset.shape))
        print('clusters        : {}'.format(num_clusters))
        print('Y rank          : {}'.format(y_rank))
        print('Scaler          : {}'.format(scaler_type))
        print('T2 Hotteling    : {}'.format(hotteling))
        print('---------------------------------')
        if hotteling:
            self.datatreated = self.hotelling_tsquared(dataset, output_columns, verbose=True)
        else:
            self.datatreated = dataset

        clusters = self.clusters(self.datatreated, output_columns, num_clusters)

        if y_rank != '':
            if len(y_rank) != 4:
                print('You need a valid y_rank! Expected format: \n [%Train, %Test, %Validation , SortingColumnName]')
                return
            [size_train,size_test,size_val,ordinator] = y_rank
        else:
            [size_train,size_test,size_val,ordinator] = [0.60,0.2,0.2,dataset.columns[-1]]

        if size_train+size_test+size_val != 1:
            print('Your proportions are not equal to ONE!')
            return

        train_val = pd.concat([self.y_rank_proportion(clusters[i], ordinator,train = 1-size_test)[0] for i in range(len(clusters))], axis = 0)
        self.test_x = pd.concat([self.y_rank_proportion(clusters[i], ordinator,train = 1-size_test)[1] for i in range(len(clusters))], axis = 0)
        self.train_x, self.val_x = self.y_rank_proportion(train_val, ordinator, size_train/(size_train+size_val))
        if len(self.test_x) == 0:
            print('Zero samples selected for Testing - you need less clusters or more samples!!')
            return

        if output_columns != '':
            self.train_y = self.train_x.loc[:, output_columns]
            self.train_x = self.train_x.drop(output_columns, axis = 1)
            self.test_y = self.test_x.loc[:, output_columns]
            self.test_x = self.test_x.drop(output_columns, axis = 1)
            self.val_y = self.val_x.loc[:, output_columns]
            self.val_x = self.val_x.drop(output_columns, axis = 1)
            self.datatreated_y = self.datatreated.loc[:, output_columns]
            self.datatreated_x = self.datatreated.drop(output_columns, axis = 1)
        else:
            self.datatreated_x = self.datatreated

        if (scaler_type == 'Robust' or scaler_type == 'Standard'):
            self.train_x = self.scaler(self.train_x, type=scaler_type)
            self.test_x = self.scaler(self.test_x, scaler_obj=self.Scaler)
            self.val_x = self.scaler(self.val_x, scaler_obj=self.Scaler)
            self.datatreated_x = self.scaler(self.datatreated_x,
                                             scaler_obj=self.Scaler)
        else:
            print('Data Not Scaled - Scaler not implemented yet')

        if output_columns != '':
            return {'train_x' : self.train_x, 'train_y': self.train_y,
                    'test_x' : self.test_x, 'test_y': self.test_y,
                    'val_x' : self.val_x, 'val_y': self.val_y,
                    'datatreated_x' : self.datatreated_x,
                    'datatreated_y' : self.datatreated_y}
        else:
            return {'train_x' : self.train_x,
                    'test_x' : self.test_x,
                    'val_x' : self.val_x,
                    'datatreated' : self.datatreated_x}

    def clusters(self, dataset, output_columns='', k=1):
        """
        call the function: cluster_1,cluster_2,...,cluster_k = clusters(X,k)\n
        The funcion returns clusters in format of DataFrames\n
        default: k = 1\n
        X is the DataFrame of data\n
        k is the number of clusters\n
        """

        if k > len(dataset):
            print('More clusters than inputs!! - now you have '+str(len(dataset)) + ' clusters!')
            k = len(dataset)-1
        
        if output_columns != '':
            Y = (dataset.drop(output_columns, axis=1)).values
        else:
            Y = dataset.values
        km = KMeans(n_clusters = k, init='k-means++', n_init=10, max_iter=300, tol=1e-05, random_state=10)
        y_km = km.fit_predict(Y)
        Clusters = [dataset.iloc[y_km == i,:] for i in range(k)]
        Clusters = list(compress(Clusters,[len(Clusters[i])>0 for i in range(len(Clusters))]))
        return Clusters

    def scaler(self, dataset, type=None, scaler_obj=None):
        dataset_scaled = dataset.copy()
        if scaler_obj == None:
            if type == 'Standard':
                Scaler = StandardScaler()
                Scaler.fit(dataset)
            elif type == 'Robust':
                Scaler = RobustScaler()
                Scaler.fit(dataset)
            else:
                print('no scaler defined')
            dataset_scaled.iloc[:,:] = Scaler.transform(dataset)
        else:
            if type != None:
                print('scaler_obj passed, ignoring the type input')
            Scaler = scaler_obj
            dataset_scaled.iloc[:,:] = Scaler.transform(dataset)
        self.Scaler = Scaler
        return dataset_scaled

    def hotelling_tsquared(self, dataset, output_columns='', alpha=0.01, to_plot=False, verbose=True):
        data = dataset.copy()
        Scaler = RobustScaler()
        pc = PCA()
        if output_columns != '':
            pca = pc.fit_transform(Scaler.fit_transform(data.drop(output_columns, axis = 1)))
        else:
            pca = pc.fit_transform(Scaler.fit_transform(data))
        x = pca.T
        cov = np.cov(x)
        w = np.linalg.solve(cov, x)
        data['T2'] = (x * w).sum(axis=0)
        (samples, variables) = data.shape
        f_stat = f.ppf(1-alpha,variables,samples-variables)
        t2_limit = variables*(samples-1)/(samples-variables)*f_stat
        dataset_cleared = data[data['T2']<=t2_limit].drop(columns='T2')
        if to_plot:
            with sns.axes_style("whitegrid"):
                plt.figure(figsize=(13, 5), dpi=80, facecolor='w', edgecolor='k')
                plt.plot(data['T2'], '.')
                plt.plot(range(len(pca)),np.ones(len(pca))*t2_limit,
                         label='Limiar T2 Hotelling (99% confiability)',
                         linestyle="--", linewidth=2, color='red')
                plt.legend(loc='best')
                plt.ylabel('T2 Hotelling PCA', fontsize=13)
                plt.xlabel('Samples', fontsize=13)
                plt.show()
        if verbose:
            print('Dropped {} points'.format(samples-dataset_cleared.shape[0]))
        return dataset_cleared

    def y_rank_proportion(self, dataset, y_column, train=0.65):
        """
        Algoritmo para separar as amostras em grupos de calibracao e teste,
         na proporcao desejada e seguindo a metodologia de systematic sampling modificada
        """
        data = dataset.copy()
        data = data.sort_values(by=[y_column], ascending = True)
        if len(data)*train == int(len(data)*train):
            size_train = int(len(data)*train)
        else:
            size_train = int(len(data)*train) + 1
        size_test = len(data) - size_train
        DNA = []
        if len(data) == 1:
            DNA = ['c']
        elif len(data) == 2:
            DNA = ['c','t']
        elif len(data) == 3:
            DNA = ['c','t','c']
        else:
            if size_train == 1:
                size_train = 2
                size_test = size_test - 1
            if size_test == 0:
                size_test = 1
                size_train = size_train - 1
            if size_train < size_test:
                RNA_m = ['c'] + ['t']*ceil(size_test/(size_train - 1))
                RNA_t = RNA_m*(size_train)
                DNA = RNA_t[0:len(data)]
                DNA[-1] = 'c'
                if DNA.count('c') != size_train:
                    auxC = list(loc for loc, val in enumerate(DNA) if val == 'c')
                    counter = -2
                    while DNA.count('c') != size_train:
                        DNA[auxC[counter] -1] = 'c'
                        counter = counter - 1
            if size_train > size_test:
                RNA_m = ['c']*ceil((size_train - 1)/(size_test+1)) + ['t']
                RNA_t = RNA_m*(size_test + 1)
                DNA = RNA_t[0:len(data)]
                DNA[-1] = 'c'
                if DNA.count('c') != size_train:
                    auxC = list(loc for loc, val in enumerate(DNA) if val == 't')
                    counter = -1
                    while DNA.count('c') != size_train:
                        DNA[auxC[counter]-1] = 't'
                        counter = counter - 1
            if size_train == size_test:
                RNA_t = ['c','t']*(size_train - 1)
                DNA = RNA_t + ['t'] + ['c']
        data['indexx'] = DNA
        if len(data) == 1:
            calibracao = data.loc[data['indexx'] == 'c',:].drop('indexx', axis = 1)
            teste = pd.DataFrame()
        elif len(data) == 2:
            calibracao = data.loc[data['indexx'] == 'c',:].drop('indexx', axis = 1)
            teste = data.loc[data['indexx'] == 't',:].drop('indexx', axis = 1)
        else:
            calibracao = data.loc[data['indexx'] == 'c',:].drop('indexx', axis = 1)
            teste = data.loc[data['indexx'] == 't',:].drop('indexx', axis = 1)
        if (len(data.T),) == teste.shape:
            teste = pd.DataFrame(teste).T
        if (len(data.T),) == calibracao.shape:
            calibracao = pd.DataFrame(calibracao).T
        return calibracao, teste

    def pca_dimension_analysis(self, dataset_scaled, output='', to_plot=''):
        pcs = PCA()
        pcs.fit_transform(dataset_scaled, output)
        pcs_vars = pcs.explained_variance_ratio_
        pcs_var_sum = np.cumsum(pcs_vars)
        n = range(pcs_vars.shape[0])
        if to_plot:
            plt.bar(n, pcs_vars)
            plt.scatter(n, pcs_var_sum)
            plt.title('PCs Var')
            plt.ylabel('%')
            plt.xlabel('PCs')
            plt.legend(['train', 'val'], loc='center right')
            plt.show()

def main():
    print('all loaded, now, do something')
#%%
if __name__ == '__main__':
    
    foldername = 'COL1_602020'

    data, inputs, compositions = read_data('COL1')

    data_treated = data_treatment().pretreatment(data,
                                                 output_columns = compositions,
                                                 num_clusters=2,
                                                 hotteling=True,
                                                 y_rank=[0.60, 0.2, 0.2, 'D/F'],
                                                 scaler_type='Standard')

    # COL1 - D/F 
    # COL2 - F/C
    # COL3 - C11TEMP
    # datasets artificiais - x1
    
    main()
#%%