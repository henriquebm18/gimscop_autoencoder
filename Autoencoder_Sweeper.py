#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 06:50:08 2018

@author: henrique

Este arquivo é parte da dissertação de mestrado Autoencoders para criação de inferencias na indústria química.
Funciona com as versões mais recentes do WinPython (3.5 em diante) sem necessidade de instalação de bibliotecas adicionais
Necessita o acompanhamento de datasets; ver função read_data

"""
#%%
import logging
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
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# from sklearn import linear_model

#%%
# from numpy.random import seed
# seed(18)
# from tensorflow import set_random_seed
# set_random_seed(18)
#%%
def writeLog(text, to_see=True):
    """
    Retorna o que está sendo despejado no log ou não
    """
    if to_see:
        print(text)
    
    logger.warning(text)

def defLogger(name, folder):
    """
    Define configuracoes basicas do logger

    Returns
    -------
    Retorna uma instancia do objeto logger
    """
    makedirs(folder, exist_ok=True)
    makedirs(folder+sep+'logs', exist_ok=True)

    LOG_FORMAT = "%(asctime)s - %(message)s"
    logging.basicConfig(filename = folder+sep+'logs'+sep+name,
                        level = logging.WARNING,
                        format = LOG_FORMAT,
                        filemode = 'a')
    logger = logging.getLogger()
    return logger

def read_data(dataset, feed_composition_as_input=False):
    """
    Le e carrega o dataset que deve estar salvo em uma pasta chamada data no mesmo local.
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

class data_treatment(object):
    """
    tratamento dos dados, deve ser feito para cada dataset levando em consideração suas caracteristicas
    """

    def pretreatment(self, dataset, output_columns='', num_clusters=1, hotteling=False, y_rank='', scaler_type='Robust'):
        """
        Pre-treatment handler
        """
        writeLog('---------------------------------')
        writeLog('----------PreTreatment-----------')
        writeLog('---------------------------------')
        writeLog('----------------')
        writeLog('data shape      : {}'.format(dataset.shape))
        writeLog('clusters        : {}'.format(num_clusters))
        writeLog('Y rank          : {}'.format(y_rank))
        writeLog('Scaler          : {}'.format(scaler_type))
        writeLog('T2 Hotteling    : {}'.format(hotteling))
        writeLog('---------------------------------')
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
        km = KMeans(n_clusters = k,init='k-means++',n_init=10,max_iter=300,tol=1e-05,random_state=10)
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
                plt.figure()
                plt.plot(data['T2'])
                plt.plot(range(len(pca)),np.ones(len(pca))*t2_limit,
                         label='Limiar T2 Hotelling (99% confiability)',
                         linestyle="--", linewidth=5, color='red')
                plt.legend(loc='best')
                plt.ylabel('T2 Hotelling PCA', fontsize=13)
                plt.xlabel('Sample', fontsize=13)
                plt.show()
        if verbose:
            writeLog('Dropped {} points'.format(samples-dataset_cleared.shape[0]))
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

class Autoencoder_Sweeper(object):
    def __init__(self, data, work_folder, architecture='shallow', just_eval=''):
        """
        Inicia a varredura dos hiper-parametros do autoencoder, os quais podem ser ajustados e escolhidos abaixo.

        data é um dicionario com o dataset separado em treino, teste e validação;
        work_folder é o nome da pasta onde serão salvos os autoencoders;
        architecture, por padrão treina autoencoders rasos. Passando '525' treina uma rede com 3 camadas internas e dois neuronios no espaço encodado.
        just_eval quando True faz apenas a avaliação dos autoencoders dadas as metricas indicadas aqui, sem treinar AEs
        """
        self.work_folder = work_folder
        makedirs(work_folder, exist_ok=True)
        
        self.metrics = ['accuracy', 'msle', 'mse']

        self.header = ['model', 'hidden_layers', 'optimizer',
                    'loss_train','{}_train'.format(self.metrics[0]),'{}_train'.format(self.metrics[1]),'{}_train'.format(self.metrics[2]),
                    'loss_test','{}_test'.format(self.metrics[0]),'{}_test'.format(self.metrics[1]),'{}_test'.format(self.metrics[2]),
                    'loss_val','{}_val'.format(self.metrics[0]),'{}_val'.format(self.metrics[1]),'{}_val'.format(self.metrics[2]),
                    'loss_all_data','{}_all_data'.format(self.metrics[0]),'{}_all_data'.format(self.metrics[1]),'{}_all_data'.format(self.metrics[2])]
        
        self.datatreated = [data['train_x'].values, data['test_x'].values, data['val_x'].values, data['datatreated_x'].values]
        
        self.losses = ['mse']
        # self.activations = ['relu', 'softplus', 'softsign', 'tanh', 'linear', 'selu', 'hard_sigmoid', 'sigmoid']
        self.activations = ['selu', 'relu', 'tanh', 'linear', 'sigmoid']
        # self.activations = ['selu']
        # self.optimizers = ['SGD', 'RMSprop', 'Adadelta', 'Adam', 'Nadam']
        self.optimizers = ['Nadam']
        # self.optimizers = [RMSprop(lr=0.0001, decay=1e-6)]
        # self.optimizers = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        self.wei_inits = ['lecun_uniform', 'glorot_uniform', 'he_uniform', 'lecun_normal', 'glorot_normal', 'he_normal']
        # self.wei_inits = ['glorot_uniform']
        
        if just_eval:
            self.write_evaluations()
        else:
            if architecture == 'shallow':
                self.shallow_aes_trainer()
            elif len(architecture) == 3:
                self.deep_aes_trainer(architecture)
            else:
                print('architecture not expected')

    def shallow_aes_trainer(self):
        makedirs(self.work_folder+sep+'shallow_models', exist_ok=True)
        train_x = self.datatreated[0]
        val_x = self.datatreated[2]
        variables = train_x.shape[1]

        epochs = 100000
        es = EarlyStopping(monitor='val_loss', min_delta=0, patience=1000, verbose=1, mode='auto')
        # Rplt = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=500, verbose=0)
                
        neurons = range(variables-1, 0, -1)
        neurons = [4]
        
        shuffle(self.activations)

        how_many = len(self.activations)**2*len(self.losses)*len(self.optimizers)*len(neurons)*len(self.wei_inits)
        writeLog('---------------------------------')
        writeLog('----------shallow-models---------')
        writeLog('---------------------------------')
        writeLog('Gridsearch for {} configurations'.format(how_many))
        writeLog('----------------')
        writeLog('metrics observed: {}'.format(self.metrics))
        writeLog('activations     : {}'.format(self.activations))
        writeLog('losses          : {}'.format(self.losses))
        writeLog('optimizers      : {}'.format(self.optimizers))
        writeLog('weigth inits    : {}'.format(self.wei_inits))
        writeLog('---------------------------------')
        start_time = time()
        counter = 0

        # zzz = pd.DataFrame(columns=['ae', 'e', 'h'])

        for neuron in neurons:
            for act1 in self.activations:
                for act2 in self.activations:
                    for opt in self.optimizers:
                        for w_i in self.wei_inits:
                            for loss in self.losses:
                                
                                counter += 1
                                current = '{}_{}_{}_{}_{}_{}'.format(neuron, act1, act2, loss, opt, w_i)
                                print('{} of {} => {}'.format(counter, how_many, current))
                                already_done = listdir(self.work_folder+sep+'shallow_models')

                                if any(current in s for s in already_done):
                                    print('Already Done! Skipped.')
                                else:
                                    time_counter = time()
                                    input_layer = Input(shape = (variables, ))
                                    encoded_layer = Dense(neuron, activation=act1, kernel_initializer=w_i)(input_layer)
                                    output_layer = Dense(variables, activation=act2, kernel_initializer=w_i)(encoded_layer)

                                    encoder = Model(input_layer, encoded_layer)
                                    autoencoder = Model(input_layer, output_layer)

                                    autoencoder.compile(optimizer=opt, loss=loss, metrics=self.metrics)
                                    encoder.compile(optimizer=opt, loss=loss, metrics=self.metrics)
                                    
                                    h = autoencoder.fit(x=train_x,
                                                        y=train_x,
                                                        epochs=epochs,
                                                        batch_size=int(len(train_x)/2),
                                                        shuffle=True,
                                                        validation_data=(val_x, val_x),
                                                        verbose=0,
                                                        callbacks=[es])
                                    print('trained in {}s'.format(int(time()-time_counter)))

                                    autoencoder.save(self.work_folder+sep+'shallow_models'+sep+current+'_AUTOENCODER.hdf5')
                                    encoder.save(self.work_folder+sep+'shallow_models'+sep+current+'_ENCODER.hdf5')
                                    # zzz.iloc[current] = [autoencoder, encoder, h]
                                    # zzz.to_hdf('sweeper.hdf5', key='obj', mode='a')

        lasted_for = time() - start_time
        writeLog('\nperformed in {} minutes'.format(int(lasted_for/60)))

    def deep_aes_trainer(self, architecture):
        makedirs(self.work_folder+sep+'{}_models'.format(architecture), exist_ok=True)
        train_x = self.datatreated[0]
        val_x = self.datatreated[2]
        variables = train_x.shape[1]

        epochs = 100000
        es = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=100, verbose=1, mode='auto')
        # Rplt = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=500, verbose=0)

        hidden1 = int(architecture[0])
        hidden2 = int(architecture[1])
        hidden3 = int(architecture[2])

        shuffle(self.activations)
        
        how_many = len(self.activations)**2*len(self.losses)*len(self.optimizers)*len(self.wei_inits)
        writeLog('---------------------------------')
        writeLog('--------deep-models--------------')
        writeLog('---------------------------------')
        writeLog('Gridsearch for {} configurations'.format(how_many))
        writeLog('----------------')
        writeLog('metrics observed: {}'.format(self.metrics))
        writeLog('activations     : {}'.format(self.activations))
        writeLog('losses          : {}'.format(self.losses))
        writeLog('optimizers      : {}'.format(self.optimizers))
        writeLog('weigth inits    : {}'.format(self.wei_inits))
        writeLog('---------------------------------')

        start_time = time()
        counter = 0

        for act1 in self.activations:
            for act2 in self.activations:
                for opt in self.optimizers:
                    for w_i in self.wei_inits:
                        for loss in self.losses:
                            counter += 1
                            current = '{}_{}_{}_{}_{}_{}'.format(architecture, act1, act2, loss, opt, w_i)
                            print('{} of {} => {}'.format(counter, how_many, current))
                            already_done = listdir(self.work_folder+sep+'{}_models'.format(architecture))

                            if any(current in s for s in already_done):
                                print('Already Done! Skipped.')
                            else:
                                time_counter = time()

                                input_layer = Input(shape=(variables, ))
                                encoding_layer = Dense(hidden1, activation=act1, kernel_initializer=w_i)(input_layer)
                                encoded_layer = Dense(hidden2, activation=act1, kernel_initializer=w_i)(encoding_layer)
                                decoding_layer = Dense(hidden3, activation=act2, kernel_initializer=w_i)(encoded_layer)
                                output_layer = Dense(variables, activation=act2, kernel_initializer=w_i)(decoding_layer)

                                encoder = Model(input_layer, encoded_layer)
                                autoencoder = Model(input_layer, output_layer)

                                autoencoder.compile(optimizer=opt, loss=loss, metrics=self.metrics)

                                autoencoder.fit(x=train_x,
                                                y=train_x,
                                                epochs=epochs,
                                                batch_size=int(len(train_x)/2),
                                                shuffle=True,
                                                validation_data=(val_x, val_x),
                                                verbose=0,
                                                callbacks=[es])
                                inter_time = time()
                                autoencoder.save(self.work_folder+sep+'{}_models'.format(architecture)+sep+current+'_AUTOENCODER.hdf5')
                                encoder.save(self.work_folder+sep+'{}_models'.format(architecture)+sep+current+'_ENCODER.hdf5')
                                writeLog('trained in {} and saved in {}s'.format(int(inter_time-time_counter), int(time()-inter_time)))

        lasted_for = time() - start_time
        writeLog('\nperformed in {} hours and {} minutes'.format(int(lasted_for/3600), int(lasted_for/60)))

    def _evaluate_model(self, model):
        """
        Função auxiliar que carrega o autoencoder e faz a avaliação dos diferentes grupos de treino[0], teste[1] e validação[2] bem como de todo dataset[3] nas metricas observadas no __init__
        """
        try:
            autoencoder = load_model(model)
        except Exception as e:
            writeLog("++++++++++++++++++++++++++++ error at "+str(model)+'\n'+str(e))
            return []
        errors = []
        errors += autoencoder.evaluate(self.datatreated[0],self.datatreated[0],batch_size=len(self.datatreated[0]), verbose=0)
        errors += autoencoder.evaluate(self.datatreated[1],self.datatreated[1],batch_size=len(self.datatreated[1]), verbose=0)
        errors += autoencoder.evaluate(self.datatreated[2],self.datatreated[2],batch_size=len(self.datatreated[2]), verbose=0)
        errors += autoencoder.evaluate(self.datatreated[3],self.datatreated[3],batch_size=len(self.datatreated[3]), verbose=0)
        return errors

    def write_evaluations(self):
        """
        Gera ou completa arquivos csvs com as metricas observadas dos autoencoders treinados.
        chama a _evaluate_model
        """
        folders, csvs = _files_in_workspace(self.work_folder)

        # if len(folders) == 1:
        #     folder = folders[0]
        # else:
        #     print(folders)
        #     _posit = input('Which folder? [1 for the first, 2 for the second and goes on]\n  ')
        #     folder = folders[int(_posit)-1]
        for folder in folders:
            print('\n{} folder:'.format(folder))
            if any(folder+'.csv' in x for x in csvs):
                started = True
                done = list(pd.read_csv(self.work_folder+sep+folder+'.csv')['model'])
                done = [model_name + '_AUTOENCODER.hdf5' for model_name in done]
            else:
                started = False

            ae_models = []
            for item in listdir(self.work_folder+sep+folder):
                if item[-16:] == 'AUTOENCODER.hdf5':
                    ae_models += [item]
            if started:
                ae_models = list(set(ae_models)-set(done))

            if ae_models == []:
                print('all models are already in the csv file, nothing to do here')
            else:
                results = pd.DataFrame(columns = self.header)
                if started:
                    print('appending new results to the csv')
                else:
                    results.to_csv(self.work_folder+sep+folder+'.csv', index=False, mode='a')
                
                counter = 0
                writeLog("Writting {} evaluations to csv".format(len(ae_models)))
                for mod_ in ae_models:
                    start_time = time()
                    _evals = self._evaluate_model(self.work_folder+sep+folder+sep+mod_)
                    if _evals == []:
                        writeLog("++++++++++++++++++++++++++++ error at model {}".format(mod_))
                    else:
                        results.loc[0] = [mod_.replace('_AUTOENCODER.hdf5', ''), mod_.split('_')[0], mod_.split('_')[-2]] + _evals
                        results.to_csv(self.work_folder+sep+folder+'.csv', index=False, header=False, mode='a')
                    counter += 1
                    elapsed_time = time() - start_time
                    print('{}/{} - in {} seconds'.format(counter, len(ae_models), int(elapsed_time)))
                    # if elapsed_time > 3:
                    #     print('__________________')
                    #     restartkernel()
                       

def main():
    inst = Autoencoder_Sweeper(data=data_treated, work_folder=foldername)
    # inst = Autoencoder_Sweeper(data=data_treated, work_folder=foldername, just_eval=True)
    # inst = Autoencoder_Sweeper(data=data_treated, work_folder=foldername, architecture='979')
    # inst = Autoencoder_Sweeper(data=data_treated, work_folder=foldername, architecture='969')
    # inst = Autoencoder_Sweeper(data=data_treated, work_folder=foldername, architecture='959')
    # inst = Autoencoder_Sweeper(data=data_treated, work_folder=foldername, architecture='949')
    # inst = Autoencoder_Sweeper(data=data_treated, work_folder=foldername, architecture='939')
    # inst = Autoencoder_Sweeper(data=data_treated, work_folder=foldername, architecture='878')
    # inst = Autoencoder_Sweeper(data=data_treated, work_folder=foldername, architecture='868')
    # inst = Autoencoder_Sweeper(data=data_treated, work_folder=foldername, architecture='858')
    # inst = Autoencoder_Sweeper(data=data_treated, work_folder=foldername, architecture='848')
    # inst = Autoencoder_Sweeper(data=data_treated, work_folder=foldername, architecture='838')
    # inst = Autoencoder_Sweeper(data=data_treated, work_folder=foldername, architecture='767')
    # inst = Autoencoder_Sweeper(data=data_treated, work_folder=foldername, architecture='757')
    # inst = Autoencoder_Sweeper(data=data_treated, work_folder=foldername, architecture='747')
    # inst = Autoencoder_Sweeper(data=data_treated, work_folder=foldername, architecture='737')

#%%
if __name__ == '__main__':
    
    foldername = 'COL2_602020'

    logger = defLogger('{}.log'.format(strftime('%Y_%m_%d', localtime())), foldername)

    data, inputs, compositions = read_data('COL2')

    data_treated = data_treatment().pretreatment(data,
                                                 output_columns = compositions,
                                                 num_clusters=1,
                                                 hotteling=True,
                                                 y_rank=[0.60, 0.20, 0.20, 'F/C'],
                                                 scaler_type='Standard')
    # D/F -%Hh%Mmin%Ss
    # F/C
    # C11TEMP
    main()
#%%
