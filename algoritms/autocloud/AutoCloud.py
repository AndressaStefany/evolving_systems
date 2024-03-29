'''
Code downloaded from:
https://github.com/MailsonRibeiroSantos/AutoCloud
'''
import numpy as np
import pandas as pd
from algoritms.autocloud.DataCloud import DataCloud


class AutoCloud:
    c = np.array([DataCloud(0)], dtype=DataCloud)
    alfa = np.array([0.0], dtype=float)
    intersection = np.zeros((1, 1), dtype=int)
    listIntersection = np.zeros((1), dtype=int)
    matrixIntersection = np.zeros((1, 1), dtype=int)
    relevanceList = np.zeros((1), dtype=int)
    k = 1

    def __init__(self, m):
        AutoCloud.m = m
        AutoCloud.c = np.array([DataCloud(0)], dtype=DataCloud)
        AutoCloud.alfa = np.array([0.0], dtype=float)
        AutoCloud.intersection = np.zeros((1, 1), dtype=int)
        AutoCloud.listIntersection = np.zeros((1), dtype=int)
        AutoCloud.relevanceList = np.zeros((1), dtype=int)
        AutoCloud.matrixIntersection = np.zeros((1, 1), dtype=int)
        AutoCloud.classIndex = []
        AutoCloud.k = 1

    def fit_predict(self, X):
        for r in X.values:
            self.run(r)

    def mergeClouds(self):
        i = 0
        while(i < len(AutoCloud.listIntersection)-1):
            merge = False
            j = i+1
            while(j < len(AutoCloud.listIntersection)):
                # print("i",i,"j",j,"l",np.size(AutoCloud.listIntersection),"m",np.size(AutoCloud.matrixIntersection),"c",np.size(AutoCloud.c))
                if(AutoCloud.listIntersection[i] == 1 and AutoCloud.listIntersection[j] == 1):
                    AutoCloud.matrixIntersection[i,
                                                 j] = AutoCloud.matrixIntersection[i, j] + 1
                nI = AutoCloud.c[i].n
                nJ = AutoCloud.c[j].n
                meanI = AutoCloud.c[i].mean
                meanJ = AutoCloud.c[j].mean
                varianceI = AutoCloud.c[i].variance
                varianceJ = AutoCloud.c[j].variance
                nIntersc = AutoCloud.matrixIntersection[i, j]
                if (nIntersc > (nI - nIntersc) or nIntersc > (nJ - nIntersc)):
                    merge = True
                    # update values
                    n = nI + nJ - nIntersc
                    mean = ((nI * meanI) + (nJ * meanJ))/(nI + nJ)
                    variance = ((nI - 1) * varianceI + (nJ - 1)
                                * varianceJ)/(nI + nJ - 2)
                    newCloud = DataCloud(mean)
                    newCloud.updateDataCloud(n, mean, variance)
                    # atualizando lista de interseção
                    AutoCloud.listIntersection = np.concatenate((AutoCloud.listIntersection[0: i], np.array(
                        [1]), AutoCloud.listIntersection[i + 1: j], AutoCloud.listIntersection[j + 1: np.size(AutoCloud.listIntersection)]), axis=None)
                    # atualizando lista de data clouds
                    AutoCloud.c = np.concatenate((AutoCloud.c[0: i], np.array(
                        [newCloud]), AutoCloud.c[i + 1: j], AutoCloud.c[j + 1: np.size(AutoCloud.c)]), axis=None)
                    # update  intersection matrix
                    M0 = AutoCloud.matrixIntersection
                    # Remover linhas
                    M1 = np.concatenate((M0[0: i, :], np.zeros(
                        (1, len(M0))), M0[i + 1: j, :], M0[j + 1: len(M0), :]))
                    # remover colunas
                    M1 = np.concatenate((M1[:, 0: i], np.zeros(
                        (len(M1), 1)), M1[:, i+1: j], M1[:, j+1: len(M0)]), axis=1)
                    # calculando nova coluna
                    col = (M0[:, i] + M0[:, j])*(M0[:, i]*M0[:, j] != 0)
                    col = np.concatenate((col[0: j], col[j + 1: np.size(col)]))
                    # calculando nova linha
                    lin = (M0[i, :]+M0[j, :])*(M0[i, :]*M0[j, :] != 0)
                    lin = np.concatenate((lin[0: j], lin[j + 1: np.size(lin)]))
                    # atualizando coluna
                    M1[:, i] = col
                    # atualizando linha
                    M1[i, :] = lin
                    M1[i, i + 1: j] = M0[i, i + 1: j] + M0[i + 1: j, j].T
                    AutoCloud.matrixIntersection = M1
                j += 1
            if(merge):
                i = 0
            else:
                i += 1

    def run(self, X):
        AutoCloud.listIntersection = np.zeros(
            (np.size(AutoCloud.c)), dtype=int)
        if AutoCloud.k == 1:
            AutoCloud.c[0] = DataCloud(X)
            AutoCloud.classIndex.append(0)
        elif AutoCloud.k == 2:
            AutoCloud.c[0].addDataClaud(X)
            AutoCloud.classIndex.append(0)
        elif AutoCloud.k >= 3:
            i = 0
            createCloud = True
            AutoCloud.alfa = np.zeros((np.size(AutoCloud.c)), dtype=float)
            for data in AutoCloud.c:
                n = data.n + 1
                mean = ((n-1)/n)*data.mean + (1/n)*X
                variance = ((n-1)/n)*data.variance + (1/n) * \
                    ((np.linalg.norm(X-mean))**2)
                eccentricity = (1/n)+((mean-X).T.dot(mean-X))/(n*variance)
                typicality = 1 - eccentricity
                norm_eccentricity = eccentricity/2
                norm_typicality = typicality/(AutoCloud.k-2)
                data.eccAn = eccentricity
                if(norm_eccentricity <= (AutoCloud.m**2 + 1)/(2*n)):
                    data.updateDataCloud(n, mean, variance)
                    AutoCloud.alfa[i] = norm_typicality
                    createCloud = False
                    AutoCloud.listIntersection.itemset(i, 1)
                else:
                    AutoCloud.alfa[i] = 0
                    AutoCloud.listIntersection.itemset(i, 0)
                i += 1

            if(createCloud):
                AutoCloud.c = np.append(AutoCloud.c, DataCloud(X))
                AutoCloud.listIntersection = np.insert(
                    AutoCloud.listIntersection, i, 1)
                AutoCloud.matrixIntersection = np.pad(
                    AutoCloud.matrixIntersection, ((0, 1), (0, 1)), 'constant', constant_values=(0))
            self.mergeClouds()
            AutoCloud.relevanceList = AutoCloud.alfa / np.sum(AutoCloud.alfa)
            classIndex = np.argmax(AutoCloud.relevanceList)
            AutoCloud.classIndex.append(classIndex)

        AutoCloud.k = AutoCloud.k+1
