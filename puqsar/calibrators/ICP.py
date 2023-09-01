#    Copyright © 2023 Merck & Co., Inc., Rahway, NJ, USA and its affiliates. All rights reserved.
#
#    This file is part of the PUQSAR package, an open source software for computing the Prediction Uncertainty for QSAR.
#
#    Prediction Uncertainty for QSAR (PUQSAR) is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys
import os
import time
import pandas as pd
import numpy as np
import math

def CP_homo(pred_val, alpha, act_col='Act',pred_col = 'Pred', unc = 'Pred_UNC'):
    absErr_val = abs(pred_val[pred_col]-pred_val[act_col])
    PI_width = np.quantile(absErr_val, alpha)  # scalar
    def fun_PI(pred_test, pred_col = 'Pred', unc = 'Pred_UNC', act_col='Act'):
        pred_test_withPI = pred_test.copy()
        pred_test_withPI['PI_width'] = PI_width
        if (act_col is not None) and (act_col in list(pred_test)):
            absErr_test = abs(pred_test[pred_col]-pred_test[act_col])
            test_coverage = (absErr_test <= PI_width) # vector
            pred_test_withPI['testCoverage'] = test_coverage
        return pred_test_withPI
    return fun_PI

def CP_expSD(pred_val, alpha, act_col='Act',pred_col = 'Pred', unc = 'Pred_UNC'):
    absErr_val = abs(pred_val[pred_col]-pred_val[act_col])
    nonconformity_val = absErr_val/np.exp(pred_val[unc])
    alpha_CL = np.quantile(absErr_val, alpha)
    def fun_PI(pred_test, pred_col = 'Pred', unc = 'Pred_UNC', act_col='Act'):
        pred_test_withPI = pred_test.copy()
        PI_width = alpha_CL*np.exp(pred_test[unc])  # vector
        pred_test_withPI['PI_width'] = PI_width
        if (act_col is not None) and (act_col in list(pred_test)):
            absErr_test = abs(pred_test[pred_col]-pred_test[act_col])
            test_coverage = (absErr_test <= PI_width)
            pred_test_withPI['testCoverage'] = test_coverage
        return pred_test_withPI
    return fun_PI

def CP_ACE(pred_val, alpha, act_col='Act',pred_col = 'Pred', unc = 'Pred_UNC'):
    absErr_val = abs(pred_val[pred_col]-pred_val[act_col])
    f_PredSD = fun_scaledUS(pred_val,alpha, act_col,pred_col,unc)
    Pred_UNC_val = [f_PredSD(x) for x in pred_val[unc]]
    nonconformity_val = absErr_val/Pred_UNC_val
    alpha_CL = np.quantile(nonconformity_val, alpha)  
    def fun_PI(pred_test, pred_col = 'Pred', unc = 'Pred_UNC', act_col='Act'):
        # Add a column of PI_width to test set
        # If act_col exists, add a column of test_coverage (TRUE/FALSE)
        pred_test_withPI = pred_test.copy()
        Pred_UNC_test = [f_PredSD(x) for x in pred_test[unc]]
        PI_width = [alpha_CL * w for w in Pred_UNC_test]
        pred_test_withPI['PI_width'] = PI_width
        if (act_col is not None) and (act_col in list(pred_test)):
            absErr_test = abs(pred_test[pred_col]-pred_test[act_col])
            test_coverage = (absErr_test <= PI_width)
            pred_test_withPI['testCoverage'] = test_coverage
        return pred_test_withPI
    return fun_PI 

def fun_scaledUS(pred_val, alpha, act_col='Act',pred_col = 'Pred', unc = 'Pred_UNC'):
    absErr_val_all = abs(pred_val[pred_col]-pred_val[act_col])
    mu_absErr_val = np.mean(absErr_val_all)
    sd_absErr_val = np.std(absErr_val_all)
    mu_predSD_val = np.mean(pred_val[unc])
    sd_predSD_val = np.std(pred_val[unc])
    b = mu_absErr_val
    a_max = b/sd_absErr_val
    R = 20
    f = np.repeat([0,1], np.ceil(pred_val.shape[0]/2), axis = 0)[0:pred_val.shape[0]]
    df_para_all = []
    for rrr in range(R):
        np.random.shuffle(f)
        pred_val_1 = pred_val[f==0]
        pred_val_2 = pred_val[f==1]
        df_k = fun_df_para(pred_val_1,pred_val_2, a_max, b, sd_absErr_val, mu_predSD_val, sd_predSD_val, alpha, act_col,pred_col,unc)
        df_para_all.append(df_k)
    df_para_all = pd.concat(df_para_all, axis = 0, ignore_index = True)
    df_para_all.dropna(axis = 0, how = 'any', inplace = True)
    df_summary_avg = df_para_all.groupby('a',as_index=True).mean().rename(columns={"test_coverage": "avg_test_coverage", 
                                                                 "PI_width": "avg_PI_width", 
                                                                 "test_coverage_errSubGroup": "avg_test_coverage_errSubGroup"})
    df_summary_sd = df_para_all.groupby('a',as_index=True).std().rename(columns={"test_coverage": "sd_test_coverage", 
                                                            "PI_width": "sd_PI_width", 
                                                             "test_coverage_errSubGroup": "sd_test_coverage_errSubGroup"})  
    df_para_all = df_summary_avg.join(df_summary_sd, on = 'a', how = 'inner') 
    df_para_all.reset_index(inplace=True)      
    df_para_all.sort_values(by='avg_test_coverage_errSubGroup',axis = 0, ascending  =True, inplace = True)
    df_para_all.reset_index(inplace=True, drop = True)
    threshold1 = df_para_all['avg_test_coverage_errSubGroup'][0]+df_para_all['sd_test_coverage_errSubGroup'][0]/np.sqrt(R)
    df_para_subset = df_para_all[df_para_all['avg_test_coverage_errSubGroup'] <= threshold1].copy() 
    if df_para_subset.shape[0]>1:
        df_para_subset.sort_values(by='avg_PI_width',axis = 0, ascending  =True, inplace = True)
        df_para_subset.reset_index(inplace=True)
        threshold2 = df_para_subset['avg_PI_width'][0]+df_para_subset['sd_PI_width'][0]/np.sqrt(R)
        df_para_subset = df_para_subset[df_para_subset['avg_PI_width'] <= threshold2] 
    a_opt = np.median(df_para_subset['a'])*sd_absErr_val
    def f_PredSD(x):
        return(a_opt*np.tanh((x - mu_predSD_val)/sd_predSD_val)+b)
    return(f_PredSD)


def fun_df_para(pred_val_1, pred_val_2, a_max, b, sd_absErr_val, mu_predSD_val, sd_predSD_val, alpha, act_col='Act',pred_col = 'Pred', unc = 'Pred_UNC'):
    absErr_val = abs(pred_val_1[pred_col]-pred_val_1[act_col])
    absErr_test = abs(pred_val_2[pred_col]-pred_val_2[act_col])
    n_test = len(absErr_test)
    if n_test <= 2000:
        subGroup=pd.cut(x=pred_val_2[pred_col], bins=np.quantile(pred_val_2[pred_col],[0,0.25,0.5,0.75,1]),labels = [1,2,3,4])
    else:
        n_bins = math.ceil(n_test/500.0)
        subGroup=pd.cut(x=pred_val_2[pred_col], bins=np.quantile(pred_val_2[pred_col],[0]+[i/n_bins for i in list(range(1,1+n_bins))]),labels = list(range(1,1+n_bins)))
    df_para = []
    for k in range(100):
        a = (k/100.0)*a_max*sd_absErr_val
        Pred_UNC_val = a*np.tanh((pred_val_1[unc] - mu_predSD_val)/sd_predSD_val)+b
        Pred_UNC_test = a*np.tanh((pred_val_2[unc] - mu_predSD_val)/sd_predSD_val)+b
        nonconformity_val = absErr_val/Pred_UNC_val
        alpha_CL = np.quantile(nonconformity_val, alpha) 
        PI_width = alpha_CL * Pred_UNC_test
        test_coverage = (absErr_test <= PI_width)
        test_coverage_k = np.mean(test_coverage)
        PI_width_k = np.mean(PI_width)
        df_temp = pd.DataFrame({'subGroup':subGroup, 'test_coverage': test_coverage})
        test_coverage_errSubGroup_k = np.mean(np.abs(df_temp.groupby('subGroup').mean().to_numpy()-alpha))
        df_para.append([(k/100.0)*a_max, test_coverage_k, PI_width_k, test_coverage_errSubGroup_k])
    df_para = pd.DataFrame(df_para, columns =['a', 'test_coverage', 'PI_width','test_coverage_errSubGroup'])
    return(df_para)
