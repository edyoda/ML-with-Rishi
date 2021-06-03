# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 20:20:11 2021

@author: RISHBANS
"""
import pandas as pd
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
data = pd.DataFrame(cancer.data, columns = cancer.feature_names)
data['cancer'] = cancer.target