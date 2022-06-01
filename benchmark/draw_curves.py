import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_relative_optimality_gap(estimator):
    min_eval=100
    max_dual=-100
    min_eval=min(min_eval,np.min(estimator.optimization_info_[1,]))
    max_dual=max(max_dual,np.max(estimator.optimization_info_[2,]))

    info = np.array([ np.maximum((estimator.optimization_info_[1,]-max_dual)/min_eval,1e-9).T, estimator.optimization_info_[-1, :].T]).T

    return info



"""
    df = pd.read_csv("test.csv")

    df.plot(kind='line', x='Time(s)', y='Duality gap', logy=True, logx=True)
    plt.show()

    info = compute_relative_optimality_gap(estimator)
"""

def recreate_filename(parameter, changing_value, position):
    groups = parameter.split('_')
    beginning_string = '_'.join(groups[:position])
    ending_string = '_'.join(groups[position:])
    return beginning_string + "_" + changing_value + "_" + ending_string

def get_variations(files, position):
    values_list = list()
    parameters_list = list()

    for file in files:
        groups = file.split("_", position + 1)
        beginning_string = '_'.join(groups[:position])
        values_list.append(groups[-2])
        parameter = beginning_string + "_" + groups[-1]
        parameters_list.append(parameter)

    unique_values = np.unique(values_list)
    unique_parameters = np.unique(parameters_list)

    return unique_parameters, unique_values






files = glob.glob("csv/*")
files = [file.split("/")[1] for file in files]
cores_list = list()
parameters_list = list()

for file in files:
    core, parameter = file.split("_", 1)
    cores_list.append(core)
    parameters_list.append(parameter)

unique_parameters = np.unique(parameters_list)
unique_cores = np.unique(cores_list)
for parameter in unique_parameters:
    for core in unique_cores:
        df = pd.read_csv("csv/" + str(core) + "_" + parameter)
        plt.plot(df['Time(s)'], df['Primal objective'], label=str(core))
    plt.yscale('log')
    plt.xscale('log')
    plt.title(parameter.rsplit('.', 1)[0])
    plt.legend()
    plt.show()

parameters, values = get_variations(files, 1)

for parameter in parameters:
    for value in values:
        df = pd.read_csv("csv/" + recreate_filename(parameter, value, 1))
        plt.plot(df['Time(s)'], df['Primal objective'], label=str(value))
    plt.yscale('log')
    plt.xscale('log')
    plt.title(parameter.rsplit('.', 1)[0])
    plt.legend()
    plt.show()
