# Print results
#
# README:   The code within this python file relates to the printing of all the results from the main file. There are
#           multiple functions found within this file that print the relative graphs and formatting of the console
#           output results and tables.

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tabulate import tabulate as tb
from qiskit_optimization.converters import QuadraticProgramToQubo


def adj_closing_price_graph(stockData):     # Adjusted Closing Price vs Time Graph
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(stockData)
    plt.title('Adjusted Close Price vs Time')
    plt.xlabel('Date', fontsize=20)
    plt.ylabel('Adjusted Closing Price (USD)', fontsize=20)
    ax.legend(stockData.columns.values)
    plt.show()


def probabilities_graph(state, probs, prob_of):
    f, ax = plt.subplots(figsize=(15, 6))
    optimized_value = sns.barplot(state, probs)
    for item in optimized_value.get_xticklabels():
        item.set_rotation(45)
    plt.title("Probabilites of Each Possible Combination of Assets For: "+prob_of, fontsize=15)
    plt.xlabel('Possible Combinations of Assets', fontsize=10)
    plt.ylabel('Probability', fontsize=10)
    plt.show()


def index_to_selection(i, num_assets):
    s = "{0:b}".format(i).rjust(num_assets)
    x = np.array([1 if s[i] == "1" else 0 for i in reversed(range(num_assets))])
    return x


def print_result(result, data, stock_list, result_of):
    print(result_of)
    selection = result.x
    print("Optimal: selection {}, value {:.4f}".format(selection, result.fval))
    stocks = []
    for i in range(len(stock_list)):
        if int(selection[i]) == 1:
            stocks.append(stock_list[i])
    print('Invest in the following stocks: ' + str(stocks))
    eigenstate = result.min_eigen_solver_result.eigenstate
    eigenvector = eigenstate if isinstance(eigenstate, np.ndarray) else eigenstate.to_matrix()
    probabilities = np.abs(eigenvector) ** 2
    i_sorted = reversed(np.argsort(probabilities)) 
    print("\n----------------- Full result ---------------------")
    print("selection\tvalue\t\tprobability")
    print("---------------------------------------------------")
    states, values, probs = [], [], []
    for i in i_sorted:
        x = index_to_selection(i, len(stock_list))
        value = QuadraticProgramToQubo().convert(data).objective.evaluate(x)
        probability = probabilities[i]
        states.append(''.join(str(i) for i in x))
        values.append(value)
        probs.append(probability)
        print("%10s\t%.4f\t\t%.4f" % (x, value, probability))
    return probabilities_graph(states, probs, result_of)


def print_table(times):
    table = tb(times, headers=['Computation Method', 'Execution Time (s)'], tablefmt='orgtbl')
    print(table)


def plot_mean_returns(stockList, mu):    
    sns.barplot(x=stockList, y=mu)


def plot_covairiance_matrix(sigma, stock_list):
    f, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(sigma, mask=np.zeros_like(sigma, dtype=bool), annot=True,
                square=True, ax=ax, xticklabels=stock_list, yticklabels=stock_list)
    plt.title("Covariance between Equities")
    plt.show()


def plot_vqe_graph(x, stock_list):
    plt.subplots(figsize=(15, 6))
    plt.title("Probabilites of Each Possible Combination of Assets For: Full VQE Method", fontsize=15)
    x_label = []
    getbinary = lambda x, n: format(x, 'b').zfill(len(stock_list))
    for i in range(2**len(stock_list)):
        x_label.append(getbinary(i, len(stock_list)))
    plt.bar(x_label, x)
    plt.show()
