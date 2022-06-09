# Group Project - We used PyCharm as our IDE of choice
#
# Members - Group 2
#           Tristan Lilford (1843691) and Robin Jonker (1827572)
#
# README:   The code within this file relates to the calling of the different computation methods and setting of
#           constants that are used within the programs. Data acquisition and then processing occurs before the calling
#           of the relative graphs and results.
#
#           The booleans below control in app functionality:
#
#           show_graphs             - creates all the different graphs relating to both data acquisition and results.
#           show_full_results_table - creates the full results table which shows stock choice vs their
#                                     corresponding eigen values.
#           use_ibm_hardware        - runs the relevant functions on quantum hardware instead of simulators.
#           install_python_packages - Installs packages which are needed by the application (if already installed
#                                     set this to false, it is assumed pip is already installed on the device).
# 
#           By default all booleans are set to TRUE, except for install_python_packages

# BOOLEANS:
show_graphs = True
show_full_results_table = True
show_execution_times_table = True
install_python_packages = False

# PACKAGE INSTALLATIONS:
import sys
import subprocess
if install_python_packages:
    def install(package):
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    install('qiskit')
    install('qiskit-optimization')
    install('qiskit-finance')
    install('yfinance')
    install('pennylane-qiskit')
    install('matplotlib')
    install('seaborn')
    install('tabulate')
    install('pylatexenc')

# IMPORTS
import portfolio_optimization 
import print_results

# CONSTANTS
RISK_FACTOR = 0.7  # Set the risk factor
BUDGET = 2  # Set budget
STOCK_LIST = ['TSLA', 'ABNB', 'FB', 'GOOGL']  # List of stocks alternate stock codes can be found at https://finance.yahoo.com
STOCK_LIST = sorted(STOCK_LIST, key=str.upper)
START_DATE = '2021-1-1'  # Start Date
END_DATE = '2022-1-1'  # End Date


def main():
    # data acquisition
    stock_data = portfolio_optimization.get_stockData(STOCK_LIST, START_DATE, END_DATE)
    returns, mean_returns = portfolio_optimization.get_mean_returns(stock_data)
    covariance_matrix = portfolio_optimization.get_covariance_of_stocks(returns)
    quadratic_program = portfolio_optimization.get_quadratic_program(mean_returns, covariance_matrix, RISK_FACTOR, BUDGET)

    # processing
    classical_result, time_classical = portfolio_optimization.classical_result(quadratic_program)
    built_in_vqe_result, time_built_in_vqe = portfolio_optimization.built_in_vqe(quadratic_program, STOCK_LIST)
    custom_ansatz_vqe_result, time_custom_ansatz_vqe = portfolio_optimization.custom_ansatz_vqe(quadratic_program, STOCK_LIST)
    full_vqe_result, time_full_vqe_result = portfolio_optimization.full_vqe(quadratic_program, STOCK_LIST)

    # printing
    if show_graphs:
        print_results.adj_closing_price_graph(stock_data)
        print_results.plot_mean_returns(STOCK_LIST, mean_returns)
        print_results.plot_covairiance_matrix(covariance_matrix, STOCK_LIST)
        print_results.plot_vqe_graph(full_vqe_result, STOCK_LIST)
    if show_full_results_table:
        print_results.print_result(classical_result, quadratic_program, STOCK_LIST, "Classical Eigen Solver Computation: ")
        print_results.print_result(built_in_vqe_result, quadratic_program, STOCK_LIST, "Built-in VQE Computation: ")
        print_results.print_result(custom_ansatz_vqe_result, quadratic_program, STOCK_LIST, "Custom Ansatz VQE Computation: ")
    if show_execution_times_table:
        print_results.print_table([['Classical Eigen Solver', time_classical], ['Built-in VQE', time_built_in_vqe],
                                  ['Custom ansatz VQE', time_custom_ansatz_vqe], ['Full VQE', time_full_vqe_result]])


main()
