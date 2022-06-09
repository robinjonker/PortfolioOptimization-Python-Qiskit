#Group 2
#Members:   
            Tristan Lilford (1843691)
            Robin Jonker (1827572)

The code is split into 3 different files:
- main.py
- portfolio_optimization.py
- print_results.py

# main:
This is the main file that gets run to output the results and graphs.
Within this file you will find:
    -Booleans that control the functionality of the program.
    -Installation of python packages if required
    -Setting of constants
    -Acquiring and processing the data
    -Outputting the desired results, graphs and tables

# portfolio_optimization:
This is the file that contains all the method computation functions for the program.
Within this file you will find:
    -The data acquisition functions
    -The data processing functions
    -The functions for the different methods:
        -Classical computation
        -Qiskit built-in VQE computation
        -Custom ansatz within Qiskit VQE computation
        -Full VQE computation

# print_results:
This is the file that contains all the functions that prints the results, graphs and tables.
Within the file you will find:
    -Plotting of the figures reletive to the data
    -Plotting the graphs of each methods probability
    -Printing the results of each methods outcome
    -Printing the table of execution times of the methods
