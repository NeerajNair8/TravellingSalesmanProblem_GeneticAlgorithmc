# TravellingSalesmanProblem_GeneticAlgorithmc

To execute the program, open the code in any python IDE and run the code .
Numpy must be installed in the device to run the program .
The program will ask the user to enter if he/she wants to input a new graph or use the default graph .To enter a new graph, type 2. To use one of the default graphs, type 0 or 1.
The program will then print the graph and ask the user to enter the size of the starting population. A chromosome consists of an ordered set which indicates the order in which the vertices are traversed. Population is created randomly by taking ordered non repeating samples from the set of vertices.
Fitness function is defined as the inverse of the total distance for a path multiplied by the largest value in the graph.  F= (1/Distance_of_the_path)*max(Graph_values)*(num_vertices-1)
Then, 50% of the population is selected as elite individuals using roulette wheel selection. 
Parents are selected randomly and crossed to produce off springs.
Ordered crossover is used to by default produce off springs. Crossover function to be used can be given as the parameter for the crossover() function (“PMX” or “OX”).
There is a 0.5%chance of mutation of off springs. Mutation randomly swaps two bits in the chromosome.   
New generation consists of 50% elite individuals and 50% off springs.
Number of iterations the program runs can be given as a parameter when calling the main() function. 
