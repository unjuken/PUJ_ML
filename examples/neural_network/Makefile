INCLUDES=-I/usr/local/include/eigen3
C_VERSION=c++17
## OPT_FLAGS=-g
OPT_FLAGS=-Ofast
EXE_FLAGS=-std=$(C_VERSION) $(OPT_FLAGS) $(INCLUDES)
OBJ_FLAGS=$(EXE_FLAGS) -c

all: test binary_classifier experiment bagging
clean:
	rm -rf *.o test binary_classifier experiment bagging

test: ActivationFunctions.o Layer.o NeuralNetwork.o test.cxx
	g++ $(EXE_FLAGS) -g -o $@ $^

factory: ActivationFunctions.o Layer.o NeuralNetwork.o NeuralNetworkFactory.cxx
	g++ -std=c++17 -I/usr/local/include/eigen3 -g -o $@ $^

%.o: %.cxx
	g++ -std=c++17 -I/usr/local/include/eigen3 -g -c -o $@ $^

binary_classifier: CSVReader.o ActivationFunctions.o Layer.o NeuralNetwork.o binary_classifier.cxx
	g++ $(EXE_FLAGS) -g -o $@ $^

experiment: CSVReader.o ActivationFunctions.o Layer.o NeuralNetwork.o experiment.cxx
	g++ $(EXE_FLAGS) -g -o $@ $^

bagging: CSVReader.o ActivationFunctions.o Layer.o NeuralNetwork.o bagging.cxx
	g++ $(EXE_FLAGS) -g -o $@ $^

%.o: %.cxx
	g++ $(OBJ_FLAGS) -g -o $@ $^

## eof - $RCSfile$
