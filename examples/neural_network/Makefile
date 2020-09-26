
all: test binary_classifier
clean:
	rm -rf *.o test binary_classifier factory

test: ActivationFunctions.o Layer.o NeuralNetwork.o test.cxx
	g++ -g -std=c++17 -I/usr/local/include/eigen3 -o $@ $^

factory: ActivationFunctions.o Layer.o NeuralNetwork.o NeuralNetworkFactory.cxx
	g++ -std=c++17 -I/usr/local/include/eigen3 -g -o $@ $^

%.o: %.cxx
	g++ -std=c++17 -I/usr/local/include/eigen3 -c -o $@ $^

binary_classifier: CSVReader.o ActivationFunctions.o Layer.o NeuralNetwork.o binary_classifier.cxx
	g++ -g -std=c++17 -I/usr/local/include/eigen3 -o $@ $^

%.o: %.cxx
	g++ -g -std=c++17 -I/usr/local/include/eigen3 -c -o $@ $^

## eof - $RCSfile$
