
all: test
clean:
	rm -rf *.o test factory

test: ActivationFunctions.o Layer.o NeuralNetwork.o test.cxx
	g++ -std=c++17 -I/usr/local/include/eigen3 -o $@ $^

factory: ActivationFunctions.o Layer.o NeuralNetwork.o NeuralNetworkFactory.cxx
	g++ -std=c++17 -I/usr/local/include/eigen3 -o $@ $^

%.o: %.cxx
	g++ -std=c++17 -I/usr/local/include/eigen3 -c -o $@ $^


## eof - $RCSfile$
