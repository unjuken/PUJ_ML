#include "ActivationFunctions.h"
#include "NeuralNetwork.h"
#include "Layer.h"
#include <iostream>
#include <iostream>
using namespace std;
using TScalar = float;
using TAnn = NeuralNetwork< TScalar >;
 using TLayer  = Layer< TScalar >;
using TActivation = typename TLayer::TActivation;


struct NeuralLayer
{
    unsigned int Size = 0;
    TActivation ActivationFunction = ActivationFunctions::Identity< TScalar >( );
};


int main( )
{
    TAnn ann;
    NeuralLayer current, next;

    NeuralLayer network [4] = {
        {
            .Size = 8,
            .ActivationFunction = ActivationFunctions::Logistic< TScalar >( )
        },
        {
            .Size = 5,
            .ActivationFunction = ActivationFunctions::Logistic< TScalar >( )
        },
        {
            .Size = 3,
            .ActivationFunction = ActivationFunctions::Logistic< TScalar >( )
        },
        {
            .Size = 1,
            .ActivationFunction = ActivationFunctions::Logistic< TScalar >( )
        }
    };

    int numberOfLayers = sizeof(network)/sizeof(network[0]) - 1;

    current.Size = network[0].Size;
    current.ActivationFunction = network[0].ActivationFunction;

    for (size_t i = 1; i <= numberOfLayers; i++)
    {
        next.Size = network[i].Size;
        next.ActivationFunction = network[i].ActivationFunction;
        
        cout << "current: " << current.Size << ". Next: " << next.Size << endl;
        ann.add( 
            current.Size, 
            next.Size, 
            current.ActivationFunction);
        
        current = next;

    }

    // Initialize the ANN with random weights and biases
    ann.init( true );

    // Test the forward-propagation with a one-filled vector
    TAnn::TRowVector x = TAnn::TRowVector::Ones( 8 );
    TAnn::TColVector y = ann( x );

    std::cout << "Input : " << x << std::endl;
    std::cout << "Output: " << std::endl << y << std::endl;

    //ActivationFunctions::ActivationFactory<TScalar>((ActivationFunctions::ActivationFunctions)activationFunction, 0.001);    
    return( 0 );
}


