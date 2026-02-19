#include <math.h>

double
relu(double x) 
{
    return x > 0 ? x : 0;
}

double
leaky_relu(double x) 
{
    return x > 0 ? x : 0.01 * x;
}

double
sigmoid(double x) 
{
    return 1.0 / (1.0 + exp(-x));
}

double
linear(double x)
{
    return x;
}

double
relu_derivative(double x) 
{
    return x > 0 ? 1 : 0;
}

double
leaky_relu_derivative(double x) 
{   
    return x > 0 ? 1 : 0.01;
}

double
sigmoid_derivative(double x) 
{
    double sig = sigmoid(x);
    return sig * (1 - sig);
}

double
linear_derivative(double x)
{
    return 1;
}
