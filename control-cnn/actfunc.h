#ifndef ACTFUNC_H
#define ACTFUNC_H

double relu(double x);
double leaky_relu(double x);
double sigmoid(double x);
double linear(double x);

double relu_derivative(double x);
double leaky_relu_derivative(double x);
double sigmoid_derivative(double x);
double linear_derivative(double x);

typedef double (*actfunc_t)(double x);

#endif // ACTFUNC_H
