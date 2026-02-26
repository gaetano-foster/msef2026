#ifndef ACTFUNC_H
#define ACTFUNC_H

enum actfunc_type {
	ACTFUNC_RELU,
	ACTFUNC_LEAKY,
	ACTFUNC_SIGMOID,
	ACTFUNC_TANH,
	ACTFUNC_SILU,
	ACTFUNC_SOFTMAX
};

double actfunc	(enum actfunc_type type, 
		 double *x,
		 size_t len,
		 size_t idx);

double der_actfunc	(enum actfunc_type type, 
			 double *x,
			 size_t len,
			 size_t idx);

double relu(double x);
double leaky_relu(double x);
double sigmoid(double x);
double silu(double x);
double softmax	(double *x,
		 size_t len,
		 size_t idx); // index must be in range of array x

double der_relu(double x);
double der_leaky_relu(double x);
double der_sigmoid(double x);
double der_tanh(double x);
double der_silu(double x);
double der_softmax	(double *x,
			 size_t len,
			 size_t idx);


#endif // ACTFUNC_H
