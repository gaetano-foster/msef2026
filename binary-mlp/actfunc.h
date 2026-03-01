#ifndef ACTFUNC_H
#define ACTFUNC_H

#include <math.h>
#include <stddef.h>

enum actfunc_type {
	ACTFUNC_RELU,
	ACTFUNC_LEAKY,
	ACTFUNC_SIGMOID,
	ACTFUNC_TANH,
	ACTFUNC_SILU,
	ACTFUNC_SOFTMAX
};

// x = array of inputs
// len = length of array
// idx = index of array being operated on
double 
actfunc	(enum actfunc_type type, 
	 double *x,
	 size_t len,
	 size_t idx);

double 
der_actfunc	(enum actfunc_type type, 
		 double *x,
		 size_t len,
		 size_t idx);

#endif // ACTFUNC_H
