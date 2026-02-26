#include "actfunc.h"
#define LEAKY_CONSTANT		(0.01)

// functions
static double 
relu	(double *x,
	 size_t len,
	 size_t idx)
{
	if (len >= idx || x == NULL) return NAN;
	return fmax(0, x[idx]);
}

static double 
leaky_relu	(double *x,
		 size_t len,
		 size_t idx)
{	
	if (len >= idx || x == NULL) return NAN;
	return (x[idx] > 0) ? x[idx] : LEAKY_CONSTANT * x[idx];
}

static double 
sigmoid	(double *x,
	 size_t len,
	 size_t idx)
{	
	if (len >= idx || x == NULL) return NAN;
	return 1.0 / (1 + exp(-x[idx]));
}

static double
gtanh	(double *x,
	 size_t len,
	 size_t idx)
{	
	if (len >= idx || x == NULL) return NAN;
	return tanh(x[idx]); // c standard library already has it
}

static double 
silu	(double *x,
	 size_t len,
	 size_t idx)
{	
	if (len >= idx || x == NULL) return NAN;
	return x[idx] / (1 + exp(-x[idx]));
}

static double 
softmax	(double *x,
	 size_t len,
	 size_t idx)
{
	if (len >= idx || x == NULL) return NAN;
	
	double sum = 0;
	for (int i = 0; i < len; i++) {
		sum += exp(x[i]);
	}

	return exp(x[idx]) / sum;
}

// derivatives
static double 
der_relu	(double *x,
		 size_t len,
		 size_t idx)
{
	if (len >= idx || x == NULL) return NAN;
	return (x[idx] > 0) ? 1 : 0;
}

static double 
der_leaky_relu	(double *x,
		 size_t len,
		 size_t idx)
{
	if (len >= idx || x == NULL) return NAN;
	return (x[idx] > 0) ? 1 : LEAKY_CONSTANT;
}

static double 
der_sigmoid	(double *x,
		 size_t len,
		 size_t idx)
{
	if (len >= idx || x == NULL) return NAN;
	
	double a = sigmoid(x, len, idx);
	return a * (1 - a);
}

static double
der_gtanh	(double *x,
		 size_t len,
		 size_t idx)
{
	if (len >= idx || x == NULL) return NAN;
	
	double t = tanh(x[idx]);
	return 1 - t * t;
}

static double 
der_silu	(double *x,
		 size_t len,
		 size_t idx)
{
	if (len >= idx || x == NULL) return NAN;
	
	double sig = 1 / (1 + exp(-x[idx]));
	return x[idx] * sig * (1 - sig) + sig;
}

static double 
der_softmax	(double *x,
		 size_t len,
		 size_t idx)
{
	if (len >= idx || x == NULL) return NAN;
	
	double sum = 0;
	for (int i = 0; i < len; i++) {
		sum += exp(x[i]);
	}

	double ex = exp(x[idx]);
	return (ex * sum - ex * ex) / (sum * sum);
}

double
actfunc(enum actfunc_type type,
	double *x,
	size_t len,
	size_t idx)
{
	switch (type) {
	case ACTFUNC_RELU:	return relu(x, len, idx);
	case ACTFUNC_LEAKY:	return leaky_relu(x, len, idx);
	case ACTFUNC_SIGMOID:	return sigmoid(x, len, idx);
	case ACTFUNC_TANH:	return gtanh(x, len, idx);
	case ACTFUNC_SILU:	return silu(x, len, idx);
	case ACTFUNC_SOFTMAX:	return softmax(x, len, idx);
	}
}


double
der_actfunc(enum actfunc_type type,
	double *x,
	size_t len,
	size_t idx)
{
	switch (type) {
	case ACTFUNC_RELU:	return der_relu(x, len, idx);     
	case ACTFUNC_LEAKY:	return der_leaky_relu(x, len, idx); 
	case ACTFUNC_SIGMOID:	return der_sigmoid(x, len, idx); 
	case ACTFUNC_TANH:	return der_gtanh(x, len, idx);    
	case ACTFUNC_SILU:	return der_silu(x, len, idx);     
	case ACTFUNC_SOFTMAX:	return der_softmax(x, len, idx);
	}
}
