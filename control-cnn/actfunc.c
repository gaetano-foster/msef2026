#include <math.h>
#include "actfunc.h"

double
actfunc(enum actfunc_type type,
	double *x,
	size_t len,
	size_t idx)
{
	switch (type) {
	case ACTFUNC_RELU: break;
	case ACTFUNC_LEAKY: break;
	case ACTFUNC_SIGMOID: break;
	case ACTFUNC_TANH: break;
	case ACTFUNC_SILU: break;
	case ACTFUNC_SOFTMAX: break;
	}
}


double
der_actfunc(enum actfunc_type type,
	double *x,
	size_t len,
	size_t idx)
{
	switch (type) {
	case ACTFUNC_RELU: break;
	case ACTFUNC_LEAKY: break;
	case ACTFUNC_SIGMOID: break;
	case ACTFUNC_TANH: break;
	case ACTFUNC_SILU: break;
	case ACTFUNC_SOFTMAX: break;
	}
}
