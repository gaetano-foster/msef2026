#include <stdlib.h>
#include <math.h>
#include "cost.h"
 
double
cross_entropy	(double *predicted,
		 double *expected,
		 size_t len)
{
	double cost = 0;

	for (size_t i = 0; i < len; i++) {
		double x = predicted[i];
		double y = expected[i];
		double v = (y == 1) ? -log(x) : -log(1 - x);
		cost += isnan(v) ? 0 : v;
	}
	return cost;
}

double
der_cross_entropy(double predicted,
		  double expected)
{
	double x = predicted;
	double y = expected;

	if (x == 0 || x == 1) 
		return 0;

	return (-x + y) / (x * (x - 1));
}

double
mean_square_err	(double *predicted,
		 double *expected,
		 size_t len)
{
	double cost = 0;
	
	for (int i = 0; i < len; i++) {
		double error = predicted[i] - expected[i];
		cost += error * error;
	}
	return 0.5 * cost;
}

double
der_mean_square_err	(double predicted,
			 double expected)
{
	return predicted - expected;
}

double
costfunc	(enum cost_type type,
		 double *predicted,
		 double *expected,
		 size_t len)
{
	switch (type) {
	case COST_MEAN_SQUARED_ERROR: return mean_square_err(predicted, expected, len);
	case COST_CROSS_ENTROPY: return cross_entropy(predicted, expected, len);
	}
}

double
der_costfunc	(enum cost_type type,
		 double predicted,
		 double expected)
{
	switch (type) {
	case COST_MEAN_SQUARED_ERROR: return der_mean_square_err(predicted, expected);
	case COST_CROSS_ENTROPY: return der_cross_entropy(predicted, expected);
	}
}
