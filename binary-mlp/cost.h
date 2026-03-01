#ifndef __COST_H__
#define __COST_H__

enum cost_type {
	COST_MEAN_SQUARED_ERROR,
	COST_CROSS_ENTROPY
};

double 
cross_entropy	(double *predicted,
		 double *expected,
		 size_t len);
double 
mean_square_err	(double *predicted,
		 double *expected,
		 size_t len);
double 
der_cross_entropy	(double predicted,
			 double expected);
double 
der_mean_square_err	(double predicted,
			 double expected);
double 
costfunc(enum cost_type type,
	 double *predicted,
	 double *expected,
	 size_t len);
double 
der_costfunc	(enum cost_type type,
		 double predicted,
		 double expected);
#endif
