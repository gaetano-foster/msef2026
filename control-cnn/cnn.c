#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "actfunc.h"
#include "cost.h"

struct layer {
	uint32_t in;			// number of input nodes
	uint32_t out;			// number of output nodes
	double *weights;		// heap allocated 2D array of weights
	double *biases;			// heap allocated array of biases
	enum actfunc_type act_type;	// activation function
};

int
layer_init	(struct layer *layer, // stack allocated layer
		 uint32_t in,
		 uint32_t out)
{
	layer->in = in;
	layer->out = out;
	layer->act_type = ACTFUNC_SIGMOID;
	if (!(layer->weights = calloc(in * out, sizeof(double)))) {
		return 0;
	}

	if (!(layer->biases = calloc(out, sizeof(double)))) {
		free(layer->weights);
		return 0;
	}
	
	return 1;
}

void
calc_layer_outputs	(struct layer *layer,
			 double *output,
			 double *inputs,
			 enum actfunc_type func_type)
{
	// this logic is wrong. fix it later
	for (int o = 0; o < layer->out; o++) {
		double out = layer->biases[o]; // weighted input
		for (int i = 0; i < layer->in; i++) {
			out += inputs[i] * layer->weights[o * layer->in + i];
		}
		output[o] = actfunc(func_type, inputs, layer->in, o);
	}
}

void
layer_destroy	(struct layer *layer)
{
	free(layer->weights);
	free(layer->biases);
}

struct network {
	struct layer *layers;
	size_t *layer_sizes;
	size_t num_layers;
};

int
network_init	(struct network *network,
		 size_t *layer_sizes,
		 size_t num_layers)
{
	network->num_layers = num_layers;
	network->layer_sizes = layer_sizes;
	
	if (!(network->layers = calloc(num_layers - 1, sizeof(struct layer)))) 
		return 0;

	for (int i = 0; i < num_layers - 1; i++) { 
		if (!layer_init(&network->layers[i], layer_sizes[i], layer_sizes[i + 1]))
			return 0;
	}
	return 1;
}

void
network_destroy	(struct network *network)
{
	for (int i = 0; i < network->num_layers - 1; i++)
		layer_destroy(&network->layers[i]);

	free(network->layers);
}

int
main	(int argc,
	 char **argv)
{
	
	return 0;
}
