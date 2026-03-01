#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "actfunc.h"
#include "cost.h"

struct layer {
	uint32_t num_inputs;		// number of input nodes
	uint32_t num_outputs;		// number of output nodes
	double *weights;	// heap allocated 2D array of weights
	double *biases;		// heap allocated array of biases
	double *scratch;	// used for calculating layer outputs
};

int
layer_init	(struct layer *layer, // stack allocated layer
		 uint32_t num_inputs,
		 uint32_t num_outputs)
{
	layer->num_inputs = num_inputs;
	layer->num_outputs = num_outputs;
	if (!(layer->weights = calloc(num_inputs * num_outputs, sizeof(double)))) {
		return 0;
	}

	if (!(layer->biases = calloc(num_outputs, sizeof(double)))) {
		free(layer->weights);
		return 0;
	}

	if (!(layer->scratch = calloc(num_outputs, sizeof(double)))) {
		free(layer->weights);
		free(layer->biases);
		return 0;
	}
	
	return 1;
}

void
calc_layer_outputs	(struct layer *layer,
			 double *output,
			 double *inputs,
			 enum actfunc_type activation)
{
	for (int out = 0; out < layer->num_outputs; out++) {
		layer->scratch[out] = layer->biases[out];
		for (int in = 0; in < layer->num_inputs; in++) {
			layer->scratch[out] += inputs[in] * 
						layer->weights[out * layer->num_inputs + in];
		}
	}

	for (int i = 0; i < layer->num_outputs; i++) {
		output[i] = actfunc(activation, layer->scratch, layer->num_outputs, i);
	}
}

void
layer_destroy	(struct layer *layer)
{
	free(layer->weights);
	free(layer->biases);
	free(layer->scratch);
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
