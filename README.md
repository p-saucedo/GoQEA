# GoQEA
Golang implementation of the Quantum Inspired Optimizer developed by [@ferwanguer](https://github.com/ferwanguer).

You can find the original Python implementation [here](https://github.com/ferwanguer/PyQEA)

> Library for Quantum Inspired optimization in Go

GoQEA is an extensive research library for Quantum inspired hyper-normal based
optimization in Golang. 

It is intended for the solution of global optimization problems where conventional 
genetic algorithms or PSO yield sub-optimal results. The current implementation of 
the algorithm allows for a fast deployment of any optimization problem, regardless of the non-linearity of its 
constraints or the complexity of the cost function. The library has the following features:

## Features developed in GoQEA with respect to PyQEA
* High level module for Quantum Inspired optimization :heavy_check_mark:
* Built-in set of objective cost functions to test the optimization algorithm :heavy_check_mark:
* Capacity to implement non-linear restrictions :x:
* Capcity to implement integral-only variables :x:

### Basic Usage: 
GoQEA provides a high level implementation of  the proposed Quantum Inspired algorithm that allows a fast implementation and usage.
It aims to be user-friendly despite the non-trivial nature of its hyper-parameters. We now show the optimization process of a paraboloid (Sphere function)
of input dimension `n` centered in the vector: `[3.8, 3.8, 3.8, 3.8, ...]`. 

### Use case example: 
The optimizer setup is as follows:
```golang
package main

import (
	"GoQEA/goqea"
	"log"
	"time"
)

func main() {
	// Use case parameters
	const n_dims int = 10
	var upper_bounds [n_dims]float64
	var lower_bounds [n_dims]float64

	for i := 0; i < n_dims; i++ {
		upper_bounds[i] = 5.12
		lower_bounds[i] = -5
	}

	var mu_scaler float64 = 20
	var sigma_scaler float64 = 1.003
	var elitist_level int = 6
	var n_iterations int = 4000
	var n_samples int = 200
	// ------------------------------

	qea := goqea.NewQuantumEvAlgorithm(n_dims, sigma_scaler, mu_scaler, elitist_level, upper_bounds[:], lower_bounds[:], goqea.F)

	start := time.Now()
	qea.Training(n_iterations, n_samples)
	elapsed := time.Since(start)
	log.Printf("Took %s\n", elapsed)

}

```
### Parameter tuning
The main limitation that the user may encounter in the use of this optimizer is
the non-trivial character of it's hyper-parameters. The critical hyper-parameters
are the ones that regulate the update of hyper-normal distribution after the evaluation
of the sampled population. This is:

![tempsnip](https://user-images.githubusercontent.com/57362874/195801476-4f99a3cc-3063-4c20-b8fa-3eef63483fa6.png)

more information about the nature of this parameters, it's justification and experimental
results is to be released in the future.

The recommended rule of thumb is the following: 

* `mu_scaler ~ 20` (It is not as critical for performance)
* `sigma_scaler ~ (1 + 1/(10*n))` being `n` the number of input dimensions of the problem

The key concept to bear in mind is that, as the dimensionality of the problem increases, it is necessary to make the algorithm more "cautious", therefore minimizing the difference between before and after distributions. In practical terms, as the complexity of a given
problem increases, sigma_scaler must tend to ~1.
