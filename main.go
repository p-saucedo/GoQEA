package main

import (
	"GoQEA/goqea"
	"log"
	"time"
)

func main() {
	// Problem parametrs
	const n_dims int = 1000
	var upper_bounds [n_dims]float64
	var lower_bounds [n_dims]float64

	for i := 0; i < n_dims; i++ {
		upper_bounds[i] = 5.12
		lower_bounds[i] = -5
	}

	var mu_scaler float64 = 10
	var sigma_scaler float64 = 1.003
	var elitist_level int = 6
	var n_iterations int = 1000
	var n_samples int = 200
	// ------------------------------

	qea := goqea.NewQuantumEvAlgorithm(n_dims, sigma_scaler, mu_scaler, elitist_level, upper_bounds[:], lower_bounds[:], goqea.F)

	start := time.Now()
	qea.Training(n_iterations, n_samples)
	elapsed := time.Since(start)
	log.Printf("Took %s\n", elapsed)

}
