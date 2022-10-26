package goqea

import (
	"log"
	"math"
	"math/rand"
	"sort"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

type QuantumEvAlgorithm struct {
	cost_function func(x *mat.Dense) *mat.VecDense
	sigma_scaler  *mat.VecDense
	mu_scaler     *mat.VecDense
	upper         *mat.VecDense
	lower         *mat.VecDense
	best_of_best  *mat.VecDense
	n_dims        int
	elitist_level int
}

type Individual struct {
	mu    *mat.VecDense
	sigma *mat.VecDense
}

type IndexedValue struct {
	value float64
	idx   int
}

func custom_random_normal_matrix(n_samples int, n_dims int, mus mat.VecDense, sigmas mat.VecDense) *mat.Dense {

	random_normal_matrix := mat.NewDense(n_samples, n_dims, nil)

	for i := 0; i < n_dims; i++ {
		values := make([]float64, n_samples)
		dimension_sigma := sigmas.AtVec(i)
		dimension_mu := mus.AtVec(i)

		for j := range values {
			values[j] = (rand.NormFloat64() * dimension_sigma) + dimension_mu
		}
		random_normal_matrix.SetCol(i, values)
	}

	return random_normal_matrix
}

func NewQuantumEvAlgorithm(n_dims int, sigma_scaler float64,
	mu_scaler float64, elitist_level int, upper []float64, lower []float64, cost_function func(x *mat.Dense) *mat.VecDense) *QuantumEvAlgorithm {
	if n_dims != len(upper) || n_dims != len(lower) {
		panic("n_dims missmatch with upper and lower bounds")
	}

	mu_scaler_vect := mat.NewVecDense(n_dims, nil)
	sigma_scaler_vect := mat.NewVecDense(n_dims, nil)
	for i := 0; i < n_dims; i++ {
		mu_scaler_vect.SetVec(i, mu_scaler)
		sigma_scaler_vect.SetVec(i, sigma_scaler)
	}

	upper_vect := mat.NewVecDense(n_dims, upper)
	lower_vect := mat.NewVecDense(n_dims, lower)

	return &QuantumEvAlgorithm{n_dims: n_dims, sigma_scaler: sigma_scaler_vect, mu_scaler: mu_scaler_vect,
		elitist_level: elitist_level, upper: upper_vect, lower: lower_vect, best_of_best: nil, cost_function: cost_function}
}

func (qea *QuantumEvAlgorithm) NewIndividual() *Individual {

	// Mus
	mus_vector := mat.NewVecDense(qea.n_dims, nil)
	for i := 0; i < qea.n_dims; i++ {
		mus_vector.SetVec(i, rand.Float64())
	}
	mus_vector.MulElemVec(mus_vector, qea.upper)
	mus_vector.AddVec(mus_vector, qea.lower)

	// Sigmas
	sigma_vector := mat.NewVecDense(qea.n_dims, nil)
	for i := 0; i < qea.n_dims; i++ {
		sigma_vector.SetVec(i, qea.upper.AtVec(i)-qea.lower.AtVec(i))
	}

	ind := Individual{mu: mus_vector, sigma: sigma_vector}

	qea.best_of_best = mus_vector

	return &ind
}

func (qea *QuantumEvAlgorithm) QuantumSampling(ind Individual, n_samples int) *mat.Dense {

	raw_samples := custom_random_normal_matrix(n_samples, qea.n_dims, *ind.mu, *ind.sigma)

	for dimension := 0; dimension < qea.n_dims; dimension++ {
		max_of_dimension := qea.upper.AtVec(dimension)
		min_of_dimension := qea.lower.AtVec(dimension)

		for sample_row := 0; sample_row < n_samples; sample_row++ {
			value_to_test := raw_samples.At(sample_row, dimension)

			switch {
			case value_to_test > max_of_dimension:
				raw_samples.Set(sample_row, dimension, max_of_dimension)
			case value_to_test < min_of_dimension:
				raw_samples.Set(sample_row, dimension, min_of_dimension)
			}
		}
	}

	return raw_samples
}

func (qea *QuantumEvAlgorithm) ElitistSampleEvaluation(samples *mat.Dense) *mat.VecDense {

	cost := qea.cost_function(samples)
	vect_len := cost.Len()

	/*---- Argsort ------*/
	indexed_vector := make([]IndexedValue, vect_len)

	for i := 0; i < vect_len; i++ {
		indexed_vector[i] = IndexedValue{i, cost.AtVec(i)}
	}

	sort.SliceStable(indexed_vector, func(i, j int) bool { return indexed_vector[i].value < indexed_vector[j].value })
	/*------------------*/

	/* --- Elit of indexes sorted by cost ---*/
	elit := make([]int, qea.elitist_level)

	for idx, i := range indexed_vector[0:qea.elitist_level] {
		elit[idx] = int(i.idx)
	}
	/*---------------------------------------*/

	/*----- Elit samples -----*/

	best_samples_t := mat.NewDense(qea.n_dims, qea.elitist_level, nil)
	for sample_idx, elit_idx := range elit {
		best_samples_t.SetCol(sample_idx, samples.RawRowView(elit_idx))
	}
	/* -----------------------*/

	/* ---- Mean of elit samples ---- */
	best_performing_sample := make([]float64, qea.n_dims)

	for i := 0; i < qea.n_dims; i++ {
		best_performing_sample[i] = stat.Mean(best_samples_t.RawRowView(i), nil)
	}
	/* ------------------------------ */

	return mat.NewVecDense(qea.n_dims, best_performing_sample)
}

func (qea *QuantumEvAlgorithm) QuantumUpdate(indiv *Individual, best_performing_sample *mat.VecDense) {

	// MU
	mu_delta := mat.VecDenseCopyOf(indiv.mu)
	mu_delta2 := mat.VecDenseCopyOf(indiv.mu)

	mu_delta.SubVec(best_performing_sample, mu_delta) // mu_delta = best_performing_sample - mu
	mu_delta2.SubVec(qea.best_of_best, mu_delta2)     // mu_delta_2 = best_of_best - mu
	mu_delta2.AddVec(mu_delta, mu_delta2)             // mu_delta_2 = mu_delta + mu_delta_2
	mu_delta2.DivElemVec(mu_delta2, qea.mu_scaler)    // mu_delta_2 = mu_delta_2 / mu_scaler
	mu_delta2.AddVec(mu_delta2, indiv.mu)             // mu_delta_2 = mu_delta_2 + mu

	// DELTA
	sigma_decider := mat.VecDenseCopyOf(mu_delta)
	for i := 0; i < qea.n_dims; i++ {
		sigma_decider.SetVec(i, math.Abs(mu_delta.AtVec(i))) // sigma_decider = abs(mu_delta)
	}
	sigma_decider.DivElemVec(sigma_decider, indiv.sigma) // sigma_decider = sigma_decider / sigma

	updated_sigma := mat.NewVecDense(qea.n_dims, nil)

	for i := 0; i < qea.n_dims; i++ {
		if sigma_decider.AtVec(i) <= 1 { // sigma_decider <= 1
			updated_sigma.SetVec(i, indiv.sigma.AtVec(i)/qea.sigma_scaler.AtVec(i)) // updated_sigma[i] = sigma/sigma_scaler
		} else { // sigma_decider > 1
			updated_sigma.SetVec(i, indiv.sigma.AtVec(i)*qea.sigma_scaler.AtVec(i)) // updated_sigma[i] = sigma * sigma_scaler
		}
	}

	indiv.mu = mu_delta2
	indiv.sigma = updated_sigma

}

func (qea *QuantumEvAlgorithm) Training(n_iterations int, sample_size int) {

	if qea.elitist_level > sample_size {
		panic("Sample size must be greater than elitist level")
	}

	q_indiv := qea.NewIndividual()

	for i := 0; i < n_iterations+1; i++ {
		samples := qea.QuantumSampling(*q_indiv, sample_size)

		if i > n_iterations-1 {
			qea.elitist_level = 1
		}

		best_performer := qea.ElitistSampleEvaluation(samples)
		if F_single(best_performer) < F_single(qea.best_of_best) {
			qea.best_of_best = best_performer
		}

		qea.QuantumUpdate(q_indiv, best_performer)

		if i%100 == 0 {
			log.Printf("Epoch %d: %.6f\n", i, F_single(best_performer))
		}
	}

}
