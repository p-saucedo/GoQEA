package goqea

import (
	"fmt"
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestElitistSampleEvaluation(t *testing.T) {
	const n_dims int = 2
	upper_bounds := [n_dims]float64{}
	lower_bounds := [n_dims]float64{}

	for i := 0; i < n_dims; i++ {
		upper_bounds[i] = 5.12
		lower_bounds[i] = -5
	}

	qea := NewQuantumEvAlgorithm(n_dims, 1.003, 20, 2, upper_bounds[:], lower_bounds[:], F)

	var tests = []struct {
		input *mat.Dense
		want  *mat.VecDense
	}{
		{mat.NewDense(4, 2, []float64{4.1, 4.1, 4.0, 4.0, 3.9, 3.9, 3.8, 3.8}), mat.NewVecDense(2, []float64{3.85, 3.85})},
	}

	for _, tt := range tests {
		testname := fmt.Sprintf("%v", tt.input)

		t.Run(testname, func(t *testing.T) {
			ans := qea.ElitistSampleEvaluation(tt.input)

			if (ans.AtVec(0) - tt.want.AtVec(0)) > 0.01 {
				t.Errorf("got %v, want %v", ans, tt.want)
			}
		})
	}
}

func TestQuantumUpdate(t *testing.T) {
	const n_dims int = 2
	upper_bounds := [n_dims]float64{}
	lower_bounds := [n_dims]float64{}

	for i := 0; i < n_dims; i++ {
		upper_bounds[i] = 5.12
		lower_bounds[i] = -5
	}

	qea := NewQuantumEvAlgorithm(n_dims, 1.003, 20, 2, upper_bounds[:], lower_bounds[:], F)
	indiv := Individual{mu: mat.NewVecDense(2, []float64{0.5, 0.5}), sigma: mat.NewVecDense(2, []float64{10, 10})}
	qea.best_of_best = mat.NewVecDense(2, []float64{0, 0})
	var tests = []struct {
		best_mu             float64
		want_mu, want_sigma float64
	}{
		{1.2, 0.51, 10.03},
		{0.2, 0.46, 9.97},
	}

	for _, tt := range tests {
		testname := fmt.Sprintf("best%f", tt.best_mu)

		t.Run(testname, func(t *testing.T) {
			best := make([]float64, n_dims)
			for i := range best {
				best[i] = tt.best_mu
			}
			qea.QuantumUpdate(&indiv, mat.NewVecDense(2, best))

			if (math.Abs(indiv.mu.AtVec(0)-tt.want_mu) > 0.1) || (math.Abs(indiv.sigma.AtVec(0)-tt.want_sigma) > 0.1) {
				t.Errorf("got %f, want %f", indiv.mu.AtVec(0), tt.want_mu)
				t.Errorf("got %f, want %f", indiv.sigma.AtVec(0), tt.want_sigma)
			}
		})
	}
}

func TestQuantumSampling(t *testing.T) {
	const n_dims int = 2
	upper_bounds := [n_dims]float64{}
	lower_bounds := [n_dims]float64{}

	for i := 0; i < n_dims; i++ {
		upper_bounds[i] = 5.12
		lower_bounds[i] = -5
	}

	qea := NewQuantumEvAlgorithm(n_dims, 1.003, 20, 2, upper_bounds[:], lower_bounds[:], F)
	indiv := Individual{mu: mat.NewVecDense(2, []float64{0.5, 0.5}), sigma: mat.NewVecDense(2, []float64{10, 10})}
	qea.best_of_best = mat.NewVecDense(2, []float64{0, 0})
	var tests = []struct {
		best_mu             float64
		want_mu, want_sigma float64
	}{
		{1.2, 0.51, 10.03},
		{0.2, 0.46, 9.97},
	}

	for _, tt := range tests {
		testname := fmt.Sprintf("best%f", tt.best_mu)

		t.Run(testname, func(t *testing.T) {
			best := make([]float64, n_dims)
			for i := range best {
				best[i] = tt.best_mu
			}
			qea.QuantumUpdate(&indiv, mat.NewVecDense(2, best))

			if (math.Abs(indiv.mu.AtVec(0)-tt.want_mu) > 0.1) || (math.Abs(indiv.sigma.AtVec(0)-tt.want_sigma) > 0.1) {
				t.Errorf("got %f, want %f", indiv.mu.AtVec(0), tt.want_mu)
				t.Errorf("got %f, want %f", indiv.sigma.AtVec(0), tt.want_sigma)
			}
		})
	}
}
