package goqea

import (
	"gonum.org/v1/gonum/mat"
)

func F(x *mat.Dense) *mat.VecDense {
	aux_x := mat.DenseCopyOf(x)
	n_samples, n_dims := aux_x.Dims()

	min_values := mat.NewVecDense(n_dims, nil)
	for i := 0; i < n_dims; i++ {
		min_values.SetVec(i, 3.8)
	}

	result := make([]float64, n_samples)

	for i := 0; i < n_samples; i++ {
		sample := mat.NewVecDense(n_dims, aux_x.RawRowView(i))
		sample.SubVec(sample, min_values)
		sample.MulElemVec(sample, sample)
		result[i] = mat.Sum(sample)
	}

	return mat.NewVecDense(n_samples, result)
}

func F_single(x *mat.VecDense) float64 {
	n_dims, _ := x.Dims()

	min_values := mat.NewVecDense(n_dims, nil)
	for i := 0; i < n_dims; i++ {
		min_values.SetVec(i, 3.8)
	}

	aux_vector := mat.VecDenseCopyOf(x)
	aux_vector.SubVec(x, min_values)
	aux_vector.MulElemVec(aux_vector, aux_vector)
	result := mat.Sum(aux_vector)

	return result
}
