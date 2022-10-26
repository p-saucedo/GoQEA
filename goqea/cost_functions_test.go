package goqea

import (
	"fmt"
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestFSingleTableDriven(t *testing.T) {
	var tests = []struct {
		input *mat.VecDense
		want  float64
	}{
		{mat.NewVecDense(3, []float64{3.8, 3.8, 3.8}), 0},
		{mat.NewVecDense(3, []float64{4, 4, 4}), 0.12},
	}
	for _, tt := range tests {
		testname := fmt.Sprintf("%v", tt.input)
		t.Run(testname, func(t *testing.T) {
			ans := F_single(tt.input)
			if math.Abs(ans-tt.want) > 0.001 {
				t.Errorf("got %f, want %f", ans, tt.want)
			}
		})
	}
}

func TestFTableDriven(t *testing.T) {
	var tests = []struct {
		input *mat.Dense
		want  *mat.VecDense
	}{
		{mat.NewDense(2, 2, []float64{3.8, 3.8, 3.8, 3.8}), mat.NewVecDense(2, []float64{0, 0})},
		{mat.NewDense(2, 2, []float64{4, 4, 4, 4}), mat.NewVecDense(2, []float64{0.08, 0.08})},
	}
	for _, tt := range tests {
		testname := fmt.Sprintf("%v", tt.input)
		t.Run(testname, func(t *testing.T) {
			ans := F(tt.input)
			aux := mat.VecDenseCopyOf(ans)
			aux.SubVec(aux, tt.want)

			if aux.Norm(1) > 0.01 {
				t.Errorf("got %v, want %v", ans, tt.want)
			}
		})
	}
}
