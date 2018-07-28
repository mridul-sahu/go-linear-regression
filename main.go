package main

import (
	"log"
	"os"
	"gonum.org/v1/plot/plotter"
	"bufio"
	"fmt"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/vg/draw"
	"golang.org/x/image/colornames"
	"path/filepath"
	"math"
	"flag"
)

type xy struct {
	x, y float64
}

var numIterations int

func main() {
	flag.IntVar(&numIterations, "n" ,1000, "number of iterations")
	flag.Parse()

	// Read the data from the file
	xys, err := readData("data.txt")

	if err != nil {
		log.Fatalf("could not read data: %v", err)
	}


	// Plot the data to the file
	err = plotData(xys, "output.png")

	if err != nil {
		log.Fatalf("could not plot data: %v", err)
	}
}

func readData(path string) (plotter.XYs, error) {

	f, err := os.Open(path)

	if err != nil {
		return nil, err
	}

	defer f.Close()

	var xys plotter.XYs

	s := bufio.NewScanner(f)

	for s.Scan() {
		var x, y float64

		_, err = fmt.Sscanf(s.Text(), "%f,%f", &x, &y)

		// Discard bad points
		if err != nil {
			log.Printf("discarding bad data point %q: %v", s.Text(), err)
			continue
		}

		// Add the point to plotter.XYs
		xys = append(xys, struct{ X, Y float64 }{x, y})
	}

	//Check for error while scanning file
	if err := s.Err(); err != nil {
		return nil, fmt.Errorf("could not scan file: %v", err)
	}

	return xys, nil
}

func plotData(xys plotter.XYs, path string) error {

	f, err := os.Create(path)

	if err != nil {
		return fmt.Errorf("could not create path %q: %v", path, err)
	}

	// Won't do defer as will check for error while closing
	// defer f.Close()

	p, err := plot.New()

	p.Title.Text = "Random Plot"
	p.X.Label.Text = "X values"
	p.Y.Label.Text = "Y values"


	if err != nil {
		return fmt.Errorf("could not create plot: %v", err)
	}

	s, err := plotter.NewScatter(xys)

	if err != nil {
		return fmt.Errorf("could not create scatter: %v", err)
	}

	s.Shape = draw.CrossGlyph{}

	s.Color = colornames.Red

	// Add scatter to the plot
	p.Add(s)

	// Get line using linear regression
	m, c := linearRegression(xys, 0.01, numIterations)

	minX := math.MaxFloat64
	maxX := -math.MaxFloat64

	// Get minimum and maximum x values in our dataset to plot the line
	for _, xy := range xys {

		if xy.X < minX {
			minX = xy.X
		}

		if xy.X > maxX {
			maxX = xy.X
		}
	}

	// line plotter using the minimum and maximum value of x
	l, err := plotter.NewLine(plotter.XYs{
		{minX, minX * m + c},
		{maxX, maxX * m + c},
	})

	if err != nil {
		return fmt.Errorf("could not create line: %v", err)
	}

	// Add line to plot
	p.Add(l)

	wt, err := p.WriterTo(256, 256, filepath.Ext(path)[1:]) // filepath.Ext(path)[1:] as need to ignore "." in ".png"

	if err != nil {
		return  fmt.Errorf("could not create writer to plot: %v", err)
	}

	_, err = wt.WriteTo(f)

	if err != nil {
		return fmt.Errorf("could not write to %q: %v", path, err)
	}

	if err := f.Close(); err != nil {
		return fmt.Errorf("could not close %q: %v", path, err)
	}

	return nil
}

func linearRegression(xys plotter.XYs, alpha float64, iterations int) (m, c float64) {

	for i := 0; i < iterations; i++ {
		// compute gradient
		dm, dc := computeGradient(xys, m, c)

		// Change values to minimize cost
		m += -dm * alpha //Opposite to the direction of the gradient
		c += -dc * alpha //Opposite to the direction of the gradient
	}

	log.Printf("final cost for (%f, %f): %f", m, c, computerCost(xys, m, c))

	return m, c
}

func computerCost (xys plotter.XYs, m, c float64) float64 {
	// cost = 1/N * sum((y - (m*x + c))^2)

	sum := 0.0

	for _, xy := range xys {
		d := xy.Y - (m * xy.X + c)
		sum += d * d
	}

	return sum/float64(len(xys))
}

func computeGradient(xys plotter.XYs, m, c float64) (dm, dc float64) {
	// cost/dm = 2/N * sum( -x * (y - (m*x + c)))
	// cost/dc = 2/N * sum(-1 * (y - (m*x + c)))

	for _, xy := range xys {
		d := xy.Y - (m * xy.X + c)
		dm += -xy.X * d
		dc += -d
	}

	n := float64(len(xys))

	dm /= n
	dc /= n

	return 2*dm, 2*dc
}
