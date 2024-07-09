#include <iostream>
#include <complex>

const int width = 80;
const int height = 80;
const int max_iter = 1000; 

// Checks if c in z_n = (z_n-1)^2 + c is in mandelbrot, starting iter with z_0 = 0
int mandelbrot(std::complex<double> c) {
    std::complex<double> z = 0; 
    int iterations = 0; 

    // While it resides in the mandelbrot set
    while (std::abs(z) < 2.0 && iterations < max_iter) {
        z = z * z + c;
        iterations++;
    }

    // Use iterations before it leaves mandelbrot set as pixel color
    return iterations; 
}

int main() {
    

    // Range of complex plane on display
    double min_re = -2.0, max_re = 1.0;
    double min_im = -1.0, max_im = 1.0;

    // Step size of each pixel, cant have infinite resolution!
    double re_step = (max_re - min_re) / width;
    double im_step = (max_im - min_im) / height;

    // Go over each pixel and write its color
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            std::complex<double> c(min_re + j * re_step, min_im + i * im_step);
            int iters = mandelbrot(c);

            // If it reached max iters, it is currently in mandelbrot set
            if (iters == max_iter) {
                std::cout << "#";
            } else {
                std::cout << "*"; 
            }
        }
        std::cout << std::endl;    
    }
    return 0; 
}