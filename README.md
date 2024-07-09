Inspired by one lone coders videos on youtube, I tried my hand at making the mandelbrot pattern in c++ and wanted to speed it up with compiler SIMD extensions and thread parallelism. I achieved around a 66x speedup by doing this from without.

commands on linux to generate exec: 
g++ -o MBThreads MBThreadsAndAVX.cpp `sdl2-config --cflags --libs` -mavx2 -mfma -pthread

