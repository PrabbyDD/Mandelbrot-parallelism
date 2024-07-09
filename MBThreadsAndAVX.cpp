#include <thread>
#include <vector>
#include <SDL2/SDL.h>
#include <complex>
#include <iostream>
#include <immintrin.h>

/*
    Because the threads dont need to share data here, we dont really need to use mutex
    We will partition the screen into equal rectangles, 1 for each thread we want. 

    Not every thread will end at the same time, because since we are dividing the regions on the screen
    to each thread, some threads will end their loops early in the iterations because they reach 
    the constraint fast, and some will be taking way longer, thus bottlenecking a bit.
    Can be solved a bit more with thread pools. 
*/

const int SCREEN_WIDTH = 800;
const int SCREEN_HEIGHT = 600;
const int MAX_ITER = 1000;
const int THREAD_COUNT = 32; 

struct Color {
    uint8_t r, g, b;
};

// Maps an iteration to a color
Color getColor(int iter) {
    if (iter == MAX_ITER) {
        return {0, 0, 0};
    } else {
        double t = static_cast<double>(iter) / MAX_ITER;
        uint8_t r = static_cast<uint8_t>(128.0 + 127.0 * sin(6.28318 * t + 0));
        uint8_t g = static_cast<uint8_t>(128.0 + 127.0 * sin(6.28318 * t + 2.09439)); // 2π/3 phase shift
        uint8_t b = static_cast<uint8_t>(128.0 + 127.0 * sin(6.28318 * t + 4.18879)); // 4π/3 phase shift
        return {r, g, b};
    }
}

// Calculates if a complex number c is in MB set and returns number of iters before leaving set
int mandelbrot(std::complex<double> c) {
    std::complex<double> z = 0;
    int iter = 0;

    while (std::norm(z) <= 4.0 && iter < MAX_ITER) {
        z = z * z + c;
        iter++;
    }

    return iter;
}

// Defining 64-bit double AVX registers, so each ymm register can fit 4 of them at once to do SIMD
__m256d _zr, _zi, _ca, _cb, _a, _b, _zr2, _zi2, _two, _four, _mask1;

// 64-bit integer AVX registers
__m256i _n, _iter, _mask2, _c, _one;

// Calc c in MB set using AVX256 intrinsics, doing 4 different cs at once 
// Calc c in MB set using AVX256 intrinsics, doing 4 different cs at once
void mandelbrotAVX(double offsetX, double offsetY, double zoom, const int max_iterations, Uint32* pixels, int bytesPerRow, int startY, int endY) {
    __m256d _four = _mm256_set1_pd(4.0);
    __m256d _two = _mm256_set1_pd(2.0);
    __m256i _one = _mm256_set1_epi64x(1);

    for (int i = startY; i < endY; ++i) {
        for (int j = 0; j < SCREEN_WIDTH; j += 4) {
            __m256i _iter = _mm256_set1_epi64x(0);
            __m256d _ca = _mm256_set_pd(
                (j + 3 - SCREEN_WIDTH / 2.0) * 4.0 / (SCREEN_WIDTH * zoom) + offsetX,
                (j + 2 - SCREEN_WIDTH / 2.0) * 4.0 / (SCREEN_WIDTH * zoom) + offsetX,
                (j + 1 - SCREEN_WIDTH / 2.0) * 4.0 / (SCREEN_WIDTH * zoom) + offsetX,
                (j + 0 - SCREEN_WIDTH / 2.0) * 4.0 / (SCREEN_WIDTH * zoom) + offsetX
            );
            __m256d _cb = _mm256_set1_pd((i - SCREEN_HEIGHT / 2.0) * 4.0 / (SCREEN_HEIGHT * zoom) + offsetY);
            __m256d _zr = _mm256_setzero_pd();
            __m256d _zi = _mm256_setzero_pd();

            for (int k = 0; k < max_iterations; ++k) {
                __m256d _zr2 = _mm256_mul_pd(_zr, _zr);
                __m256d _zi2 = _mm256_mul_pd(_zi, _zi);

                __m256d _abs = _mm256_add_pd(_zr2, _zi2);
                __m256d _mask1 = _mm256_cmp_pd(_abs, _four, _CMP_LT_OQ);
                __m256i _mask2 = _mm256_castpd_si256(_mask1);
                if (_mm256_testz_si256(_mask2, _mask2)) break;

                _zi = _mm256_add_pd(_mm256_mul_pd(_two, _mm256_mul_pd(_zr, _zi)), _cb);
                _zr = _mm256_add_pd(_mm256_sub_pd(_zr2, _zi2), _ca);

                _iter = _mm256_add_epi64(_iter, _one);
            }

            for (int k = 0; k < 4; ++k) {
                int iter = reinterpret_cast<int64_t*>(&_iter)[3 - k];
                Color color = getColor(iter);
                pixels[i * (bytesPerRow / 4) + j + k] = (color.r << 24) | (color.g << 16) | (color.b << 8) | 0xFF;
            }
        }
    }
}
void mandelbrotThreads(double offsetX, double offsetY, double zoom, const int max_iterations, Uint32* pixels, int bytesPerRow) {
    std::vector<std::thread> threads; 
    int rows_per_thread = SCREEN_HEIGHT / THREAD_COUNT;

    for (int i = 0; i < THREAD_COUNT; ++i) {
        int startY = i * rows_per_thread;
        int endY = (i == THREAD_COUNT - 1) ? SCREEN_HEIGHT : (i + 1) * rows_per_thread;

        threads.emplace_back(mandelbrotAVX, offsetX, offsetY, zoom, max_iterations, pixels, bytesPerRow, startY, endY);
    }

    for (auto& thread : threads) {
        thread.join();
    }
}
int main() {
    // Init SDL2
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL could not start! Error: " << SDL_GetError() << std::endl;
        return 1;
    }

    // Create window
    SDL_Window* window = SDL_CreateWindow("Mandelbrot Set", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
    if (!window) {
        std::cerr << "Window could not be made! Error: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return 1;
    }

    // Create renderer
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        std::cerr << "Renderer could not be made! Error: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    // Create texture
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, SCREEN_WIDTH, SCREEN_HEIGHT);
    if (!texture) {
        std::cerr << "Texture could not be made! Error: " << SDL_GetError() << std::endl;
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    // Initial zoom and position
    double zoom = 1.0;
    double offsetX = -0.5;
    double offsetY = 0.0;

    // Main loop flags
    bool quit = false;
    SDL_Event e;

    // Main loop
    while (!quit) {
        // Handle events
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) {
                quit = true;
            } else if (e.type == SDL_KEYDOWN) {
                switch (e.key.keysym.sym) {
                    case SDLK_UP:
                        offsetY -= 0.1 / zoom;
                        break;
                    case SDLK_DOWN:
                        offsetY += 0.1 / zoom;
                        break;
                    case SDLK_LEFT:
                        offsetX -= 0.1 / zoom;
                        break;
                    case SDLK_RIGHT:
                        offsetX += 0.1 / zoom;
                        break;
                    case SDLK_PLUS:
                    case SDLK_EQUALS:
                        zoom *= 1.1;
                        break;
                    case SDLK_MINUS:
                        zoom /= 1.1;
                        break;
                }
            } else if (e.type == SDL_MOUSEWHEEL) {
                if (e.wheel.y > 0) { // upscroll
                    zoom *= 1.1;
                } else if (e.wheel.y < 0) { // scroll down
                    zoom /= 1.1;
                }
            }
        }

        // Lock texture for manipulation
        void* pixels;
        int bytesPerRow;
        SDL_LockTexture(texture, NULL, &pixels, &bytesPerRow);

        // Draw mandelbrot and manipulate pixels directly using SIMD
        Uint32* pixelData = static_cast<Uint32*>(pixels);
        mandelbrotThreads(offsetX, offsetY, zoom, MAX_ITER, pixelData, bytesPerRow);

        SDL_UnlockTexture(texture);

        // Clear screen
        SDL_SetRenderDrawColor(renderer, 0x00, 0x00, 0x00, 0xFF);
        SDL_RenderClear(renderer);

        // Copy texture to renderer
        SDL_RenderCopy(renderer, texture, NULL, NULL);

        // Update screen
        SDL_RenderPresent(renderer);
    }

    // Destroy texture, renderer, and window
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
