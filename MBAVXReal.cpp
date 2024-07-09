#include <SDL2/SDL.h>
#include <complex>
#include <iostream>
#include <immintrin.h>

const int SCREEN_WIDTH = 800;
const int SCREEN_HEIGHT = 600;
const int MAX_ITER = 1000;

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
void mandelbrotAVX(double offsetX, double offsetY, double zoom, const int max_iterations, Uint32* pixels, int bytesPerRow) {
    _four = _mm256_set1_pd(4.0);
    _two = _mm256_set1_pd(2.0);
    _one = _mm256_set1_epi64x(1);
    for (int i = 0; i < SCREEN_HEIGHT; i++) {
        for (int j = 0; j < SCREEN_WIDTH; j += 4) {
            // Init registers for max iterations for comparison
            _iter = _mm256_set1_epi64x(max_iterations);
            
            // Init c and z registers
            _ca = _mm256_set_pd(
                (j + 3 - SCREEN_WIDTH / 2.0) * 4.0 / (SCREEN_WIDTH * zoom) + offsetX,
                (j + 2 - SCREEN_WIDTH / 2.0) * 4.0 / (SCREEN_WIDTH * zoom) + offsetX,
                (j + 1 - SCREEN_WIDTH / 2.0) * 4.0 / (SCREEN_WIDTH * zoom) + offsetX,
                (j + 0 - SCREEN_WIDTH / 2.0) * 4.0 / (SCREEN_WIDTH * zoom) + offsetX
            );

            _cb = _mm256_set1_pd((i - SCREEN_HEIGHT / 2.0) * 4.0 / (SCREEN_HEIGHT * zoom) + offsetY);
            _zr = _mm256_setzero_pd();
            _zi = _mm256_setzero_pd();

            _n = _mm256_setzero_si256(); // Init iterations count

            for (int iter = 0; iter < MAX_ITER; iter++) {
                _zr2 = _mm256_mul_pd(_zr, _zr);
                _zi2 = _mm256_mul_pd(_zi, _zi);

                _a = _mm256_sub_pd(_zr2, _zi2);
                _a = _mm256_add_pd(_a, _ca);

                _b = _mm256_mul_pd(_zr, _zi);
                _b = _mm256_fmadd_pd(_two, _b, _cb); // mult b and 2 then add with cb

                _zr = _a;
                _zi = _b;

                _a = _mm256_add_pd(_zr2, _zi2);
                _mask1 = _mm256_cmp_pd(_a, _four, _CMP_LT_OQ);

                _mask2 = _mm256_cmpgt_epi64(_iter, _n);
                _mask2 = _mm256_and_si256(_mask2, _mm256_castpd_si256(_mask1));

                _c = _mm256_and_si256(_one, _mask2);
                _n = _mm256_add_epi64(_n, _c);

                if (_mm256_testz_si256(_mask2, _mask2)) break;
            }

            Color color1 = getColor(int(_n[3]));
            Color color2 = getColor(int(_n[2]));
            Color color3 = getColor(int(_n[1]));
            Color color4 = getColor(int(_n[0]));

            pixels[i * (bytesPerRow / 4) + j + 0] = (color1.r << 24) | (color1.g << 16) | (color1.b << 8) | 0xFF;
            pixels[i * (bytesPerRow / 4) + j + 1] = (color2.r << 24) | (color2.g << 16) | (color2.b << 8) | 0xFF;
            pixels[i * (bytesPerRow / 4) + j + 2] = (color3.r << 24) | (color3.g << 16) | (color3.b << 8) | 0xFF;
            pixels[i * (bytesPerRow / 4) + j + 3] = (color4.r << 24) | (color4.g << 16) | (color4.b << 8) | 0xFF;
        }
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
        mandelbrotAVX(offsetX, offsetY, zoom, MAX_ITER, pixelData, bytesPerRow);

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
