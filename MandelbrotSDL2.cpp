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
        uint8_t t = iter / MAX_ITER; 
        uint8_t r = (uint8_t)(128.0 + 127.0 * sin(6.28318 * t + 0));
        uint8_t g = (uint8_t)(128.0 + 127.0 * sin(6.28318 * t + 2.09439)); // 2π/3 phase shift
        uint8_t b = (uint8_t)(128.0 + 127.0 * sin(6.28318 * t + 4.18879)); // 4π/3 phase shift
        return {r, g, b};
    }
}

// Calculates if a complex number c is in MB set and returns number of iters before leaving set
int mandelbrot(std::complex<double> c) {
    std::complex<double> z = 0;
    int iter = 0;

    while ((z.real() * z.real() + z.imag() * z.imag()) <= 2 && iter < MAX_ITER) {
        z = z * z + c;
        iter++;
    }

    return iter;
}


// Defining 64 bit double AVX registers, so each ymm register can fit 4 of them at once to do SIMD
// We are doing the parallelism by doing 4 c's at a time. 
__m256d _zr, _zi, _ca, _cb, _a, _b, _zr2, _zi2, _two, _four, _mask1; 

// 64 bit integer AVX registers
__m256i _n, _iter, _mask2, _c, _one; 

// Calc c in MB set using AVX256 intrinsics, doing 4 different cs at once 
void mandelbrotAVX(double offsetX, double offsetY, double zoom, const int iterations, Uint32* pixels, int bytesPerRow) {
    /*
        Normally we have: z_n+1 = (z_n * z_n) + c
        And we store z_n+1 and z_n in the same variable


        But we have to unroll this statement some more and do it more manually
        The maths for this work out to be:
        
        z = zr + zi * i
        c = c_a + c_bi
        z * z = zr * zr + 2*zr*zi*i + zi * zi * i * i
        z * z = zr^2 - zi^2 + 2* zr* zi*i
        z * z + c = (zr^2 - zi^2 + c_a) + (2* zr*zi*i + c_bi)
        New a = (zr^2 - zi^2 + c_a) 
        New b = (2*zr*zi*i + c_bi)
        Feed these a and b into next iteration so new zr = a, zi = b. 
    */
    _four = _mm256_set1_pd(4.0); 

   for (int i = 0; i < SCREEN_HEIGHT; i++) {
        for (int j = 0; j < SCREEN_WIDTH; j+=4) {
           
            repeat: 
            // Init registers for max iterations for comparision
            _iter = _mm256_set1_epi64x(iterations); 
            _one = _mm256_set1_epi64x(1); 

            // Init c and z registers
            // For c: do 4 in x direction at a time, y is gonna be same for all of them
            _ca = _mm256_set_pd(
                (j + 3 - SCREEN_WIDTH / 2.0) * 4.0 / (SCREEN_WIDTH * zoom) + offsetX,
                (j + 2 - SCREEN_WIDTH / 2.0) * 4.0 / (SCREEN_WIDTH * zoom) + offsetX,
                (j + 1 - SCREEN_WIDTH / 2.0) * 4.0 / (SCREEN_WIDTH * zoom) + offsetX,
                (j + 0 - SCREEN_WIDTH / 2.0) * 4.0 / (SCREEN_WIDTH * zoom) + offsetX
            );

            _cb = _mm256_set1_pd(((i - SCREEN_HEIGHT / 2.0) * 4.0 / (SCREEN_HEIGHT * zoom) + offsetY));
            _zr = _mm256_setzero_pd();
            _zi = _mm256_setzero_pd(); 


            // In increments of 4, we are doing these calculations for our new z
            _zr2 = _mm256_mul_pd(_zr, _zr); 
            _zi2 = _mm256_mul_pd(_zi, _zi); 

            _a = _mm256_sub_pd(_zr2, _zi2); 
            _a = _mm256_add_pd(_a, _ca); 

            _b = _mm256_mul_pd(_zr, _zi); 
            // For adding constants the register we are adding with needs to look like this
            // |2.0|2.0|2.0|2.0|
            _two = _mm256_set1_pd(2.0); 

            // Can do multiply and add at same time
            _b = _mm256_fmadd_pd(_b, _two, _cb); // mult b and 2 then add with cb

            _zr = _a;
            _zi = _b; 

            // Loop time, which is difficlt in AVX because some of the operations need to have increased iters, and some dont
            // Checking (zr^2 + zi^2) < 4 and iters < max iters
            // Re use the _a and _b for this purpose
            _a = _mm256_add_pd(_zr2, _zi2); 

            // Return of a comparison in AVX is a mask similar to:
            // | 1111..| 0000...| 1111...| 0000...|
            // We did | a[3] < 4 | a[2] < 4 | a[1] < 4 | a[0] < 4|
            _mask1 = _mm256_cmp_pd(_a, _four, _CMP_LT_OQ); 

            // mask 2 is for comparing current iters < max iters
            // gonna look similar to mask1 even tho this is int register and not double
            // Does a greater than comparison, cant find less than
            _mask2 = _mm256_cmpgt_epi64(_iter, _n);

            // Since these registers mask1 and mask2 have the same structure, we can still AND them 
            // We are using AND because condition for mandelbrot set is n < iter AND abs(z) < 2
            // This will tell us which c values we are currently calc at this iter are still in MB set
            // Convert mask1 into integer register by casting
            _mask2 = _mm256_and_si256(_mask2, _mm256_castpd_si256(_mask1)); 

            // c = |(int)1|(int)1|(int)1|(int)1| AND mask2 -> make new register that contains just 1
            // c = |000..1|000..1|000..1|000..1| 
            // m2= |000..0|000..0|1111.1|000..0| 
            // AND=|000..0|000..0|0000.1|000..0| 
            _c = _mm256_and_si256(_one, _mask2); 

            // Increment iterations by 1 by adding one
            _n = _mm256_add_epi64(_n, _c); 


            // Use goto repeat because we dont want program to terminate if one of the 4 elements
            // meets the while condition, only if all 4, so we need to keep going 
            // We know which elements are still processing if their mask is all 1's

            // If our while condition is true then go to repeat
            // if mask has any elements that are 1, then the bit sequence > 0 and we goto repeat
            if (_mm256_movemask_pd(_mm256_castsi256_pd(_mask2)) > 0) {
                goto repeat; 
            }

            // _n now contains number of iterations for a pixel we are processing, which we will visualize
                
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

        // Draw mandelbrot and manipulate pixels directly
        // Uint32* pixelData = static_cast<Uint32*>(pixels);
        // for (int i = 0; i < SCREEN_HEIGHT; i++) {
        //     for (int j = 0; j < SCREEN_WIDTH; j++) {
        //         // Map pixel to complex plane

        //         /*
        //             j - ScreenWIdth / 2.0 lets us center the x coordinate. Because when we iterate over pixels
        //             the top left is 0,0 and bottom right is SW - 1, SW - 1. So doing this puts it in range 
        //             of -SW, SW. 

        //             THen doing * 4 allows us to scale the range by 4 which is useful for next step
        //             THen doing / (SW * zoom) takes the scaled range and puts it in MB range of -2, 2, and multiplying that by zoom increases or decreases range
        //             Then adding offset pans the screen left or right
        //         */
        //         std::complex<double> c((j - SCREEN_WIDTH / 2.0) * 4.0 / (SCREEN_WIDTH * zoom) + offsetX,
        //                                (i - SCREEN_HEIGHT / 2.0) * 4.0 / (SCREEN_HEIGHT * zoom) + offsetY);
        //         int iterations = mandelbrot(c);
        //         Color color = getColor(iterations);

        //         // Set pixel color bytesPerRow / 4 tells us how many 32 bit colors are in that row, and we increment by them instead of bytes
        //         pixelData[i * (bytesPerRow / 4) + j] = (color.r << 24) | (color.g << 16) | (color.b << 8) | 0xFF;
        //     }
        // }

        // Draw mandelbrot and manipulate pixels directly using simd extensions
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
