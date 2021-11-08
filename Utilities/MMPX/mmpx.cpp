#include "../stdafx.h"
#include <stdint.h>
#include "mmpx.h"

void mmpx_scale(const unsigned scale, const uint32_t* src, 
uint32_t* dst, const unsigned width, const unsigned height) {
    uint32_t* outputBuffer = dst;

	for (uint32_t y = 0; y < height; y++) {
		for(uint32_t x = 0; x < width; x++) {
			for(uint32_t i = 0; i < scale; i++) {
				*(outputBuffer++) = *src;
			}
			src++;
		}
		for(uint32_t i = 1; i < scale; i++) {
			memcpy(outputBuffer, outputBuffer - width*scale, width*scale*4);
			outputBuffer += width*scale;
		}
	}
}