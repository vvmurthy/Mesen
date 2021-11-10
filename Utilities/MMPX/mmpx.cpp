#include "../stdafx.h"
#include <stdint.h>
#include <cmath>
#include "mmpx.h"

////////////////////////////////////////////////////////////////

//    return B == A0 && B == A1;
// Optimized (faster bitwise is more important than early out)
bool all_eq2(uint32_t B, uint32_t A0, uint32_t A1) {
    return ((B ^ A0) | (B ^ A1)) == 0;
}

//    return B == A0 && B == A1 && B == A2;
// Optimized (faster bitwise is more important than early out)
bool all_eq3(uint32_t B, uint32_t A0, uint32_t A1, uint32_t A2) {
    return ((B ^ A0) | (B ^ A1) | (B ^ A2)) == 0;
}

//    return B == A0 && B == A1 && B == A2 && B == A3;
// Optimized (faster bitwise is more important than early out)
bool all_eq4(uint32_t B, uint32_t A0, uint32_t A1, uint32_t A2, uint32_t A3) {
    return ((B ^ A0) | (B ^ A1) | (B ^ A2) | (B ^ A3)) == 0;
}

bool all_eq8(uint32_t B, uint32_t A0, uint32_t A1, uint32_t A2, uint32_t A3, uint32_t A4, uint32_t A5, uint32_t A6, uint32_t A7) {
    return ((B ^ A0) | (B ^ A1) | (B ^ A2) | (B ^ A3) | (B ^ A4) | (B ^ A5) | (B ^ A6) | (B ^ A7)) == 0;
}

bool all_eq9(uint32_t B, uint32_t A0, uint32_t A1, uint32_t A2, uint32_t A3, uint32_t A4, uint32_t A5, uint32_t A6, uint32_t A7, uint32_t A8) {
    return ((B ^ A0) | (B ^ A1) | (B ^ A2) | (B ^ A3) | (B ^ A4) | (B ^ A5) | (B ^ A6) | (B ^ A7) | (B ^ A8)) == 0;
}

////////////////////////////////////////////////////////////////
bool any_eq3(uint32_t B, uint32_t A0, uint32_t A1, uint32_t A2) {
    return B == A0 || B == A1 || B == A2;
}

bool any_eq4(uint32_t B, uint32_t A0, uint32_t A1, uint32_t A2, uint32_t A3) {
    return B == A0 || B == A1 || B == A2 || B == A3;
}

bool any_eq8(uint32_t B, uint32_t A0, uint32_t A1, uint32_t A2, uint32_t A3, uint32_t A4, uint32_t A5, uint32_t A6, uint32_t A7) {
    return (B == A0 || B == A1 || B == A2 || B == A3 || B == A4 || B == A5 || B == A6 || B == A7);
}

////////////////////////////////////////////////////////////////
bool none_eq2(uint32_t B, uint32_t A0, uint32_t A1) {
    return (B != A0) && (B != A1);
}

bool none_eq3(uint32_t B, uint32_t A0, uint32_t A1, uint32_t A2) {
    return (B != A0) && (B != A1) && (B != A2);
}

bool none_eq4(uint32_t B, uint32_t A0, uint32_t A1, uint32_t A2, uint32_t A3) {
    return B != A0 && B != A1 && B != A2 && B != A3;
}

bool none_eq5(uint32_t B, uint32_t A0, uint32_t A1, uint32_t A2, uint32_t A3, uint32_t A4) {
    return B != A0 && B != A1 && B != A2 && B != A3 && B != A4;
}

bool none_eq6(uint32_t B, uint32_t A0, uint32_t A1, uint32_t A2, uint32_t A3, uint32_t A4, uint32_t A5) {
    return B != A0 && B != A1 && B != A2 && B != A3 && B != A4 && B != A5;
}

uint32_t src_get(const uint32_t* src, int x, int y, int width, int height) {
	return src[std::max(0, std::min(x, width - 1)) +
                         std::max(0, std::min(y, height - 1)) * width];
}

uint32_t luma(uint32_t C) {
    const uint32_t alpha = (C & 0xFF000000) >> 24;
    return (((C & 0x00FF0000) >> 16) + ((C & 0x0000FF00) >> 8) + (C & 0x000000FF) + 1) * (256 - alpha);
}


void mmpx_scale(const unsigned scale, const uint32_t* src, 
uint32_t* dst, const unsigned width, const unsigned height) {
    // If it was important to run this faster (e.g., as a full screen
    // filter) then it would make sense to remove the alpha cases and
    // to precompute horizontal and vertical binary gradients
    // (inequalities), since most of the tests are redundantly looking
    // at whether adjacent elements are the same are different.

    // Select sets of rules for exploring and debugging. These
    // constants should all be set to true for the normal algorithm
    // and the branches compiled away.
    const bool run1to1SlopeCases             = true;
    const bool runIntersectionCases          = true;
    const bool run2to1SlopeCases             = true;
    
    // Debugging constants
	for (int srcY = 0; srcY < height; ++srcY) {
		int srcX = 0;

		// Read initial values at the beginning of a row, dealing with clamping
		uint32_t A = src_get(src, srcX - 1, srcY - 1, width, height);
		uint32_t B = src_get(src, srcX, srcY - 1, width, height);
		uint32_t C = src_get(src, srcX + 1, srcY - 1, width, height);
		uint32_t D = src_get(src, srcX - 1, srcY    , width, height);
		uint32_t E = src_get(src, srcX    , srcY    , width, height);
		uint32_t F = src_get(src, srcX + 1, srcY    , width, height);
		uint32_t G = src_get(src, srcX - 1, srcY + 1, width, height);
		uint32_t H = src_get(src, srcX    , srcY + 1, width, height);
		uint32_t I = src_get(src, srcX + 1, srcY + 1, width, height);
		uint32_t Q = src_get(src, srcX - 2, srcY, width, height);
		uint32_t R = src_get(src, srcX + 2, srcY, width, height);

		for (srcX = 0; srcX < width; ++srcX) {
			//    .
			//    P
			//   ABC
			// .QDEFR.
			//   GHI
			//    S
			//    .
			
			// Default output to nearest neighbor interpolation, with
			// all four outputs equal to the center input pixel.
			uint32_t J = E;
			uint32_t K = E;
			uint32_t L = E;
			uint32_t M = E;

			// Skip constant 3x3 centers and just use nearest-neighbor
			// them.  This gives a good speedup on spritesheets with
			// lots of padding and full screen images with large
			// constant regions such as skies.
			
			//if (! all_eq8(E, A,B,C, D,F, G,H,I)) {
			// Optimized by inlining:
			if (((A ^ E) | (B ^ E) | (C ^ E) | (D ^ E) | (F ^ E) | (G ^ E) | (H ^ E) | (I ^ E)) != 0) {

				// Read P and S, the vertical extremes that are not passed between iterations
				uint32_t P = src_get(src, srcX, srcY - 2, width, height);
				uint32_t S = src_get(src, srcX, srcY + 2, width, height);

				// Compute only the luminances required (faster than
				// passing all around between iterations).  Note that
				// these are premultiplied by alpha and biased so that
				// black has greater luminance than transparent.
				uint32_t Bl = luma(B);
				uint32_t Dl = luma(D);
				uint32_t El = luma(E);
				uint32_t Fl = luma(F);
				uint32_t Hl = luma(H);
			
				if (run1to1SlopeCases) {
					// Round some corners and fill in 1:1 slopes, but preserve
					// sharp right angles.
					//
					// In each expression, the left clause is from
					// EPX and the others are new. EPX
					// recognizes 1:1 single-pixel lines because it
					// applies the rounding only to the LINE, and not
					// to the background (it looks at the mirrored
					// side).  It thus fails on thick 1:1 edges
					// because it rounds *both* sides and produces an
					// aliased edge shifted by 1 dst pixel.  (This
					// also yields the mushroom-shaped arrow heads,
					// where that 1-pixel offset runs up against the
					// 2-pixel aligned end; this is an inherent
					// problem with 2X in-palette scaling.)
					//
					// The 2nd clause clauses avoid *double* diagonal
					// filling on 1:1 slopes to prevent them becoming
					// aliased again. It does this by breaking
					// symmetry ties using luminance when working with
					// thick features (it allows thin and transparent
					// features to pass always).
					//
					// The 3rd clause seeks to preserve square corners
					// by considering the center value before
					// rounding.
					//
					// The 4th clause identifies 1-pixel bumps on
					// straight lines that are darker than their
					// background, such as the tail on a pixel art
					// "4", and prevents them from being rounded. This
					// corrects for asymmetry in this case that the
					// luminance tie breaker introduced.

					// .------------ 1st ------------.      .----- 2nd ---------.      .------ 3rd -----.      .--------------- 4th -----------------------.
					// ◤
					if ((D == B && D != H && D != F)  &&  (El >= Dl || E == A)  &&  any_eq3(E, A,C,G)  &&  ((El < Dl) || A != D || E != P || E != Q)) {
						J = D;
					}

					// ◥
					if ((B == F && B != D && B != H)  &&  (El >= Bl || E == C)  &&  any_eq3(E, A,C,I)  &&  ((El < Bl) || C != B || E != P || E != R)) {
						K = B;
					}

					// ◣
					if ((H == D && H != F && H != B)  &&  (El >= Hl || E == G)  &&  any_eq3(E, A,G,I)  &&  ((El < Hl) || G != H || E != S || E != Q)) {
						L = H;
					}

					// ◢
					if ((F == H && F != B && F != D)  &&  (El >= Fl || E == I)  &&  any_eq3(E, C,G,I)  &&  ((El < Fl) || I != H || E != R || E != S)) {
						M = F;
					}
				}

				if (runIntersectionCases) {
					// Clean up disconnected line intersections.
					//
					// The first clause recognizes being on the inside
					// of a diagonal corner and ensures that the "foreground"
					// has been correctly identified to avoid
					// ambiguous cases such as this:
					//
					//  o#o#
					//  oo##
					//  o#o#
					//
					// where trying to fix the center intersection of
					// either the "o" or the "#" will leave the other
					// one disconnected. This occurs, for example,
					// when a pixel-art letter "B" or "R" is next to
					// another letter on the right.
					//
					// The second clause ensures that the pattern is
					// not a notch at the edge of a checkerboard
					// dithering pattern.
					// 
					
					// >
					//  .--------------------- 1st ------------------------.      .--------- 2nd -----------. 
					if ((E != F && all_eq4(E, C,I,D,Q) && all_eq2(F, B, H))  &&  (F != src_get(src, srcX + 3, srcY, width, height))) {
						M = F;
						K = M;
					}

					// <
					if ((E != D && all_eq4(E, A,G,F,R) && all_eq2(D, B, H))  &&  (D != src_get(src, srcX - 3, srcY, width, height))) {
						L = D;
						J = D;
					}

					// v
					if ((E != H && all_eq4(E, G,I,B,P) && all_eq2(H, D, F))  &&  (H != src_get(src, srcX, srcY + 3, width, height))) {
						L = H;
						M = H;
					} 

					// ∧
					if ((E != B && all_eq4(E, A,C,H,S) && all_eq2(B, D, F))  &&  (B != src_get(src, srcX, srcY - 3, width, height))) {
						J = B;
						K = B;
					}

					// Remove tips of bright triangles on dark
					// backgrounds. The luminance tie breaker for 1:1
					// pixel lines leaves these as sticking up squared
					// off, which makes bright triangles and diamonds
					// look bad.
					//
					// Extracting common subexpressions slows this down
					
					// ▲
					if (Bl < El && all_eq4(E, G,H,I,S) && none_eq4(E, A,D,C,F)){
						K = B;
						J = B;
					} 

					// ▼
					if (Hl < El && all_eq4(E, A,B,C,P) && none_eq4(E, D,G,I,F)) {
						L = H;
						M = H;
					} 

					// ▶
					if (Fl < El && all_eq4(E, A,D,G,Q) && none_eq4(E, B,C,I,H)) {
						M = F;
						K = F;
					} 

					// ◀
					if (Dl < El && all_eq4(E, C,F,I,R) && none_eq4(E, B,A,G,H)) {
						L = D;
						J = L;
					} 
				}
				
			
				//////////////////////////////////////////////////////////////////////////////////
				// Do further neighborhood peeking to identify
				// 2:1 and 1:2 slopes of constant color.
				//
				// No performance gain for outer pretest requiring some pixel in common.
				if (run2to1SlopeCases) {
					
					// The first clause of each rule identifies a 2:1 slope line
					// of consistent color.
					//
					// The second clause verifies that the line is separated from
					// every adjacent pixel on one side and not part of a more
					// complex pattern. Common subexpressions from the second clause
					// are lifted to an outer test on pairs of rules.
					// 
					// The actions taken by rules are unusual in that they extend
					// a color assigned by previous rules rather than drawing from
					// the original source image.
					//
					//
					// The comments show a diagram of the local
					// neighborhood in which letters shown with the
					// same shape and color must match each other and
					// everything else without annotation must be
					// different from the solid colored, square
					// letters.

					if (H != B) { // Common subexpression
						// Above a 2:1 slope or -2:1 slope   ◢ ◣
						// First:
						if (H != A && H != E && H != C) {
							
							// Second:
							//     P 
							//   Ⓐ B C .
							// Q D 🄴 🅵 🆁
							//   🅶 🅷 I
							//     S
							if (all_eq3(H, G,F,R)  &&  none_eq2(H, D, src_get(src, srcX+2, srcY-1, width, height))) {
								L = M;
							}
							
							// Third:
							//     P 
							// . A B Ⓒ
							// 🆀 🅳 🄴 F R
							//   G 🅷 🅸
							//     S
							if (all_eq3(H, I,D,Q)  &&  none_eq2(H, F, src_get(src, srcX-2, srcY-1, width, height))) {
								M = L;
							} 
						}
						
						// Below a 2:1 (◤) or -2:1 (◥) slope (reflect the above 2:1 patterns vertically)  
						if (B != I && B != G && B != E) {
							
							//     P 
							//   🅰 🅱 C
							// Q D 🄴 🅵 🆁
							//   Ⓖ H I .
							//     S
							if (all_eq3(B, A,F,R)  &&  none_eq2(B, D, src_get(src, srcX+2, srcY+1, width, height))) {
								J = K;
							}
							
							
							//     P 
							//   A 🅱 🅲
							// 🆀 🅳 🄴 F R
							// . G H Ⓘ 
							//     S
							if (all_eq3(B, C,D,Q)  &&  none_eq2(B, F, src_get(src, srcX-2, srcY+1, width, height))) {
								K = J;
							}
						}
					} // H != B

					if (F != D) { // Common subexpression
						
						// Right of a -1:2 (\) or -1:2 (/) slope (reflect the left 1:2 patterns horizontally)
						if (D != I && D != E && D != C) {
							
							//     P
							//   🅰 B Ⓒ
							// Q 🅳 🄴 F R
							//   G 🅷 I
							//     🆂 .
							if (all_eq3(D, A,H,S)  &&  none_eq2(D, B, src_get(src, srcX+1, srcY+2, width, height))) {
								J = L;
							}
							
							//     🅿 .
							//   A 🅱 C
							// Q 🅳 🄴 F R
							//   🅶 H Ⓘ
							//     S
							if (all_eq3(D, G,B,P)  &&  none_eq2(D, H, src_get(src, srcX+1, srcY-2, width, height))) {
								L = J;
							}
						}
						
						// Left of a 1:2 (/) slope or -1:2 (\) slope (transpose the above 2:1 patterns)
						// Pull common none_eq subexpressions out
						if (F != E && F != A && F != G) {
							
							//     P     
							//   Ⓐ B 🅲   
							// Q D 🄴 🅵 R 
							//   G 🅷 I   
							//   . 🆂     
							if (all_eq3(F, C,H,S)  &&  none_eq2(F, B, src_get(src, srcX-1, srcY+2, width, height))) {
								K = M;
							}
							
							//   . 🅿
							//   A 🅱 C
							// Q D 🄴 🅵 R
							//   Ⓖ H 🅸
							//     S
							if (all_eq3(F, I,B,P)  &&  none_eq2(F, H, src_get(src, srcX-1, srcY-2, width, height))) {
								M = K;
							}
						}
					} // F != D
				} // 2:1 slope
			} // not constant

			// srcX * 2 + srcY * 4 * srcWidth
			int dstIndex = ((srcX + srcX) + (srcY << 2) * width);
			dst[dstIndex] = J;
			++dstIndex;
			dst[dstIndex] = K; 
			dstIndex += width + width - 1;
			dst[dstIndex] = L;
			++dstIndex;
			dst[dstIndex] = M;

			// Shift over already-read/computed values and bring in
			// the next srcX column. Note that we're reading from srcX
			// + 2 instead of srcX + 1 because this is reading ahead
			// one iteration.
			A = B; 
			B = C; 
			C = src_get(src, srcX + 2, srcY - 1, width, height);
			Q = D; 
			D = E; 
			E = F; 
			F = R; 
			R = src_get(src, srcX + 3, srcY, width, height);
			G = H; 
			H = I; 
			I = src_get(src, srcX + 2, srcY + 1, width, height);
		} // X
	} // Y
}