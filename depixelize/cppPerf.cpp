/*
  cppPerf.cpp

  Copyright 2020 Morgan McGuire & Mara Gagiu.
  Available under the MIT license.

  Minimal performance testing framework for CPU pixel scaling algorithms

  The hqx algorithm design is Copyright 2003 Maxim Stepin ( maxst@hiend3d.com ),
  Copyright 2010 Cameron Zemek ( grom@zeminvaders.net ), and
  Copyright 2010 Dominic Szablewski ( mail@phoboslab.org ).
  It is available under the LGPL:https://github.com/Treeki/libxbr-standalone/

  xBR algorithm design by Hyllian. The implementation here is also from https://github.com/Treeki/libxbr-standalone/ and is
  Copyright (c) 2015 Treeki <treeki@gmail.com>. It is available under the GNU Lesser General Public License.
*/

#include <G3D/G3D.h>
#include "../libxbr/filters.h"

// Define to examine the output
//#define SAVE_OUTPUT

typedef uint32 ABGR8;

class Algorithm {
protected:

    mutable const ABGR8* m_srcBuffer;
    mutable int m_srcWidth;
    mutable int m_srcHeight;
    mutable int m_srcMaxX;
    mutable int m_srcMaxY;

    inline ABGR8 src(int x, int y) const {
        // Clamp to border
        if (uint32(x) > uint32(m_srcMaxX) || uint32(y) > uint32(m_srcMaxY)) {
            x = G3D::clamp(x, 0, m_srcMaxX);
            y = G3D::clamp(y, 0, m_srcMaxY);
        }
        
        return m_srcBuffer[y * m_srcWidth + x];
    }

    // Call to make src() available
    void init(const Color4unorm8* srcBuffer, int srcWidth, int srcHeight) const {
        m_srcBuffer = (const ABGR8*)srcBuffer;
        m_srcWidth = srcWidth;
        m_srcHeight = srcHeight;
        m_srcMaxX = srcWidth - 1;
        m_srcMaxY = srcHeight - 1;
    }

public:
  
    RealTime time = 0;

    virtual ~Algorithm() {}
    virtual String name() const = 0;
    virtual void run(const Color4unorm8* srcBuffer, Color4unorm8* dstBuffer, int srcWidth, int srcHeight) const = 0;
}; // Algorithm


class NearestAlgorithm : public Algorithm {
public:
    virtual String name() const override { return "Nearest"; }
    virtual void run(const Color4unorm8* src, Color4unorm8* dst, int srcWidth, int srcHeight) const override {
        const int dstWidth = srcWidth * 2;
        
        for (int srcY = 0; srcY < srcHeight; ++srcY) {
            const Color4unorm8* srcRow = src +srcY * srcWidth;
            Color4unorm8* dstRow = dst + srcY * 2 * dstWidth;

            for (int srcX = 0; srcX < srcWidth; ++srcX, ++srcRow, dstRow += 2) {
                dstRow[0] = dstRow[1] = dstRow[dstWidth] = dstRow[dstWidth + 1] = srcRow[0];
            }
        }
    }
}; // Nearest

// Implementation of Eric Johnston and Andrea Mazzoleni's
// EPX algorithm based on https://www.scale2x.it/algorithm

class EPXAlgorithm : public Algorithm {
public:
    virtual String name() const override { return "EPX"; }
    virtual void run
    (const Color4unorm8* srcBuffer,
     Color4unorm8*       dst,
     int                 srcWidth,
     int                 srcHeight) const override {

        init(srcBuffer, srcWidth, srcHeight);

        for (int srcY = 0; srcY < srcHeight; ++srcY) {
            int srcX = 0;

	    ABGR8 D = src(srcX - 1, srcY), E = src(srcX, srcY);

            for (int srcX = 0; srcX < srcWidth; ++srcX) {
                const ABGR8 B = src(srcX, srcY - 1);
                const ABGR8 F = src(srcX + 1, srcY);
                const ABGR8 H = src(srcX, srcY + 1);
                
                // Default to nearest neighbor, with all four outputs
                // equal to the input.
                ABGR8 J = E, K = E, L = E, M = E;                

                // Round some corners
                if (D == B && D != H && D != F) J = D;
                if (B == F && B != D && B != H) K = B;
                if (H == D && H != F && H != B) L = H;
                if (F == H && F != B && F != D) M = F;

                int dstIndex = ((srcX + srcX) + (srcY << 2) * srcWidth);
                ABGR8* dstPacked = (ABGR8*)dst + dstIndex;
                
                *dstPacked = J; dstPacked++;
                *dstPacked = K; dstPacked += srcWidth + srcWidth - 1;
                *dstPacked = L; dstPacked++;
                *dstPacked = M;

                // Pass along already-read values
                D = E; E = F;
            } // X
        } // Y
    }
}; // EPX

struct SquareComparator {
  bool operator() (std::pair<int, int> i,std::pair<int, int> j) { return (i.first * i.first + i.second * i.second < j.first * j.first + j.second * j.second);}
} SquareComparatorObject;

class DepixelizeAlgorithm : public Algorithm {
public:
    static const int SCALE = 5;

    /**
    # bit structure
    # 0 -> top left
    TOP_LEFT = (0b00000001 << 0)
    TOP_LEFT_POSITION = 0x00000000000000FF

    # 1 -> mid top
    MID_TOP = (0b00000001 << 1)
    MID_TOP_POSITION = 0x000000000000FF00

    # 2 -> top right
    TOP_RIGHT = (0b00000001 << 2)
    TOP_RIGHT_POSITION = 0x0000000000FF0000

    # 3 -> mid left
    MID_LEFT = (0b00000001 << 3)
    MID_LEFT_POSITION = 0x00000000FF000000

    # 4 -> mid right
    MID_RIGHT = (0b00000001 << 4)
    MID_RIGHT_POSITION = 0x000000FF00000000

    # 5 -> bottom left
    BOTTOM_LEFT = (0b00000001 << 5)
    BOTTOM_LEFT_POSITION = 0x0000FF0000000000

    # 6 -> bottom mid
    BOTTOM_MID = (0b00000001 << 6)
    BOTTOM_MID_POSITION = 0x00FF000000000000

    # 7 -> bottom right
    BOTTOM_RIGHT = (0b00000001 << 7)
    BOTTOM_RIGHT_POSITION = 0xFF00000000000000
    */
    static const unsigned char TOP_LEFT = (1 << 0);
    static const unsigned char MID_TOP = (1 << 1);
    static const unsigned char TOP_RIGHT = (1 << 2);
    static const unsigned char MID_LEFT = (1 << 3);
    static const unsigned char MID_RIGHT = (1 << 4);
    static const unsigned char BOTTOM_LEFT = (1 << 5);
    static const unsigned char BOTTOM_MID = (1 << 6);
    static const unsigned char BOTTOM_RIGHT = (1 << 7);

    virtual String name() const override { return "Depixel"; }

    /**
    def rgb_to_yuv(color0):
        y0 = 0.299*color0[0] + 0.587*color0[1] + 0.114*color0[2]
        u0 = 0.492 * (color0[2] - y0)
        v0 = 0.877 * (color0[0] - y0)
        return (y0, u0, v0)

    def similar(color, color1):
        yuv0 = rgb_to_yuv(color)
        yuv1 = rgb_to_yuv(color1)

        return abs(yuv0[0] - yuv1[0]) <= 48 and abs(yuv0[1] - yuv1[1]) <= 7 and abs(yuv0[2] - yuv1[2]) <= 6

    */
    inline static bool similar(ABGR8 color0, ABGR8 color1)
    {
        unsigned char color0_b = ((color0 & 0x00FF0000) >> 16); 
        unsigned char color0_g = ((color0 & 0x00FF0000) >> 8);
        unsigned char color0_r = ((color0 & 0x00FF0000) >> 0);

        float y0 = 0.299 * color0_r + 0.587 * color0_g + 0.114 * color0_b;
        float u0 = 0.492 * ((float)color0_b - y0);
        float v0 = 0.877 * ((float)color0_r - y0);


        unsigned char color1_b = ((color1 & 0x00FF0000) >> 16); 
        unsigned char color1_g = ((color1 & 0x00FF0000) >> 8);
        unsigned char color1_r = ((color1 & 0x00FF0000) >> 0);

        float y1 = 0.299 * color1_r + 0.587 * color1_g + 0.114 * color1_b;
        float u1 = 0.492 * (color1_b - y1);
        float v1 = 0.877 * (color1_r - y1);

        return std::abs(y0 - y1) <= 48 && std::abs(u0 - u1) <= 7 && std::abs(v0 - v1) <= 6;

    }

    inline ABGR8 src(const int r, const int c) const {
        return m_srcBuffer[r * m_srcWidth + c];
    }

   uint8_t* make_graph(const Color4unorm8* srcBuffer, int srcWidth, int srcHeight)  const
    {
        uint8_t* graph = new uint8_t[srcWidth * srcHeight];
        
        for (int r = 0; r < srcHeight; ++r) {
            for(int c = 0 ; c < srcWidth ; c++) {
                if(r > 0 && similar(src(r-1, c), src(r, c))) {
                    graph[r * srcWidth + c] = (graph[r * srcWidth + c] | MID_TOP);
                }
                if(r > 0 && c > 0 && similar(src(r-1, c-1), src(r, c))) {
                    graph[r * srcWidth + c] = (graph[r * srcWidth + c] | TOP_LEFT);
                }
                if(r > 0 && c < srcWidth - 1 && similar(src(r-1, c+1), src(r, c))) {
                    graph[r * srcWidth + c] = (graph[r * srcWidth + c] | TOP_RIGHT);
                }
                if(c > 0 && similar(src(r, c-1), src(r, c))) {
                    graph[r * srcWidth + c] = (graph[r * srcWidth + c] | MID_LEFT);
                }
                
                if(c < srcWidth - 1 && similar(src(r, c+1), src(r, c))) {
                    graph[r * srcWidth + c] = (graph[r * srcWidth + c] | MID_RIGHT);
                }
                if(r < srcHeight - 1 && c > 0 && similar(src(r+1, c-1), src(r, c))) {
                    graph[r * srcWidth + c] = (graph[r * srcWidth + c] | BOTTOM_LEFT);
                }
                if(r < srcHeight - 1 && similar(src(r+1, c), src(r, c))) {
                    graph[r * srcWidth + c] = (graph[r * srcWidth + c] | BOTTOM_MID);
                }
                if(r < srcHeight - 1 && c < srcWidth - 1 && similar(src(r+1, c+1), src(r, c))) {
                    graph[r * srcWidth + c] = (graph[r * srcWidth + c] | BOTTOM_RIGHT);
                }
            }
        }

        for (int r = 0; r < srcHeight - 1; ++r) {
            for(int c = 0 ; c < srcWidth -1 ; c++) {
                if((graph[r * srcWidth + c] & BOTTOM_RIGHT) != 0 && 
                (graph[(r+1) * srcWidth + c] & TOP_RIGHT) != 0 && 
                (graph[r * srcWidth + c + 1] & BOTTOM_LEFT) != 0 &&
                (graph[(r+1) * srcWidth + c + 1] & TOP_LEFT) != 0)
                {
                    graph[r * srcWidth + c] = graph[r * srcWidth + c] & ~BOTTOM_RIGHT;
                    graph[(r+1) * srcWidth + c] = graph[(r+1) * srcWidth + c] & ~TOP_RIGHT;
                    graph[(r+1) * srcWidth + c + 1] = graph[(r+1) * srcWidth + c + 1] & ~TOP_LEFT;
                    graph[r * srcWidth + c + 1] = graph[r * srcWidth + c + 1]  & ~BOTTOM_LEFT;
                }
            }
        }


        return graph;
    }


    virtual void run(const Color4unorm8* srcBuffer, Color4unorm8* dst, int srcWidth, int srcHeight) const override {
        init(srcBuffer, srcWidth, srcHeight);
        uint8_t* graph = make_graph(srcBuffer, srcWidth, srcHeight);

        std::vector<std::pair<int, int>> orders;
        for(int first = 0 ; first < SCALE / 2 + 1 ; first++) {
            for(int second = 0 ; second < SCALE / 2 + 1 ; second++)
            {
                if(first + second > 0) {
                    std::pair<int, int> pair1 = std::make_pair(first, second);
                    std::pair<int, int> pair2 = std::make_pair(-1 * first, second);
                    std::pair<int, int> pair3 = std::make_pair(first, -1 * second);
                    std::pair<int, int> pair4 = std::make_pair(-1 * first, -1 * second);
                    orders.push_back(pair1);
                    orders.push_back(pair2);
                    orders.push_back(pair3);
                    orders.push_back(pair4);
                }
            }
        }
        std::sort (orders.begin(), orders.end(), SquareComparatorObject);

        for(int r = 1 ; r < srcHeight - 1 ; r++) {
            for(int c = 1 ; c < srcWidth - 1; c++) {

                uint8_t* threeByThree = new uint8_t[9];
                for(int i = 0 ; i < 3 ; i++) {
                    int k = -1 + i;
                    for(int j = 0 ; j < 3 ; j++) {
                        int q = -1 + j;
                        threeByThree[i * 3 + j]  = graph[(r + k) * srcWidth + c + q];
                    }
                }

                uint8_t* evaluated = new uint8_t[9];
                for(int i = 0 ; i < 9 ; i++) {
                    evaluated[i] = 0;
                }
                std::vector<std::pair<int, int>> to_evaluate;
                to_evaluate.push_back(std::make_pair(1, 1));

                int left = 0;
                int right = 2;
                int top = 0;
                int bottom = 2;
                while(to_evaluate.size() > 0) {
                    std::pair<int, int> evl = to_evaluate.back();
                    to_evaluate.pop_back();
                    int position_value = evl.first * 3 + evl.second;
                    if(evaluated[position_value] == 1) {
                        continue;
                    }
                    evaluated[position_value] = 1;
                    uint8_t value = threeByThree[evl.first * 3 + evl.second];
                    if(evl.first > top && evl.second > left && (value & TOP_LEFT) != 0) {
                        to_evaluate.push_back(std::make_pair(evl.first - 1, evl.second - 1));
                    }
                    if(evl.first > top && (value & MID_TOP) != 0) {
                        to_evaluate.push_back(std::make_pair(evl.first - 1, evl.second));
                    }
                    if(evl.first > top && evl.second < right && (value & TOP_RIGHT) != 0) {
                        to_evaluate.push_back(std::make_pair(evl.first - 1 , evl.second + 1));
                    }
                    if(evl.second > left && (value & MID_LEFT) != 0) {
                        to_evaluate.push_back(std::make_pair(evl.first, evl.second - 1));
                    } 
                    if(evl.second < right && (value & MID_RIGHT) != 0) {
                        to_evaluate.push_back(std::make_pair(evl.first, evl.second + 1));
                    }
                    if(evl.first < bottom && evl.second > left && (value & BOTTOM_LEFT) != 0) {
                        to_evaluate.push_back(std::make_pair(evl.first + 1, evl.second - 1));
                    }
                    if(evl.first < bottom && (value & BOTTOM_MID) != 0) {
                        to_evaluate.push_back(std::make_pair(evl.first + 1, evl.second));
                    }
                    if(evl.first < bottom && evl.second < right && (value & BOTTOM_RIGHT) != 0) {
                        to_evaluate.push_back(std::make_pair(evl.first + 1, evl.second + 1));
                    }
                }
                bool allEvaluated = true;
                for(int i = 0 ; i < 9 ; i++) {
                    if(evaluated[i] == 0) {
                        allEvaluated = false;
                        break;
                    }
                }
                delete[] evaluated;

                int newHeight = SCALE * 2 + 1;
                ABGR8* magnified = new ABGR8[newHeight * newHeight];
                if(true) {
                    for(int i = 0 ; i < newHeight * newHeight ; i++) {
                        magnified[i] = m_srcBuffer[r * srcWidth + c];
                    }
                } else {
                   char* known = new char[newHeight * newHeight];
                    for(int x = 0 ; x < 3 ; x++) {
                        for(int y = 0 ; y < 3 ; y++) {
                            uint8_t value = threeByThree[x * 3 + y];
                            known[x * SCALE * newHeight + y * SCALE] = 1;
                            magnified[x * SCALE * newHeight + y * SCALE] = m_srcBuffer[srcWidth * (r - 1 + x) + c - 1 + y];

                            if(x > 0 && y > 0 && (value & TOP_LEFT) != 0) {
                                for (int q = 1; q < 1 + SCALE / 2;  q++) {
                                    known[(x * SCALE - q) * newHeight + y * SCALE - q] = 1;
                                    magnified[(x * SCALE - q) * newHeight + y * SCALE - q] = m_srcBuffer[srcWidth * (r - 1 + x) + c - 1 + y];
                                }
                                if(x == 2 && y == 2) {
                                    for(int q = 2 + SCALE / 2 ; q < SCALE + 1 ; q++) {
                                        known[(x * SCALE - q) * newHeight + y * SCALE - q] = 1;
                                        magnified[(x * SCALE - q) * newHeight + y * SCALE - q] = m_srcBuffer[srcWidth * r + c];
                                    }
                                }
                            }
                            if(x > 0 && (value & MID_TOP) != 0) {
                                for (int q = 1; q < 1 + SCALE / 2; q++) {
                                    known[(x * SCALE - q) * newHeight + y * SCALE] = 1;
                                    magnified[(x * SCALE - q) * newHeight + y * SCALE] = m_srcBuffer[srcWidth * (r - 1 + x) + c - 1 + y];
                                }
                                if(x == 2 && y == 1) {
                                    for(int q = 2 + SCALE / 2 ; q < SCALE + 1 ; q++) {
                                        known[(x * SCALE - q) * newHeight + y * SCALE - q] = 1;
                                        magnified[(x * SCALE - q) * newHeight + y * SCALE - q] = m_srcBuffer[srcWidth * r + c];
                                    }
                                }
                            }
                            if(x > 0 && y < 2 && (value & TOP_RIGHT) != 0) 
                            {
                                for (int q = 1; q < 1 + SCALE / 2; q++) {
                                    known[(x * SCALE - q) * newHeight + y * SCALE] = 1;
                                    magnified[(x * SCALE - q) * newHeight + y * SCALE] = m_srcBuffer[srcWidth * (r - 1 + x) + c - 1 + y];
                                }
                                if(x == 2 && y == 0) {
                                    for(int q = 2 + SCALE / 2 ; q < SCALE + 1 ; q++) {
                                        known[(x * SCALE - q) * newHeight + y * SCALE - q] = 1;
                                        magnified[(x * SCALE - q) * newHeight + y * SCALE - q] = m_srcBuffer[srcWidth * r + c];
                                    }
                                }
                            }
                            if(y > 0 && (value & MID_LEFT) != 0) 
                            {
                                for (int q = 1; q < 1 + SCALE / 2; q++) {
                                    known[(x * SCALE - q) * newHeight + y * SCALE] = 1;
                                    magnified[(x * SCALE - q) * newHeight + y * SCALE] = m_srcBuffer[srcWidth * (r - 1 + x) + c - 1 + y];
                                }
                                if(x == 1 && y == 2) {
                                    for(int q = 2 + SCALE / 2 ; q < SCALE + 1 ; q++) {
                                        known[(x * SCALE - q) * newHeight + y * SCALE - q] = 1;
                                        magnified[(x * SCALE - q) * newHeight + y * SCALE - q] = m_srcBuffer[srcWidth * r + c];
                                    }
                                }
                            }
                            if(y < 2 && (value & MID_RIGHT) != 0)
                            {
                                for (int q = 1; q < 1 + SCALE / 2; q++) {
                                    known[(x * SCALE - q) * newHeight + y * SCALE] = 1;
                                    magnified[(x * SCALE - q) * newHeight + y * SCALE] = m_srcBuffer[srcWidth * (r - 1 + x) + c - 1 + y];
                                }
                                if(x == 1 && y == 0) {
                                    for(int q = 2 + SCALE / 2 ; q < SCALE + 1 ; q++) {
                                        known[(x * SCALE - q) * newHeight + y * SCALE - q] = 1;
                                        magnified[(x * SCALE - q) * newHeight + y * SCALE - q] = m_srcBuffer[srcWidth * r + c];
                                    }
                                }
                            }
                            if(x < 2 && y > 0 && (value & BOTTOM_LEFT) != 0)
                            {
                                for (int q = 1; q < 1 + SCALE / 2; q++) {
                                    known[(x * SCALE - q) * newHeight + y * SCALE] = 1;
                                    magnified[(x * SCALE - q) * newHeight + y * SCALE] = m_srcBuffer[srcWidth * (r - 1 + x) + c - 1 + y];
                                }
                                if(x == 0 && y == 2) {
                                    for(int q = 2 + SCALE / 2 ; q < SCALE + 1 ; q++) {
                                        known[(x * SCALE - q) * newHeight + y * SCALE - q] = 1;
                                        magnified[(x * SCALE - q) * newHeight + y * SCALE - q] = m_srcBuffer[srcWidth * r + c];
                                    }
                                }
                            }
                            if(x < 2 && (value & BOTTOM_MID) != 0)
                            {
                                for (int q = 1; q < 1 + SCALE / 2; q++) {
                                    known[(x * SCALE - q) * newHeight + y * SCALE] = 1;
                                    magnified[(x * SCALE - q) * newHeight + y * SCALE] = m_srcBuffer[srcWidth * (r - 1 + x) + c - 1 + y];
                                }
                                if(x == 0 && y == 1) {
                                    for(int q = 2 + SCALE / 2 ; q < SCALE + 1 ; q++) {
                                        known[(x * SCALE - q) * newHeight + y * SCALE - q] = 1;
                                        magnified[(x * SCALE - q) * newHeight + y * SCALE - q] = m_srcBuffer[srcWidth * r + c];
                                    }
                                }
                            }
                            if(x < 2 && y < 2 && (value & BOTTOM_RIGHT) != 0)
                            {
                                for (int q = 1; q < 1 + SCALE / 2; q++) {
                                    known[(x * SCALE - q) * newHeight + y * SCALE] = 1;
                                    magnified[(x * SCALE - q) * newHeight + y * SCALE] = m_srcBuffer[srcWidth * (r - 1 + x) + c - 1 + y];
                                }
                                if(x == 0 && y == 0) {
                                    for(int q = 2 + SCALE / 2 ; q < SCALE + 1 ; q++) {
                                        known[(x * SCALE - q) * newHeight + y * SCALE - q] = 1;
                                        magnified[(x * SCALE - q) * newHeight + y * SCALE - q] = m_srcBuffer[srcWidth * r + c];
                                    }
                                }
                            }
                        }

                        for(int x = 0 ; x < 3 ; x++) {
                            for(int y = 0 ; y < 3 ; y++) {
                                if(known[x * 3 + y] == 0) {
                                    std::vector<ABGR8> possible;
                                    int magnitude = 100;
                                    for(std::pair<int, int> orr : orders) {
                                        int mag = orr.first * orr.first + orr.second * orr.second;
                                        if(x + orr.first >= 0 && x + orr.first <= newHeight - 1
                                            && y + orr.second >= 0 && y + orr.second <= newHeight - 1)
                                            {
                                                ABGR8 color = magnified[(x + orr.first) * newHeight + y + orr.second];
                                                if(mag < magnitude) {
                                                    possible.clear();
                                                    possible.push_back(color);
                                                    magnitude = mag;
                                                } else if(mag == magnitude && std::find(possible.begin(), possible.end(), color) != possible.end()) {
                                                    possible.push_back(color);
                                                }
                                            }
                                    }
                                    if(possible.size() == 1) {
                                        known[x * 3 + y] = 2;
                                        magnified[x * 3 + y] = possible[0];
                                    } else if(possible.size() > 1) {
                                        known[x * 3 + y] = 3;
                                    }
                                }
                            }
                        }
                    }
                    delete known;       
                }
                for(int x = 0 ; x < newHeight ; x++) {
                    for(int y = 0 ; y < newHeight ;y++) {
                       ((ABGR8*)dst)[(srcWidth * SCALE) * (r * SCALE + x) + (c * SCALE + y)] = magnified[x * newHeight + y];
                    }
                }
                    
                

                delete[] magnified;
                delete[] threeByThree;
            }
        }

        delete[] graph;
        while (orders.size() > 0)
           orders.erase(orders.begin());
            
    
    }
   

};


/* Algorithm by Morgan McGuire and Mara Gagiu. See the js-demo.html for a fully commented version. */
class MMPXAlgorithm : public Algorithm {
public:
    virtual String name() const override { return "MMPX"; }

    inline static uint32 luma(ABGR8 color) {
        const uint32 alpha = (color & 0xFF000000) >> 24;
        return (((color & 0x00FF0000) >> 16) + ((color & 0x0000FF00) >> 8) + (color & 0x000000FF) + 1) * (256 - alpha);
    }

    inline static bool all_eq2(ABGR8 B, ABGR8 A0, ABGR8 A1) {
        return ((B ^ A0) | (B ^ A1)) == 0;
    }

    inline static bool all_eq3(ABGR8 B, ABGR8 A0, ABGR8 A1, ABGR8 A2) {
        return ((B ^ A0) | (B ^ A1) | (B ^ A2)) == 0;
    }

    inline static bool all_eq4(ABGR8 B, ABGR8 A0, ABGR8 A1, ABGR8 A2, ABGR8 A3) {
        return ((B ^ A0) | (B ^ A1) | (B ^ A2) | (B ^ A3)) == 0;
    }

    inline static bool any_eq3(ABGR8 B, ABGR8 A0, ABGR8 A1, ABGR8 A2) {
        return B == A0 || B == A1 || B == A2;
    }

    inline static bool none_eq2(ABGR8 B, ABGR8 A0, ABGR8 A1) {
        return (B != A0) && (B != A1);
    }

    inline static bool none_eq4(ABGR8 B, ABGR8 A0, ABGR8 A1, ABGR8 A2, ABGR8 A3) {
        return B != A0 && B != A1 && B != A2 && B != A3;
    }

    virtual void run(const Color4unorm8* srcBuffer, Color4unorm8* dst, int srcWidth, int srcHeight) const override {
        init(srcBuffer, srcWidth, srcHeight);

        for (int srcY = 0; srcY < srcHeight; ++srcY) {
            int srcX = 0;

            // Inputs carried along rows
            ABGR8 A = src(srcX - 1, srcY - 1);
            ABGR8 B = src(srcX, srcY - 1);
            ABGR8 C = src(srcX + 1, srcY - 1);

            ABGR8 D = src(srcX - 1, srcY);
            ABGR8 E = src(srcX, srcY);
            ABGR8 F = src(srcX + 1, srcY);

            ABGR8 G = src(srcX - 1, srcY + 1);
            ABGR8 H = src(srcX, srcY + 1);
            ABGR8 I = src(srcX + 1, srcY + 1);

            ABGR8 Q = src(srcX - 2, srcY);
            ABGR8 R = src(srcX + 2, srcY);

            for (srcX = 0; srcX < srcWidth; ++srcX) {
                // Outputs
                ABGR8 J = E, K = E, L = E, M = E;

                if (((A ^ E) | (B ^ E) | (C ^ E) | (D ^ E) | (F ^ E) | (G ^ E) | (H ^ E) | (I ^ E)) != 0) {
                    ABGR8 P = src(srcX, srcY - 2), S = src(srcX, srcY + 2);
                    ABGR8 Bl = luma(B), Dl = luma(D), El = luma(E), Fl = luma(F), Hl = luma(H);

                    // 1:1 slope rules
                    {
                        if ((D == B && D != H && D != F) && (El >= Dl || E == A) && any_eq3(E, A, C, G) && ((El < Dl) || A != D || E != P || E != Q)) J = D;
                        if ((B == F && B != D && B != H) && (El >= Bl || E == C) && any_eq3(E, A, C, I) && ((El < Bl) || C != B || E != P || E != R)) K = B;
                        if ((H == D && H != F && H != B) && (El >= Hl || E == G) && any_eq3(E, A, G, I) && ((El < Hl) || G != H || E != S || E != Q)) L = H;
                        if ((F == H && F != B && F != D) && (El >= Fl || E == I) && any_eq3(E, C, G, I) && ((El < Fl) || I != H || E != R || E != S)) M = F;
                    }

                    // Intersection rules
                    {
                        if ((E != F && all_eq4(E, C, I, D, Q) && all_eq2(F, B, H)) && (F != src(srcX + 3, srcY))) K = M = F;
                        if ((E != D && all_eq4(E, A, G, F, R) && all_eq2(D, B, H)) && (D != src(srcX - 3, srcY))) J = L = D;
                        if ((E != H && all_eq4(E, G, I, B, P) && all_eq2(H, D, F)) && (H != src(srcX, srcY + 3))) L = M = H;
                        if ((E != B && all_eq4(E, A, C, H, S) && all_eq2(B, D, F)) && (B != src(srcX, srcY - 3))) J = K = B;
                        if (Bl < El && all_eq4(E, G, H, I, S) && none_eq4(E, A, D, C, F)) J = K = B;
                        if (Hl < El && all_eq4(E, A, B, C, P) && none_eq4(E, D, G, I, F)) L = M = H;
                        if (Fl < El && all_eq4(E, A, D, G, Q) && none_eq4(E, B, C, I, H)) K = M = F;
                        if (Dl < El && all_eq4(E, C, F, I, R) && none_eq4(E, B, A, G, H)) J = L = D;
                    }

                    // 2:1 slope rules
                    {
                        if (H != B) {
                            if (H != A && H != E && H != C) {
                                if (all_eq3(H, G, F, R) && none_eq2(H, D, src(srcX + 2, srcY - 1))) L = M;
                                if (all_eq3(H, I, D, Q) && none_eq2(H, F, src(srcX - 2, srcY - 1))) M = L;
                            }

                            if (B != I && B != G && B != E) {
                                if (all_eq3(B, A, F, R) && none_eq2(B, D, src(srcX + 2, srcY + 1))) J = K;
                                if (all_eq3(B, C, D, Q) && none_eq2(B, F, src(srcX - 2, srcY + 1))) K = J;
                            }
                        } // H !== B

                        if (F != D) {
                            if (D != I && D != E && D != C) {
                                if (all_eq3(D, A, H, S) && none_eq2(D, B, src(srcX + 1, srcY + 2))) J = L;
                                if (all_eq3(D, G, B, P) && none_eq2(D, H, src(srcX + 1, srcY - 2))) L = J;
                            }

                            if (F != E && F != A && F != G) {
                                if (all_eq3(F, C, H, S) && none_eq2(F, B, src(srcX - 1, srcY + 2))) K = M;
                                if (all_eq3(F, I, B, P) && none_eq2(F, H, src(srcX - 1, srcY - 2))) M = K;
                            }
                        } // F !== D
                    } // 2:1 slope
                }

                int dstIndex = ((srcX + srcX) + (srcY << 2) * srcWidth) >> 0;
                ABGR8* dstPacked = (ABGR8*)dst + dstIndex;

                *dstPacked = J; dstPacked++;
                *dstPacked = K; dstPacked += srcWidth + m_srcMaxX;
                *dstPacked = L; dstPacked++;
                *dstPacked = M;
                 
                A = B; B = C; C = src(srcX + 2, srcY - 1);
                Q = D; D = E; E = F; F = R; R = src(srcX + 3, srcY);
                G = H; H = I; I = src(srcX + 2, srcY + 1);
            } // X
        } // Y
    }
}; // MMPX


/* Correct bilinear interpolation assuming unassociated alpha; each operation must
   convert to premultiplied alpha and then back. */
class BilinearAlgorithm : public Algorithm {
public:
    virtual String name() const override { return "Bilinear"; }

    //inline Color4 static average2(Color4 C, Color4 D ) {
    //    const float a = (C.a + D.a) * 0.5f;

    //    if (a == 0) {
    //        return Color4unorm8::zero();
    //    } // early-out

    //    const float k = 0.5f / a;
    //    return Color4((C.rgb() * C.a + D.rgb() * D.a) * k, a);
    //}

    inline ABGR8 static average2(ABGR8 C, ABGR8 D) {
        // Early out on common case
        if ((C | D) == 0) { return 0; }

        uint32 Ca = ((C & 0xFF000000) >> 24) | 0, Cb = (C & 0x00FF0000) >> 16, Cg = (C & 0x0000FF00) >> 8, Cr = (C & 0x000000FF);
        uint32 Da = ((D & 0xFF000000) >> 24) | 0, Db = (D & 0x00FF0000) >> 16, Dg = (D & 0x0000FF00) >> 8, Dr = (D & 0x000000FF);

        uint32 Ea = Ca + Da;
        uint32 Er, Eg, Eb;

        if ((Ca < 255) || (Da < 255)) {
            // Premultiplied alpha. The missing 2 from the average cancels the
            // missing 2 in Ea from 9-bit to 8-bit conversion.
            const uint32 denom = Ea == 0 ? 1 : Ea;
            Er = min((Cr * Ca + Dr * Da) / denom, 255u);
            Eg = min((Cg * Ca + Dg * Da) / denom, 255u);
            Eb = min((Cb * Ca + Db * Da) / denom, 255u);
        }
        else {
            // No point in rounding, each is always at a 0.0 or 0.5
            Er = (Cr + Dr) >> 1;
            Eg = (Cg + Dg) >> 1;
            Eb = (Cb + Db) >> 1;
        }

        // Back to 8-bit. No point in rounding Ea, it is always at a 0.0 or 0.5
        Ea >>= 1;

        return (Ea << 24) | (Eb << 16) | (Eg << 8) | Er;
    }

  
    virtual void run(const Color4unorm8* srcBuffer, Color4unorm8* dst, int srcWidth, int srcHeight) const override {
        init(srcBuffer, srcWidth, srcHeight);

        for (int srcY = 0; srcY < srcHeight; ++srcY) {
            int srcX = 0;
            const int nextY = min(srcY + 1, m_srcMaxY);
            ABGR8 E = src(srcX, srcY);
            ABGR8 H = src(srcX, nextY);

            for (int srcX = 0; srcX < srcWidth; ++srcX) {
                // E F
                // H I
                // srcY will never be out of bounds here, so drop the unnecessary bounds checks.
                const int nextX = min(srcX + 1, m_srcMaxX);
                const ABGR8 F = m_srcBuffer[nextX + srcY * m_srcWidth];
                const ABGR8 I = m_srcBuffer[nextX + nextY * m_srcWidth];

                // Biased bilinear:
                const ABGR8 J = E;
                const ABGR8 K = ABGR8(average2(E, F));
                const ABGR8 L = ABGR8(average2(E, H));
                const ABGR8 M = ABGR8(average2(K, average2(H, I)));

                // Write output
                int dstIndex = ((srcX + srcX) + (srcY << 2) * srcWidth) >> 0;
                ABGR8* dstPacked = (ABGR8*)dst + dstIndex;

                *dstPacked = J; dstPacked++;
                *dstPacked = K; dstPacked += srcWidth + srcWidth - 1;
                *dstPacked = L; dstPacked++;
                *dstPacked = M;

                // Shift values over
                E = F;
                H = I;

            } // X
        } // Y
    }
}; // Bilinear

// XBR Implementation: https://github.com/Treeki/libxbr-standalone 
class XBRAlgorithm : public Algorithm {
public:
    virtual String name() const override { return "XBR"; }
    virtual void run(const Color4unorm8* src, Color4unorm8* dst, int srcWidth, int srcHeight) const override {
        const int dstWidth = srcWidth * 2;
        xbr_params xbrParams;
        xbr_data *xbrData;

        xbrData =  (xbr_data*) malloc(sizeof(xbr_data));
        xbr_init_data(xbrData);

        xbrParams.data = xbrData;
        xbrParams.input = reinterpret_cast<const uint8_t*>(src);
        xbrParams.output = reinterpret_cast<uint8_t*>(dst);
        xbrParams.inWidth = srcWidth;
        xbrParams.inHeight = srcHeight;
        xbrParams.inPitch = srcWidth * 4;
        xbrParams.outPitch = dstWidth * 4;

        xbr_filter_xbr2x(&xbrParams);
        
        free(xbrData);
    }
}; // XBR


// HQX Implementation: https://github.com/Treeki/libxbr-standalone 
class HQXAlgorithm : public Algorithm {
public:
    virtual String name() const override { return "HQX"; }
    virtual void run(const Color4unorm8* src, Color4unorm8* dst, int srcWidth, int srcHeight) const override {
        const int dstWidth = srcWidth * 2;
        xbr_params xbrParams;
        xbr_data* xbrData;

        xbrData = (xbr_data*)malloc(sizeof(xbr_data));
        xbr_init_data(xbrData);

        xbrParams.data = xbrData;
        xbrParams.input = reinterpret_cast<const uint8_t*>(src);
        xbrParams.output = reinterpret_cast<uint8_t*>(dst);
        xbrParams.inWidth = srcWidth;
        xbrParams.inHeight = srcHeight;
        xbrParams.inPitch = srcWidth * 4;
        xbrParams.outPitch = dstWidth * 4;

        xbr_filter_hq2x(&xbrParams);

        free(xbrData);     
    }
}; // HQX



void runAllCPPPerfTests() {
    const String FILENAME = "test-512.png";
    const int NUM_ITERATIONS = 60;

    // Load the test image
    const shared_ptr<Image>& image = Image::fromFile(FilePath::concat("../test-data", FILENAME));

    const int srcWidth = image->width(), srcHeight = image->height();

    // Extract the raw sRGBA bytes; these will be called RGBA8 in the API
    image->convert(ImageFormat::RGBA8());
    shared_ptr<CPUPixelTransferBuffer> ptb = CPUPixelTransferBuffer::create(srcWidth, srcHeight, image->format());
    image->toPixelTransferBuffer(Rect2D::xywh(0, 0, float(srcWidth), float(srcHeight)), ptb);

    const Color4unorm8* src = static_cast<const Color4unorm8*>(ptb->buffer());
    
    const shared_ptr<CPUPixelTransferBuffer>& dstBuffer = CPUPixelTransferBuffer::create(srcWidth * 2, srcHeight * 2, ImageFormat::RGBA8());
    Color4unorm8* dst = static_cast<Color4unorm8*>(dstBuffer->buffer());

    const shared_ptr<CPUPixelTransferBuffer>& dstBufferDepixel = CPUPixelTransferBuffer::create(srcWidth * DepixelizeAlgorithm::SCALE + 1, srcHeight * DepixelizeAlgorithm::SCALE + 1, ImageFormat::RGBA8());
    Color4unorm8* dstDepixel = static_cast<Color4unorm8*>(dstBufferDepixel->buffer());

    Array<Algorithm*> algorithmArray(
        new NearestAlgorithm(),
        new EPXAlgorithm(),
        new MMPXAlgorithm(), 
        new DepixelizeAlgorithm(),
        new BilinearAlgorithm(),
        new XBRAlgorithm(),
        new HQXAlgorithm());

    // Warm up the data and code caches by running each algorithm once
    for (Algorithm* algorithm : algorithmArray) {
        if(algorithm->name() == "Depixel") {
            algorithm->run(src, dstDepixel, srcWidth, srcHeight);
            #ifdef SAVE_OUTPUT
                    const shared_ptr<Image>& dest_image = Image::fromPixelTransferBuffer(dstBufferDepixel);
                    dest_image->save(( "results/" + algorithm->name() + "-out.png").c_str());
            #endif
        } else {
            algorithm->run(src, dst, srcWidth, srcHeight);
            #ifdef SAVE_OUTPUT
                const shared_ptr<Image>& dest_image = Image::fromPixelTransferBuffer(dstBuffer);
                dest_image->save(( "results/" + algorithm->name() + "-out.png").c_str());
            #endif
        }
        


    }

    for (Algorithm* algorithm : algorithmArray) {
        const RealTime before = System::time();
        for (int i = 0; i < NUM_ITERATIONS; ++i) {
            if(algorithm->name() == "Depixel") {
                algorithm->run(src, dstDepixel, srcWidth, srcHeight);
            } else {
                algorithm->run(src, dst, srcWidth, srcHeight);
            }
            
        }

        algorithm->time = (System::time() - before) / double(NUM_ITERATIONS);
    }

    // Print results
#   ifdef G3D_DEBUG
        debugPrintf("WARNING: Running in debug mode! Profiling results are not valid.\n\n");
#   endif

    debugPrintf("Algorithm   | Image ms | ns/pix\n");
    debugPrintf("------------|---------:|-----------:\n");
    const int dstPix = srcWidth * srcHeight * 4;
    for (Algorithm* algorithm : algorithmArray) {
        const RealTime ms = algorithm->time * 1e3;
        const RealTime ns = algorithm->time * 1e9 / double(dstPix);
        debugPrintf("%-12s|    %4.3f |    %4.3f\n", algorithm->name().c_str(), ms, ns);
    }
    debugPrintf("[Average of %d single-threaded CPU tests on `%s`]\n", NUM_ITERATIONS, FILENAME.c_str());

    algorithmArray.invokeDeleteOnAllElements();
    dst = nullptr;
}
