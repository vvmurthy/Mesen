import cv2
import numpy as np
import itertools

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



def rgb_to_yuv(color0):
    y0 = 0.299*color0[0] + 0.587*color0[1] + 0.114*color0[2]
    u0 = 0.492 * (color0[2] - y0)
    v0 = 0.877 * (color0[0] - y0)
    return (y0, u0, v0)

def similar(color, color1):
    yuv0 = rgb_to_yuv(color)
    yuv1 = rgb_to_yuv(color1)

    return abs(yuv0[0] - yuv1[0]) <= 48 and abs(yuv0[1] - yuv1[1]) <= 7 and abs(yuv0[2] - yuv1[2]) <= 6

def write_graph_image(graph, image, filename="file.png"):
    scale = 9
    output = np.zeros((image.shape[0] * scale, image.shape[1] * scale, 3), dtype=np.uint8)
    for r in range(0, image.shape[0]):
        for c in range(0, image.shape[1]):
            # write the color in the center
            start = int(scale / 3)
            x = 9 * r
            y = 9 * c
            for chan in range(2):
                output[x + start : x + 2 * start, 
                    y + start : y + 2 * start, chan] = image[r, c, chan]


            # write the left hand edge
            if (graph[r, c] & MID_LEFT) != 0:
                for chan in range(3):
                    output[x + scale//2, y : y + start, chan] = 200
            
            # write the top edge
            if (graph[r, c] & MID_TOP) != 0:
                for chan in range(3):
                    output[x : x + start, y + scale // 2, chan] = 200

            # write the right hand edge
            if (graph[r, c] & MID_RIGHT) != 0:
                for chan in range(3):
                    output[x + scale//2, y + 2 * start : y + 3 * start, chan] = 200

            # write the bottom edge
            if (graph[r, c] & BOTTOM_MID) != 0:
                for chan in range(3):
                    output[x + 2 * start : x + 3 * start, y + scale // 2, chan] = 200
            
            # top left
            if (graph[r, c] & TOP_LEFT) != 0:
                for chan in range(3):
                    for i in range(3):
                        output[x + i : x + i + 1, y + i : y + i + 1, chan] = 200

            # top right
            if (graph[r, c] & TOP_RIGHT) != 0:
                for chan in range(3):
                    for i in range(3):
                        output[x + i : x + i + 1, y + scale - 1 - i : y + scale - i, chan] = 200
            
            # bottom left
            if (graph[r, c] & BOTTOM_LEFT) != 0:
                for chan in range(3):
                    for i in range(3):
                        output[x + scale - 1 - i : x + scale - i, y + i : y + i + 1, chan] = 200

            # bottom right
            if (graph[r, c] & BOTTOM_RIGHT) != 0:
                for chan in range(3):
                    for i in range(3):
                        output[x + scale - 1 - i : x + scale - i, y + scale - 1 - i : y + scale - i, chan] = 200
            

    cv2.imwrite(filename, output)



def read_image():
    im = cv2.imread("testcases/smw_yoshi_input.png")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    graph = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)

    for r in range(0, im.shape[0]):
        for c in range(0, im.shape[1]):
            if r > 0 and similar(im[r-1, c, :], im[r, c, :]):
                graph[r, c] = (graph[r, c] | MID_TOP)
            if r > 0 and c > 0 and similar(im[r-1, c-1, :], im[r, c, :]):
                graph[r, c] = (graph[r, c] | TOP_LEFT)
            if r > 0 and c < im.shape[1] - 1 and similar(im[r-1, c+1, :], im[r, c, :]):
                graph[r, c] = (graph[r, c] | TOP_RIGHT)
            if c > 0 and similar(im[r, c-1, :], im[r, c, :]):
                graph[r, c] = (graph[r, c] | MID_LEFT)
            if c < im.shape[1] - 1 and similar(im[r, c+1, :], im[r, c, :]):
                graph[r, c] = (graph[r, c] | MID_RIGHT)
            if r < im.shape[0] - 1 and c > 0 and similar(im[r+1, c-1, :], im[r, c, :]):
                graph[r, c] = (graph[r, c] | BOTTOM_LEFT)
            if r < im.shape[0] - 1 and similar(im[r+1, c, :], im[r, c, :]):
                graph[r, c] = (graph[r, c] | BOTTOM_MID)
            if r < im.shape[0] - 1 and c < im.shape[1] - 1 and similar(im[r+1, c+1, :], im[r, c, :]):
                graph[r, c] = (graph[r, c] | BOTTOM_RIGHT)
    

    # remove fully connected 2 x 2
    for r in range(0, im.shape[0] - 1):
        for c in range(0, im.shape[1] - 1):
            if (graph[r, c] & BOTTOM_RIGHT) != 0 and (graph[r+1, c] & TOP_RIGHT) != 0 and  (graph[r, c+1] & BOTTOM_LEFT) != 0 and  (graph[r+1, c+1] & TOP_LEFT) != 0:
                graph[r, c] = graph[r, c] & ~(BOTTOM_RIGHT)
                graph[r+1, c] = graph[r + 1, c] & ~(TOP_RIGHT)
                graph[r+1, c+1] = graph[r + 1 , c + 1] & ~(TOP_LEFT)
                graph[r, c+1] = graph[r, c + 1] & ~(BOTTOM_LEFT)
    write_graph_image(graph, im)
    construct_voronoi_templates(graph, im)


def construct_voronoi_templates(graph, im):
    """
    X ----- X ---- X
    | .  .  |  . . |
    | .  .  |  . . |
    X ------X------X
    |   . . |  . . |
    |   . . |  . . |
    X-------X------X

    There are 20 edges. There are 2^20 = 1 048 576 combinations
    of edges in a 3x3 square

    FC_SQS = 4 * (2 ^ 14) = 65536
    
    (this is a lower bound than the typical planar bound of 3n - 6 = 21)
    """

    # do connected components of template and determine equations of each template
    for r in range(1, im.shape[0] - 1):
        for c in range(1, im.shape[1] - 1):
            three_by_three = np.zeros((3, 3), dtype=np.uint8)
            for i in range(3):
                k = -1 + i
                for j in range(3):
                    q = -1 + j
                    three_by_three[i, j] = graph[r + k, c + q]
            
            evaluated = [] # 0 -> 8, includes center pixel, 4 is center
            to_evaluate = [(1, 1)]
            left = 0
            right = 2
            top = 0
            bottom = 2
            while len(to_evaluate) > 0:
                evl = to_evaluate.pop()
                position_value = evl[0] * 3 + evl[1]
                if position_value in evaluated:
                    continue
                evaluated.append(position_value)
                value = three_by_three[evl[0], evl[1]]
                if evl[0] > top and evl[1] > left and (value & TOP_LEFT) != 0:
                    to_evaluate.append((evl[0] - 1, evl[1] - 1))
                if evl[0] > top and (value & MID_TOP) != 0:
                    to_evaluate.append((evl[0] - 1, evl[1]))
                if evl[0] > top and evl[1] < right and (value & TOP_RIGHT) != 0:
                    to_evaluate.append((evl[0] - 1, evl[1] + 1))
                if evl[1] > left and (value & MID_LEFT) != 0:
                    to_evaluate.append((evl[0], evl[1] - 1))
                if evl[1] < right and (value & MID_RIGHT) != 0:
                    to_evaluate.append((evl[0], evl[1] + 1))
                if evl[0] < bottom and evl[1] > left and (value & BOTTOM_LEFT) != 0:
                    to_evaluate.append((evl[0] + 1, evl[1] - 1))
                if evl[0]  < bottom and (value & BOTTOM_MID) != 0:
                    to_evaluate.append((evl[0] + 1, evl[1]))
                if evl[0]  < bottom and evl[1] < right and (value & BOTTOM_RIGHT) != 0:
                    to_evaluate.append((evl[0] + 1, evl[1] + 1))


            if len(evaluated) != 9:
                # we need to evaluate this component, it is disconnected
                print("evaluate component")

                # use  7 x 7 magnification
                # centered around the center pixel this is 
                # L ---  --- C --- --- R = 3 + 12 = 15 
                order = [
                    (1, 0),
                    (0, 1),
                    (0, -1),
                    (-1, 0),
                    (1, 1),
                    (-1, -1),
                    (1, -1),
                    (-1, 1),
                ]
                magnified = np.zeros((15, 15), dtype=np.uint8)
                known = np.zeros((15, 15), dtype=np.uint8) # 1 -> known, 2 -> interpolated, 3 -> edge
                
                # fill in known pixels

                # fill all unknown pixels (will be edges, mark in known as = 3)

if __name__ == "__main__":
    read_image()