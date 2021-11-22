import cv2
import numpy as np
import itertools
import math

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

def make_graph(im, remove_full_connections = False):
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
    
    # remove edges that are internal to the connected component
    if remove_full_connections:
        for r in range(0, im.shape[0]):
            for c in range(0, im.shape[1]):

                if ((graph[r, c] & MID_LEFT) != 0 and (graph[r, c] & MID_RIGHT) != 0 and (graph[r, c] & MID_TOP) != 0 and (graph[r, c] & BOTTOM_MID) != 0):
                    graph[r, c] = graph[r, c] & ~(MID_LEFT | MID_RIGHT | BOTTOM_MID | MID_TOP)
                   
        for r in range(0, im.shape[0]):
            for c in range(0, im.shape[1]):
                if (graph[r, c] & MID_LEFT) != 0 and (graph[r, c - 1] & MID_RIGHT) == 0:
                    graph[r, c] = graph[r, c] & ~(MID_LEFT)
                if (graph[r, c] & MID_RIGHT) != 0 and (graph[r, c + 1] & MID_LEFT) == 0:
                    graph[r, c] = graph[r, c] & ~(MID_RIGHT)
                if (graph[r, c] & MID_TOP) != 0 and (graph[r - 1, c] & BOTTOM_MID) == 0:
                    graph[r, c] = graph[r, c] & ~(MID_TOP)
                if (graph[r, c] & BOTTOM_MID) != 0 and (graph[r + 1, c] & MID_TOP) == 0:
                    graph[r, c] = graph[r, c] & ~(BOTTOM_MID)

    return graph

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

def do_nearest_naive(im, scale):
    output = np.zeros((im.shape[0] * scale, im.shape[1] * scale, 3), dtype=np.uint8)
    for r in range(im.shape[0]):
        for c in range(im.shape[1]):
            output[r * scale : r * scale + scale, c * scale : c * scale + scale, :] = im[r, c, :]
    return output


def read_image():
    im = cv2.imread("testcases/win31_control_panel_input.png")
    #im = cv2.imread("testcases/win31_mushroom_input.png")

    max_dim = max(im.shape[0], im.shape[1])
    im_out = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    if im.shape[0] == max_dim:
        im_out[:, 0 : im.shape[1], :] = im
    else:
        im_out[0 : im.shape[0], :, :] = im
    im = im_out 

    resized = cv2.resize(im, (im.shape[0] * 15, im.shape[1] * 15), 0, 0, cv2.INTER_NEAREST)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    cv2.imwrite("resized_nearest.png", resized)
    cv2.imwrite("resized_nearest_naive.png", do_nearest_naive(im, 15))



    graph = make_graph(im)
    write_graph_image(graph, im)
    construct_voronoi_templates(graph, im, 15)


def construct_voronoi_templates(graph, im, scale_factor):
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

    # use  scale x scale magnification
    # centered around the center pixel this is 
    # L ---  --- C --- --- R = 3 + 12 = 15 
    scale = 3
    orders = []
    for first in range(scale // 2 + 1):
        for second in range(scale // 2 + 1):
            if first + second > 0:
                orders.append((first, second))
                orders.append((-1 * first, second))
                orders.append((first, -1 * second))
                orders.append((-1 * first, -1 * second))
    orders = sorted(orders,key=lambda x:x[0] ** 2 + x[1] ** 2)
    

    # do connected components of template and determine equations of each template
    index = 0
    output_intermediate = np.zeros((im.shape[0] * scale + 1, im.shape[1] * scale + 1, 3), dtype=np.uint8)
    output = np.zeros((im.shape[0] * scale_factor + 1, im.shape[1] * scale_factor + 1, 3), dtype=np.uint8)
    for r in range(0, im.shape[0] - 1):
        for c in range(0, im.shape[1] - 1):
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


            magnified = np.zeros((scale * 2 + 1, scale * 2 + 1, 3), dtype=np.uint8)
            if len(evaluated) == 9:
                # fill with central color
                for chan in range(3):
                    magnified[:, :, chan] = im[r, c, chan]
            else:
                # we need to evaluate this component, it is disconnected
                known = np.zeros((scale * 2 + 1, scale * 2 + 1), dtype=np.uint8) # 1 -> known, 2 -> interpolated, 3 -> edge
                
                # fill in known pixels
                for x in range(three_by_three.shape[0]):
                    for y in range(three_by_three.shape[1]):

                        value = three_by_three[x, y]
                        known[x * scale, y * scale] = 1
                        magnified[x * scale , y * scale, :] = im[r - 1 + x, c - 1 + y, :]
                        
                        if x > 0 and y > 0 and (value & TOP_LEFT) != 0:
                            for q in range(1, 1 + scale//2):
                                known[x * scale - q, y * scale - q] = 1
                                magnified[x * scale - q, y * scale - q, :] = im[r - 1 + x, c - 1 + y, :]
                            
                            # fill in central pixel color if edge exists
                            if x == 2 and y == 2:
                                for q in range(2 + scale//2, scale + 1):
                                    known[x * scale - q, y * scale - q] = 1
                                    magnified[x * scale - q, y * scale - q, :] = im[r, c, :]
                        
                        if x > 0 and (value & MID_TOP) != 0:
                            for q in range(1, 1 + scale//2):
                                known[x * scale - q, y * scale] = 1
                                magnified[x * scale - q, y * scale, :] = im[r - 1 + x, c - 1 + y, :]
                            
                            if x == 2 and y == 1:
                                for q in range(2 + scale//2, scale + 1):
                                    known[x * scale - q, y * scale] = 1
                                    magnified[x * scale - q, y * scale, :] = im[r, c, :]

                        if x > 0 and y < 2 and (value & TOP_RIGHT) != 0:
                            for q in range(1, 1 + scale//2):
                                known[x * scale - q, y * scale + q] = 1
                                magnified[x * scale - q, y * scale + q, :] = im[r - 1 + x, c - 1 + y, :]
                            
                            if x == 2 and y == 0:
                                for q in range(2 + scale//2, scale + 1):
                                    known[x * scale - q, y * scale + q] = 1
                                    magnified[x * scale - q, y * scale + q, :] = im[r, c, :]

                        if y > 0 and (value & MID_LEFT) != 0:
                            for q in range(1, 1 + scale//2):
                                known[x * scale, y * scale - q] = 1
                                magnified[x * scale, y * scale - q, :] = im[r - 1 + x, c - 1 + y, :]
                            
                            if x == 1 and y == 2:
                                for q in range(2 + scale//2, scale + 1):
                                    known[x * scale, y * scale - q] = 1
                                    magnified[x * scale, y * scale - q, :] = im[r, c, :]

                        if y < 2 and (value & MID_RIGHT) != 0:
                            for q in range(1, 1 + scale//2):
                                known[x * scale, y * scale + q] = 1
                                magnified[x * scale, y * scale + q, :] = im[r - 1 + x, c - 1 + y, :]

                            if x == 1 and y == 0:
                                for q in range(2 + scale//2, scale + 1):
                                    known[x * scale, y * scale + q] = 1
                                    magnified[x * scale, y * scale + q, :] = im[r, c, :]

                        if x < 2 and y > 0 and (value & BOTTOM_LEFT) != 0:
                            for q in range(1, 1 + scale//2):
                                known[x * scale + q, y * scale - q] = 1
                                magnified[x * scale + q, y * scale - q, :] = im[r - 1 + x, c - 1 + y, :]
                            
                            if x == 0 and y == 2:
                                for q in range(2 + scale//2, scale + 1):
                                    known[x * scale + q, y * scale - q] = 1
                                    magnified[x * scale + q, y * scale - q, :] = im[r, c, :]

                        if x < 2 and (value & BOTTOM_MID) != 0:
                            for q in range(1, 1 + scale//2):
                                known[x * scale + q, y * scale] = 1
                                magnified[x * scale + q, y * scale, :] = im[r - 1 + x, c - 1 + y, :]
                            
                            if x == 0 and y == 1:
                                for q in range(2 + scale//2, scale + 1):
                                    known[x * scale + q, y * scale] = 1
                                    magnified[x * scale + q, y * scale, :] = im[r, c, :]

                        if x < 2 and y < 2 and (value & BOTTOM_RIGHT) != 0:
                            for q in range(1, 1 + scale//2):
                                known[x * scale + q, y * scale + q] = 1
                                magnified[x * scale + q, y * scale + q, :] = im[r - 1 + x, c - 1 + y, :]

                            if x == 0 and y == 0:
                                for q in range(2 + scale//2, scale + 1):
                                    known[x * scale + q, y * scale + q] = 1
                                    magnified[x * scale + q, y * scale + q, :] = im[r, c, :]

                
                # fill all unknown pixels (will be edges, mark in known as = 3)
                for x in range(magnified.shape[0]):
                    for y in range(magnified.shape[1]):
                        if known[x, y] == 0:
                            possible = []
                            magnitude = 100
                            for orr in orders:
                                mag = orr[0] ** 2 + orr[1] ** 2
                                if x + orr[0] >= 0 and x + orr[0] <= magnified.shape[0] - 1 and y + orr[1] >= 0 and y + orr[1] <= magnified.shape[1] - 1 and known[x + orr[0], y + orr[1]] == 1:
                                    color = ((magnified[x + orr[0], y + orr[1], 0]) << 16) + ((magnified[x + orr[0], y + orr[1], 1]) << 8) + ((magnified[x + orr[0], y + orr[1], 2]) << 0)  
                                    if mag < magnitude:
                                        possible = [color,]
                                        magnitude = mag
                                    if mag == magnitude and color not in possible:
                                        possible.append(color)
                            if len(possible) == 1:
                                known[x, y] = 2
                                magnified[x, y, 0] = possible[0] >> 16
                                magnified[x, y, 1] = possible[0] >> 8
                                magnified[x, y, 2] = possible[0] >> 0
                            elif len(possible) > 1:
                                known[x, y] = 3
                
                # check visually all edges are 1px approx
                #if(np.sum(known[known == 0]) > 0):
                #    print("unknowns found " + str(index))
                #if(np.sum(known[known == 3]) > 0):
                #    print("edges found " + str(index))
                
            # write to the output
            output_intermediate[r * scale : r * scale + magnified.shape[0], c * scale : c * scale + magnified.shape[0], :] = magnified[:, :, :]

    cv2.imwrite("output_intermediate.png", output_intermediate)
    gr = make_graph(output_intermediate, True)
    write_graph_image(gr, output_intermediate, "intermediate_graph.png")


    # determine connected components in graph
    component_index = 1
    components = np.zeros(gr.shape, dtype=np.uint32)

    components_to_shapes = []

    to_evaluate = [0]
    DELTA = 0.1

    current_component = None

    while len(to_evaluate) > 0:
        evall = to_evaluate.pop()
        gr_x = evall // 1000
        gr_y = evall % 1000

        if gr[gr_x, gr_y] == 0 or components[gr_x, gr_y] != 0:
            if len(to_evaluate) == 0:
                rows, cols = np.where(gr != 0)
                if len(rows) > 0:
                    to_evaluate.append(rows[0] * 1000 + cols[0])
            continue
    
        # receive propagated component number
        determined_component_number = 0
        if (gr[gr_x, gr_y] & TOP_LEFT) != 0 and components[gr_x - 1, gr_y - 1] != 0:
            determined_component_number = components[gr_x - 1, gr_y - 1]
        elif (gr[gr_x, gr_y] & MID_LEFT) != 0 and components[gr_x, gr_y - 1] != 0:
            determined_component_number = components[gr_x, gr_y - 1] 
        elif (gr[gr_x, gr_y] & MID_TOP) != 0 and components[gr_x - 1, gr_y] != 0:
            determined_component_number = components[gr_x - 1, gr_y]
        elif (gr[gr_x, gr_y] & TOP_RIGHT) != 0 and components[gr_x - 1, gr_y + 1] != 0:
            determined_component_number = components[gr_x - 1, gr_y + 1]
        elif (gr[gr_x, gr_y] & BOTTOM_LEFT) != 0 and components[gr_x + 1, gr_y - 1] != 0:
            determined_component_number = components[gr_x + 1, gr_y - 1] 
        elif (gr[gr_x, gr_y] & BOTTOM_RIGHT) != 0 and components[gr_x + 1, gr_y + 1] != 0:
            determined_component_number = components[gr_x + 1, gr_y + 1]
        elif (gr[gr_x, gr_y] & BOTTOM_MID) != 0 and components[gr_x + 1, gr_y] != 0:
            determined_component_number = components[gr_x + 1, gr_y]
        elif (gr[gr_x, gr_y] & MID_RIGHT) != 0 and components[gr_x, gr_y + 1] != 0:
            determined_component_number = components[gr_x, gr_y + 1]
        
        if determined_component_number != 0:
            components[gr_x, gr_y] = determined_component_number
        else:
            if current_component is not None:
                components_to_shapes.append(current_component)
            current_component = (output_intermediate[gr_x, gr_y], {})
            determined_component_number = component_index
            components[gr_x, gr_y] = determined_component_number
            component_index += 1
        
        # propagate component number to edges
        if (gr[gr_x, gr_y] & TOP_LEFT) != 0:
            delt = 0
            while delt <= 0.5:
                delt = round(delt, 1)
                x = round(gr_x + 0.5 - delt, 1)
                if x not in current_component[1]:
                    current_component[1][x] = set()
                current_component[1][x].add(gr_y + 0.5 - delt)
                delt += DELTA

            to_evaluate.append((gr_x - 1) * 1000 + (gr_y - 1))
        
        if (gr[gr_x, gr_y] & MID_LEFT) != 0:
            to_evaluate.append((gr_x) * 1000 + (gr_y - 1))

            delt = 0
            while delt <= 0.5:
                delt = round(delt, 1)
                y = round(gr_y + 0.5 - delt, 1)
                x = round(gr_x + 0.5, 1)
                if x not in current_component[1]:
                    current_component[1][x] = set()
                current_component[1][x].add(y)
                delt += DELTA

        if (gr[gr_x, gr_y] & MID_TOP) != 0:
            to_evaluate.append((gr_x - 1) * 1000 + (gr_y))

            delt = 0
            while delt <= 0.5:
                delt = round(delt, 1)
                x = round(gr_x + 0.5 - delt, 1)
                y = round(gr_y + 0.5, 1)
                if x not in current_component[1]:
                    current_component[1][x] = set()
                current_component[1][x].add(y)
                delt += DELTA

        if (gr[gr_x, gr_y] & TOP_RIGHT) != 0:
            to_evaluate.append((gr_x - 1) * 1000 + (gr_y + 1))

            delt = 0
            while delt <= 0.5:
                delt = round(delt, 1)
                x = round(gr_x + 0.5 - delt, 1)
                y = round(gr_y + 0.5 + delt, 1)
                if x not in current_component[1]:
                    current_component[1][x] = set()
                current_component[1][x].add(y)
                delt += DELTA

        if (gr[gr_x, gr_y] & BOTTOM_LEFT) != 0:
            to_evaluate.append((gr_x + 1) * 1000 + (gr_y - 1))

            delt = 0
            while delt <= 0.5:
                delt = round(delt, 1)
                x = round(gr_x + 0.5 + delt, 1)
                y = round(gr_y + 0.5 - delt, 1)
                if x not in current_component[1]:
                    current_component[1][x] = set()
                current_component[1][x].add(y)
                delt += DELTA

        if (gr[gr_x, gr_y] & BOTTOM_RIGHT) != 0:
            to_evaluate.append((gr_x + 1) * 1000 + (gr_y+ 1))

            delt = 0
            while delt <= 0.5:
                delt = round(delt, 1)
                x = round(gr_x + 0.5 + delt, 1)
                if x not in current_component[1]:
                    current_component[1][x] = set()
                current_component[1][x].add(gr_y + 0.5 + delt)
                delt += DELTA



        if (gr[gr_x, gr_y] & BOTTOM_MID) != 0:
            to_evaluate.append((gr_x + 1) * 1000 + (gr_y))

            delt = 0
            while delt <= 0.5:
                delt = round(delt, 1)
                x = round(gr_x + 0.5 + delt, 1)
                y = round(gr_y + 0.5, 1)
                if x not in current_component[1]:
                    current_component[1][x] = set()
                current_component[1][x].add(y)
                delt += DELTA

        if (gr[gr_x, gr_y] & MID_RIGHT) != 0:
            to_evaluate.append((gr_x) * 1000 + (gr_y + 1))

            delt = 0
            while delt <= 0.5:
                delt = round(delt, 1)
                y = round(gr_y + 0.5 + delt, 1)
                x = round(gr_x + 0.5, 1)
                if x not in current_component[1]:
                    current_component[1][x] = set()
                current_component[1][x].add(y)
                delt += DELTA
        
        gr[gr_x, gr_y] = 0
        if len(to_evaluate) == 0:
            rows, cols = np.where(gr != 0)
            if len(rows) > 0:
                    to_evaluate.append(rows[0] * 1000 + cols[0])
        gr[gr_x, gr_y] = 0

    
    im_out = np.zeros(output.shape, dtype=np.uint8)
    points_polygon = []
    for z in range(1, component_index - 1):
        im_out[:, :, :] = 0
        component = components_to_shapes[z - 1]
        color = component[0]
        verts = component[1]

        num_pts = 0
        for ky in verts.keys():
            num_pts += len(verts[ky])
        
        pts = np.zeros((num_pts, 2), dtype=np.int32)
        index = 0
        for ky in verts.keys():
            for y_val in verts[ky]:
                pts[index, 0] = ky / output_intermediate.shape[0] * output.shape[0]
                pts[index, 1] = y_val / output_intermediate.shape[1] * output.shape[1]
                index+= 1
        points_polygon.append((color, pts))

        cv2.polylines(im_out,[pts],True,(int(color[0]), int(color[1]), int(color[2])))

        cv2.imwrite("combos/component" + str(z) + ".png", im_out)
    
    intermediate = np.zeros((10 * output_intermediate.shape[0] + 1, 10 * output_intermediate.shape[0] + 1, 3), dtype=np.uint8)
    
    for ox in range(intermediate.shape[0]):
        for oy in range(intermediate.shape[1]):
            
            mag_x = round(ox / 10, 1)
            shape_found = False
            for shape in components_to_shapes:
                if mag_x not in shape[1]:
                    continue
                cols_of_row = shape[1][mag_x]

                less_than = 0
                
                for col in cols_of_row:
                    if col * 10 == oy:
                        shape_found = True
                        intermediate[ox, oy, :] = (255, 0, 0)
            if not shape_found and ox < intermediate.shape[0] - 1 and oy < intermediate.shape[1] - 1:
                intermediate[ox, oy, :] = output_intermediate[ox // 10, oy // 10]

    index += 1
    cv2.imwrite("outline.png", intermediate)
    
    
    for ox in range(output.shape[0]):
        for oy in range(output.shape[1]):
            
            shape_found = None
            dist = output.shape[0]

            color = None
            for poly in points_polygon:
                inside = cv2.pointPolygonTest(poly[1], (ox, oy), False) 
                dist2 = abs(cv2.pointPolygonTest(poly[1], (ox, oy), True) )
            
                if inside >= 0 and dist2 < dist:
                    
                    color = poly[0]
            print(ox, oy, color is not None)
            if color is not None:
                output[ox, oy, :] = color
            
            
            """
            for shape in components_to_shapes:
                if mag_x not in shape[1]:
                    continue

                cols_of_row = shape[1][mag_x]

                less_than = 0
                shape_dist = dist
                for col in cols_of_row:
                    dist2 = abs(mag_y - col)
                    if dist2 < shape_dist:
                        shape_dist = dist2
                    if col == mag_y:
                        shape_found = shape[0]
                        shape_dist = 0
                    if col < mag_y:
                        less_than += 1
                if less_than % 2 == 1 and shape_dist < dist:
                    shape_found = shape[0]
                    break
                if shape_found is not None:
                    break
            """
            
            # fill the color based on shape
            #if shape_found is not None:
            #    output[ox, oy, :] = shape_found
        
    
    cv2.imwrite("output.png", output)



if __name__ == "__main__":
    read_image()