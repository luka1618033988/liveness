import cv2
import numpy as np


def calculate_grid(frame, widths, heights):
    total_width = sum(widths)
    total_height = sum(heights)
    
    height, width = frame.shape
    width_units = [width * w / total_width for w in widths]
    height_units = [height * h / total_height for h in heights]
    
    grid = np.empty((len(heights), len(widths)), dtype=object)
    y_start = 0
    for i, h_unit in enumerate(height_units):
        x_start = 0
        for j, w_unit in enumerate(width_units):
            grid[i, j] = frame[int(y_start):int(y_start + h_unit), int(x_start):int(x_start + w_unit)]
            x_start += w_unit
        y_start += h_unit
    return grid

def read_frame(cap, widths, heights):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_grid = calculate_grid(gray, widths, heights)
    return ret, frame, gray_grid, gray


def draw_gridlines(frame, widths, heights):
        y_start = 0
        for h_unit in [frame.shape[0] * h / sum(heights) for h in heights]:
            cv2.line(frame, (0, int(y_start)), (frame.shape[1], int(y_start)), (0, 255, 0), 1)
            y_start += h_unit
        x_start = 0
        for w_unit in [frame.shape[1] * w / sum(widths) for w in widths]:
            cv2.line(frame, (int(x_start), 0), (int(x_start), frame.shape[0]), (0, 255, 0), 1)
            x_start += w_unit


def draw_middle_rectangle(frame, widths, heights):
        cum_widths = np.cumsum([0] + [frame.shape[1] * w / sum(widths) for w in widths])
        cum_heights = np.cumsum([0] + [frame.shape[0] * h / sum(heights) for h in heights])
        top_left = (int(cum_widths[1]), int(cum_heights[1]))
        bottom_right = (int(cum_widths[2]), int(cum_heights[2]))
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 1)


def detect_face(gray, face_cascade):
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(100, 100), maxSize=(200, 200)
    )
    return  len(faces) > 0


def calculate_optical_flow(prev_grid, curr_grid, threshold=0.1):
    flow_grid = np.empty_like(prev_grid)
    for i in range(prev_grid.shape[0]):
        for j in range(prev_grid.shape[1]):
            prev = prev_grid[i, j]
            curr = curr_grid[i, j]
            flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            angle = np.abs(angle)
            angle -= np.mean(angle)
            vectors = magnitude# * angle
            motion_mask = magnitude > threshold
            flow_grid[i, j] = np.mean(vectors[motion_mask]) > threshold, np.std(vectors[motion_mask]), np.mean(vectors[motion_mask])
    return flow_grid


def extract_grid_flows(flow, widths, heights, threshold=0):
    total_width = sum(widths)
    total_height = sum(heights)
    
    h, w = flow.shape[:2]
    width_units = [w * wu / total_width for wu in widths]
    height_units = [h * hu / total_height for hu in heights]
    
    grid_flows = np.empty((len(heights), len(widths)), dtype=object)
    y_start = 0
    for i, h_unit in enumerate(height_units):
        x_start = 0
        for j, w_unit in enumerate(width_units):
            curr_grid_flow = flow[int(y_start):int(y_start + h_unit), int(x_start):int(x_start + w_unit)]
            magnitude, angle = cv2.cartToPolar(curr_grid_flow[..., 0], curr_grid_flow[..., 1])
            
            motion_mask = magnitude > threshold
            if np.count_nonzero(motion_mask) > 0:
                mean_val = np.mean(magnitude[motion_mask])
                std_dev = np.std(magnitude[motion_mask])
            else:
                mean_val = 0
                std_dev = 0

            grid_flows[i, j] = mean_val, std_dev, np.mean(magnitude)

            x_start += w_unit
        y_start += h_unit
    
    return grid_flows

def display_flows(curr_frame_normal,heights,widths,flow_grid):
    y_start = 0
    for i, h_unit in enumerate([curr_frame_normal.shape[0] * h / sum(heights) for h in heights]):
        x_start = 0
        for j, w_unit in enumerate([curr_frame_normal.shape[1] * w / sum(widths) for w in widths]):
            mean_val, std_dev, mean_motion = flow_grid[i, j]
            
            # Displaying the values
            cv2.putText(curr_frame_normal, f"Mean: {mean_val:.2f}", (int(x_start + 5), int(y_start + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(curr_frame_normal, f"Std: {std_dev:.2f}", (int(x_start + 5), int(y_start + 40)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(curr_frame_normal, f"Mean Motion: {mean_motion:.2f}", (int(x_start + 5), int(y_start + 60)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            x_start += w_unit
        y_start += h_unit

def check_flow(flow_grid,elements_to_check, threshold=0.2):
    def compare_tuples(t1, t2):
        return ((t1[1] + threshold) >= t2[1]) and ((t1[2] + threshold) >= t2[2])
    comparison_results = []
    for el in elements_to_check:
        target_element = flow_grid[el[0],el[1]]
        for i in range(flow_grid.shape[0]):
            for j in range(flow_grid.shape[1]):
                if (i,j) not in elements_to_check:
                    element = flow_grid[i, j]
                    comparison_results.append(compare_tuples(target_element, element))
    return check_true_perc(comparison_results)

def check_true_perc(list_to_check, threshold=0.8):
    count_true = sum(1 for element in list_to_check if element)
    return count_true >= threshold * len(list_to_check)

def display_status(curr_frame_normal,text,color):
    cv2.putText(
            curr_frame_normal, text, (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )