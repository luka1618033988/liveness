import time
import cv2
import argparse
from utils import read_frame, draw_gridlines, draw_middle_rectangle, detect_face, check_flow, check_true_perc, display_status, display_flows, extract_grid_flows
from config import RED_COLOR, GREEN_COLOR, status, checks_passed, DETECTION_TIME, FLOW_THRESHOLD, WIDTHS, HEIGHTS


parser = argparse.ArgumentParser()
parser.add_argument("--dev", default=False)


def main():
    args = parser.parse_args()
    dev_mode = (args.dev == 'True')
    start_time = None

    cap = cv2.VideoCapture(0)
    ret, prev_frame_normal, prev_gray_grid, prev_gray  = read_frame(cap, WIDTHS, HEIGHTS)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    space_pressed = False
    while True:
        ret, curr_frame_normal, curr_gray_grid, curr_gray = read_frame(cap, WIDTHS, HEIGHTS)
        if not ret:
            print("Can't receive frame. Exiting ...")
            break
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            space_pressed = True
            start_time = time.time()
        if dev_mode:
            draw_gridlines(curr_frame_normal, WIDTHS, HEIGHTS)
        else:
            draw_middle_rectangle(curr_frame_normal, WIDTHS, HEIGHTS)
            
        if space_pressed and (time.time() - start_time) <= DETECTION_TIME:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_grid = extract_grid_flows(flow, WIDTHS, HEIGHTS)
            if dev_mode:
                display_flows(curr_frame_normal, HEIGHTS, WIDTHS,flow_grid)
            checks_passed['face_detected'].append(detect_face(curr_gray_grid[1][1],face_cascade))
            checks_passed['flow_check'].append(check_flow(flow_grid,[(1,1),(2,0),(2,1),(2,2)], FLOW_THRESHOLD))
            status['text'] = 'LET ME SEE'
            status['color'] = GREEN_COLOR
            prev_gray = curr_gray.copy()

        elif space_pressed:
            liveness_checks = list(checks_passed.values())
            live = all([check_true_perc(check,0.6) for check in liveness_checks])
            status['text'] = "PASSED. Click Space to check again" if live else "FAILED. Click Space to check again"
            status['color']  = GREEN_COLOR if live else RED_COLOR
            space_pressed = False
            [check.clear() for check in checks_passed.values()]

        display_status(curr_frame_normal, status['text'],status['color'])
        
        cv2.imshow('Webcam', curr_frame_normal)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()