RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)

status = {
    'color': GREEN_COLOR,
    'text': 'PRESS SPACE TO CHECK LIVENESS'
}
checks_passed = {'face_detected':[],
                 'flow_check':[]}

DETECTION_TIME = 3
FLOW_THRESHOLD = 0.1
WIDTHS = [1, 0.8, 1]
HEIGHTS = [1, 1.9, 1]