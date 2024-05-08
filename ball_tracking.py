import numpy as np
import cv2

def track_ball(frame, boxes, class_ids, classes):
    ball_data = {}
    ball_index = classes.index('basketball')

    for i, box in enumerate(boxes):
        if class_ids[i] == ball_index:
            x, y, w, h = box
            cx = x + w // 2
            cy = y + h // 2
            cz = 0  

            ball_data = {
                'cx': cx,
                'cy': cy,
                'cz': cz
            }

            ball_data['cvx'], ball_data['cvy'], ball_data['cvz'] = 0, 0, 0 
            ball_data['cax'], ball_data['cay'], ball_data['caz'] = 0, 0, 0  
            ball_data['cv'] = 0 
            ball_data['ca'] = 0  

            break

    return ball_data

def print_ball_data(ball_data):
    if ball_data:
        print(f"Ball Coordinates: ({ball_data['cx']}, {ball_data['cy']}, {ball_data['cz']})")
        # print(f"Velocity: ({ball_data['cvx']}, {ball_data['cvy']}, {ball_data['cvz']}) ft/s")
        # print(f"Acceleration: ({ball_data['cax']}, {ball_data['cay']}, {ball_data['caz']}) ft/s^2")
    else:
        print("No basketball detected in the frame.")
