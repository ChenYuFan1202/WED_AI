import cv2
import mediapipe as mp
import numpy as np
import pygame
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to check if a point is inside a circle
def is_point_inside_circle(point, circle_center, circle_radius):
    return np.linalg.norm(np.array(point) - np.array(circle_center)) <= circle_radius

# Initialize Pygame for sound
pygame.mixer.init()
hit_sound = pygame.mixer.Sound("Pop_up_sound.wav")  # Replace "your_hit_sound_file.wav" with the actual file path

# For static images:
IMAGE_FILES = []
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.5) as pose:
    
    # Create a list to store falling balls
    balls = []

    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        image_height, image_width, _ = image.shape
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            continue

        # Check if any pose landmark is inside the circle
        for landmark in results.pose_landmarks.landmark:
            landmark_point = (int(landmark.x * image_width), int(landmark.y * image_height))
            for ball in balls:
                if is_point_inside_circle(landmark_point, ball['center'], ball['radius']):
                    ball['click_count'] += 1
                    ball['color'] = (0,0,255)
                    if ball['click_count'] >= 3:
                        score += 1
                        balls.remove(ball)
                        hit_sound.play()  # Play the hit sound

        # Update the position of falling balls
        for ball in balls:
            ball['center'] = (ball['center'][0], ball['center'][1] + 5)  # Adjust the falling speed

        # Add a new ball with a certain probability
        if np.random.rand() < 0.02:  # Adjust the probability as needed
            new_ball = {
                'center': (np.random.randint(50, 500), 0),
                'radius': 30,
                'color': [(255, 0, 0)],  # Red color
                'opacity': 1.0,
                'click_count': 0
            }
            balls.append(new_ball)

        # Draw the balls on the image
        overlay = image.copy()
        for ball in balls:
            cv2.circle(overlay, ball['center'], ball['radius'], ball['color'], -1)  # -1 fills the circle
        cv2.addWeighted(overlay, 1.0, image, 0.5, 0, image)  # Adjust the alpha for transparency

        # Draw pose landmarks on the image.
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)


        # Plot pose world landmarks.
        mp_drawing.plot_landmarks(
            results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    
    # Create a list to store falling balls
    balls = []
    score = 0  # Initialize the score variable
    start_time = time.time()
    countdown_duration = 30  # Set the countdown duration in seconds
    timer_font = cv2.FONT_HERSHEY_SIMPLEX
    timer_font_size = 1
    timer_font_color = (0, 0, 0)
    timer_font_thickness = 2

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
#         print(time.time())
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        remaining_time = max(countdown_duration - int(time.time() - start_time), 0)
        timer_text = f'Time: {remaining_time} s'
        image_umat = cv2.UMat(image)
        cv2.putText(image_umat, timer_text, (10, 70), timer_font, timer_font_size, timer_font_color, timer_font_thickness)

        # Check if any pose landmark is inside the circle
#         print(results.pose_landmarks)
        if results.pose_landmarks == None:
            pass
        else:
            for landmark in results.pose_landmarks.landmark:
                landmark_point = (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
                for ball in balls:
                    if is_point_inside_circle(landmark_point, ball['center'], ball['radius']):
                        ball['click_count'] += 1
                        ball['color'] = (0,0,255)
                        if ball['click_count'] >= 3:                        
                            # Remove the ball if clicked 3 times
                            balls.remove(ball)
                            hit_sound.play()
                            score += 1

        # Update the position of falling balls
        for ball in balls:
            ball['center'] = (ball['center'][0], ball['center'][1] + 5)  # Adjust the falling speed
#             print(ball["center"][1])

        # Add a new ball with a certain probability
        if np.random.rand() < 0.02:  # Adjust the probability as needed
            new_ball = {
                'center': (np.random.randint(50, 550), 0),
                'radius': 30,
                'color': (255, 0, 0),  # Red color
                'opacity': 1.0,
                'click_count': 0
            }
            balls.append(new_ball)

        # Draw the balls on the image
        overlay = image.copy()
        for ball in balls:
            cv2.circle(overlay, ball['center'], ball['radius'], ball['color'], -1)  # -1 fills the circle
        cv2.addWeighted(overlay, 1.0, image_umat, 0.5, 0, image_umat)  # Adjust the alpha for transparency

        # Draw the pose annotation on the image.
        # 將UMat對象轉換為numpy陣列，因為使用了cv2.putText和cv2.imshow函數，這些函數可能無法直接處理UMat對象。
        image_np = cv2.UMat.get(image_umat)
        image.flags.writeable = True
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        # print(image_umat.get().shape)
        # print(image_umat.get())
        mp_drawing.draw_landmarks(
            image_np, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(image_np, f'Score: {score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('MediaPipe Pose', image_np)
        
        k = cv2.waitKey(5)

        if time.time() - start_time >= 30:
            print(f'Total Score: {score}')
            break
            
        if k == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
