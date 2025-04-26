import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

# Cooldown timers
last_action_time = {
    'left': 0,
    'right': 0,
    'up': 0,
    'down': 0
}
cooldown = 0.8  # seconds

def fingers_up(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    fingers = []

    for tip in finger_tips:
        is_up = hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y
        fingers.append(is_up)

    thumb_tip = hand_landmarks.landmark[4].x
    thumb_ip = hand_landmarks.landmark[3].x
    fingers.insert(0, thumb_tip < thumb_ip)

    return fingers  # [thumb, index, middle, ring, pinky]

def is_only_index_up(fingers):
    return fingers[1] and not any(f for i, f in enumerate(fingers) if i != 1)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    current_time = time.time()

    left_hand = None
    right_hand = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, hand_handedness in enumerate(results.multi_handedness):
            hand_label = hand_handedness.classification[0].label
            hand_landmarks = results.multi_hand_landmarks[i]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if hand_label == "Right":
                right_hand = hand_landmarks
            elif hand_label == "Left":
                left_hand = hand_landmarks

    # 🎯 First: check for SLIDE (both hands together)
    if left_hand and right_hand:
        lf = fingers_up(left_hand)
        rf = fingers_up(right_hand)
        if lf[1] and rf[1] and not any(lf[2:]) and not any(rf[2:]):
            if current_time - last_action_time['down'] > cooldown:
                print("✌️ Both index fingers → SLIDE")
                pyautogui.press('down')
                last_action_time['down'] = current_time
        else:
            # If not sliding, check for other gestures
            if right_hand:
                right_fingers = fingers_up(right_hand)
                if is_only_index_up(right_fingers):
                    if current_time - last_action_time['right'] > cooldown:
                        print("👉 Right index up → RIGHT")
                        pyautogui.press('right')
                        last_action_time['right'] = current_time
                elif all(right_fingers):
                    if current_time - last_action_time['up'] > cooldown:
                        print("🖐️ All right fingers up → JUMP")
                        pyautogui.press('up')
                        last_action_time['up'] = current_time

            if left_hand:
                left_fingers = fingers_up(left_hand)
                if is_only_index_up(left_fingers):
                    if current_time - last_action_time['left'] > cooldown:
                        print("👈 Left index up → LEFT")
                        pyautogui.press('left')
                        last_action_time['left'] = current_time
    else:
        # If only one hand detected
        if right_hand:
            right_fingers = fingers_up(right_hand)
            if is_only_index_up(right_fingers):
                if current_time - last_action_time['right'] > cooldown:
                    print("👉 Right index up → RIGHT")
                    pyautogui.press('right')
                    last_action_time['right'] = current_time
            elif all(right_fingers):
                if current_time - last_action_time['up'] > cooldown:
                    print("🖐️ All right fingers up → JUMP")
                    pyautogui.press('up')
                    last_action_time['up'] = current_time

        if left_hand:
            left_fingers = fingers_up(left_hand)
            if is_only_index_up(left_fingers):
                if current_time - last_action_time['left'] > cooldown:
                    print("👈 Left index up → LEFT")
                    pyautogui.press('left')
                    last_action_time['left'] = current_time

    cv2.imshow("Subway Surfer Hand Controller", frame)

    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
