import cv2
import numpy as np
import smtplib
import pygame
import threading
import vonage
import requests
from math import sqrt
import matplotlib.pyplot as plt
import sys

# Redirect standard error to a file
sys.stderr = open('error.log', 'w')

Fire_Reported = 0
Email_Status = False
Alarm_Status = False
wtsp = False
Telegram = False
sms = False

#---------------------------TELEGRAM----------------------------------#
def telegram():
    print("Sending Telegram message...")
    base_url = "https://api.telegram.org/bot<your_bot_token>/sendMessage?chat_id=<chat_id>&text=Fire reported"
    try:
        requests.get(base_url)
        print(base_url)
    except Exception as e:
        print(f"Error sending Telegram message: {e}")

#------------------------------------------SMS MESSAGE----------------------------------#
def send_sms():
    print("Sending SMS...")
    client = vonage.Client(key="", secret="*")
    sms = vonage.Sms(client)
    try:
        responseData = sms.send_message(
            {
                "from": "Vonage APIs",
                "to": "212767790246",
                "text": "Attention!!! A fire",
            }
        )
        if responseData["messages"][0]["status"] == "0":
            print("Message sent successfully.")
        else:
            print("Message failed with error:", responseData["messages"][0]["error-text"])
    except Exception as e:
        print(f"Error sending SMS: {e}")

#--------------------ALARM SOUND--------------------------------#
def play_alarm_sound_function():
    print("Playing alarm sound...")
    try:
        pygame.mixer.init()
        s = pygame.mixer.Sound("alarm.mp3")
        s.play()
    except Exception as e:
        print(f"Error playing alarm sound: {e}")

#----------------------FIRE DETECTION--------------------------------#
def detect_fire(frame):
    global Fire_Reported, Alarm_Status, sms, Telegram
    print("Detecting fire...")
    frame = cv2.resize(frame, (960, 540))
    blur = cv2.GaussianBlur(frame, (21, 21), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower = [18, 90, 180]
    upper = [35, 255, 255]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(frame, hsv, mask=mask)
    no_red = cv2.countNonZero(mask)
    if int(no_red) > 15000:
        Fire_Reported += 1
    cv2.imshow("Fire Detection", output)
    if Fire_Reported >= 1:
        if not Alarm_Status:
            threading.Thread(target=play_alarm_sound_function).start()
            Alarm_Status = True
        if not sms:
            send_sms()
            sms = True
        if not Telegram:
            telegram()
            Telegram = True

#--------------------------------FIRE PREDICTION MODELING----------------------------#

NON_burned = 1
burned = 0

def euclidean_distance(V1, V2):
    distance = 0.0
    for i in range(len(V1) - 1):
        distance += (V1[i] - V2[i]) ** 2
    return sqrt(distance)

def find_neighbors(DATA, test_vector, num_neighbors):
    distances = []
    for DATA_vector in DATA:
        dist = euclidean_distance(test_vector, DATA_vector)
        distances.append((DATA_vector, dist))
    distances.sort(key=lambda tup: tup[1])
    
    neighbors = []
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

def predict_classification(train, test_row, num_neighbors):
    neighbors = find_neighbors(train, test_row, num_neighbors)
    classification = [row[-1] for row in neighbors]
    prediction = max(set(classification), key=classification.count)
    return prediction

# Function to input dataset data
def input_dataset():
    print("Entering dataset input mode...")
    dataset = []
    while True:
        try:
            x = float(input("Enter the value of x (or any character to quit): "))
            y = float(input("Enter the value of y: "))
            state = int(input("Enter the state (0 for Non-Burned, 1 for Burned): "))
            data_point = [x, y, state]
            dataset.append(data_point)
        except ValueError:
            break
    return dataset

# Function to plot points
def plot_points(x_values, y_values, color, marker, label):
    plt.scatter(x_values, y_values, color=color, marker=marker, label=label)

# Plot the curve before prediction
def plot_before_prediction(dataset):
    print("Plotting before prediction...")
    x_non_burned = []
    y_non_burned = []
    x_burned = []
    y_burned = []
    for data_point in dataset:
        x, y, state = data_point
        if state == 0:
            x_non_burned.append(x)
            y_non_burned.append(y)
        else:
            x_burned.append(x)
            y_burned.append(y)

    plt.subplot(1, 2, 1)
    plot_points(x_non_burned, y_non_burned, color='blue', marker='o', label='Non-Burned')
    plot_points(x_burned, y_burned, color='red', marker='x', label='Burned')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Before Prediction')
    plt.legend()

# Plot the curve after prediction
def plot_after_prediction(dataset, num_neighbors):
    print("Plotting after prediction...")
    x_non_burned_pred = []
    y_non_burned_pred = []
    x_burned_pred = []
    y_burned_pred = []
    for data_point in dataset:
        x, y, state = data_point
        prediction = predict_classification(dataset, data_point, num_neighbors)
        if prediction == 0:
            x_non_burned_pred.append(x)
            y_non_burned_pred.append(y)
        else:
            x_burned_pred.append(x)
            y_burned_pred.append(y)

    plt.subplot(1, 2, 2)
    plot_points(x_non_burned_pred, y_non_burned_pred, color='blue', marker='o', label='Non-Burned (Prediction)')
    plot_points(x_burned_pred, y_burned_pred, color='red', marker='x', label='Burned (Prediction)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('After Prediction')
    plt.legend()

def main():
    print("Starting the program...")
    choice = input("Who are you? Firefighter or No: ").strip().lower()
    if choice == "no":
        print("Starting video capture for fire detection...")
        video = cv2.VideoCapture(0)
        while True:
            grabbed, frame = video.read()
            if not grabbed:
                break
            detect_fire(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                pygame.quit()
                break
        cv2.destroyAllWindows()
        video.release()
    else:
        print("ENTER DATASET OF YOUR LOCATION:")
        dataset = input_dataset()

        num_neighbors = int(input("Enter K: "))

        plt.figure(figsize=(12, 6))
        plot_before_prediction(dataset)
        plot_after_prediction(dataset, num_neighbors)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()