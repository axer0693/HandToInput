import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import Label, Button, Entry, Listbox, END, SINGLE
from PIL import Image, ImageTk
import numpy as np
import math
import pyautogui
import csv
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

saved_patterns = []
CSV_FILE = "hand_patterns.csv"

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

def compare_hand_patterns(saved_pattern, current_pattern, threshold=0.1):
    total_distance = 0
    for i in range(21):
        total_distance += euclidean_distance(saved_pattern[i], current_pattern[i])
    avg_distance = total_distance / 21
    return avg_distance < threshold

def save_patterns_to_csv():
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['landmark_x', 'landmark_y', 'landmark_z'] * 21 + ['assigned_key'])
        for pattern, key in saved_patterns:
            flat_pattern = [coord for landmark in pattern for coord in landmark]
            writer.writerow(flat_pattern + [key])

def load_patterns_from_csv(app):
    try:
        with open(CSV_FILE, mode='r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                pattern = [(float(row[i]), float(row[i+1]), float(row[i+2])) for i in range(0, 63, 3)]
                key = row[63]
                saved_patterns.append((pattern, key))
                app.pattern_listbox.insert(END, f"Pattern {len(saved_patterns)} -> {key}")
    except FileNotFoundError:
        print("CSV file not found. Starting with an empty pattern list.")

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.vid = VideoCapture(self.video_source)
        self.canvas = Label(window)
        self.canvas.pack()
        self.save_button = Button(window, text="Save Current Hand Pattern", width=50, command=self.save_hand_pattern)
        self.save_button.pack(anchor=tk.CENTER, expand=True)
        self.key_entry = Entry(window, width=50)
        self.key_entry.pack(anchor=tk.CENTER, expand=True)
        self.key_entry.insert(0, "Enter a key to assign to the saved pattern...")
        self.pattern_listbox = Listbox(window, width=50, selectmode=SINGLE)
        self.pattern_listbox.pack(anchor=tk.CENTER, expand=True)
        self.delete_button = Button(window, text="Delete Selected Pattern", width=50, command=self.delete_selected_pattern)
        self.delete_button.pack(anchor=tk.CENTER, expand=True)
        load_patterns_from_csv(self)
        self.hand_detected = False
        self.current_pattern = None
        self.last_keypress_time = 0
        self.cooldown = 1
        self.update()
        self.window.mainloop()

    def save_hand_pattern(self):
        entered_key = self.key_entry.get()
        if self.hand_detected and self.current_pattern and entered_key:
            saved_patterns.append((self.current_pattern, entered_key))
            self.pattern_listbox.insert(END, f"Pattern {len(saved_patterns)} -> {entered_key}")
            print(f"Hand pattern saved and mapped to '{entered_key}'")
            save_patterns_to_csv()

    def delete_selected_pattern(self):
        selected_index = self.pattern_listbox.curselection()
        if selected_index:
            selected_index = selected_index[0]
            del saved_patterns[selected_index]
            self.pattern_listbox.delete(selected_index)
            print(f"Deleted pattern {selected_index + 1}")
            save_patterns_to_csv()

    def update(self):
        ret, frame, landmarks = self.vid.get_frame()
        if ret:
            self.hand_detected = landmarks is not None
            if self.hand_detected:
                self.current_pattern = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
                for pattern, assigned_key in saved_patterns:
                    if compare_hand_patterns(pattern, self.current_pattern):
                        self.perform_action(assigned_key)
                        break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk
            self.canvas.configure(image=imgtk)
        self.window.after(10, self.update)

    def perform_action(self, action):
        current_time = time.time()
        if current_time - self.last_keypress_time >= self.cooldown:
            pyautogui.press(action)
            print(f"{action} key pressed!")
            self.last_keypress_time = current_time
        else:
            print("Cooldown in effect. Please wait a second before pressing the key again.")

class VideoCapture:
    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    return (ret, frame, results.multi_hand_landmarks[0])
                else:
                    return (ret, frame, None)
            else:
                return (ret, None, None)
        else:
            return (False, None, None)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root, "Hand Pattern to Key Mapping")
