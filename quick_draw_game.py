import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import random
import time
from threading import Thread

model = tf.keras.models.load_model("model_final.h5")
class_names = ['gazebo', 'kalachuchi', 'Siklab', 'Tomorrow logo']

CANVAS_SIZE = 280
DRAW_SIZE = 5
TIME_LIMIT = 15

class QuickDrawGame:
    def __init__(self, master):
        self.master = master
        self.master.title("Quick, Draw! (Pisay Edition)")

        self.prompts = random.sample(class_names, len(class_names))
        self.round_index = 0
        self.score = 0

        self.label = tk.Label(master, text="", font=("Arial", 18, "bold"))
        self.label.pack(pady=(10, 5))

        self.canvas = tk.Canvas(master, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black")
        self.canvas.pack()

        self.predict_label = tk.Label(master, text="", font=("Arial", 14))
        self.predict_label.pack()

        self.clear_button = tk.Button(master, text="CLEAR", command=self.clear_canvas)
        self.clear_button.pack(pady=5)

        self.timer_label = tk.Label(master, text="", font=("Arial", 14))
        self.timer_label.pack(pady=(5, 10))

        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        self.start_new_round()

    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x - DRAW_SIZE, y - DRAW_SIZE, x + DRAW_SIZE, y + DRAW_SIZE, fill="white", outline="white")
        self.draw.ellipse([x - DRAW_SIZE, y - DRAW_SIZE, x + DRAW_SIZE, y + DRAW_SIZE], fill=255)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.predict_label.config(text="")

    def start_new_round(self):
        self.clear_canvas()
        if self.round_index >= len(self.prompts):
            self.end_game()
            return
        self.current_class = self.prompts[self.round_index]
        self.label.config(text=f"Draw: {self.current_class}")
        self.remaining_time = TIME_LIMIT
        self.update_timer()
        Thread(target=self.countdown).start()

    def update_timer(self):
        self.timer_label.config(text=f"Time left: {self.remaining_time}s")

    def countdown(self):
        while self.remaining_time > 0:
            time.sleep(1)
            self.remaining_time -= 1
            self.update_timer()
        self.predict()

    def predict(self):
        img_resized = self.image.resize((28, 28))
        img_array = np.array(img_resized).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))
        prediction = model.predict(img_array)[0]
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = prediction[predicted_index]

        if confidence >= 0.95:
            if predicted_class == self.current_class:
                self.score += 1
                result = f"The AI guessed: {predicted_class} ({confidence*100:.1f}%)\nCorrect!"
            else:
                result = f"The AI guessed: {predicted_class} ({confidence*100:.1f}%)\nWrong!"
        else:
            result = "The AI doesn't know what you're drawing..."

        self.predict_label.config(text=result)
        self.round_index += 1
        self.master.after(3000, self.start_new_round)


    def end_game(self):
        self.canvas.pack_forget()
        self.clear_button.pack_forget()
        self.timer_label.config(text="")
        self.label.config(text="GAME OVER!")
        self.predict_label.config(text=f"You scored {self.score} out of {len(self.prompts)}!")

        self.new_game_button = tk.Button(self.master, text="NEW GAME", font=("Arial", 14), command=self.reset_game)
        self.new_game_button.pack(pady=10)

    def reset_game(self):
        self.prompts = random.sample(class_names, len(class_names))
        self.round_index = 0
        self.score = 0
        self.new_game_button.destroy()

        self.canvas.pack()
        self.clear_button.pack(pady=5)
        self.predict_label.config(text="")
        self.start_new_round()


root = tk.Tk()
game = QuickDrawGame(root)
root.attributes("-fullscreen", True)
root.mainloop()