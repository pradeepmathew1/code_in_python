import tkinter as tk
import random


class RockPaperScissors:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Rock Paper Scissors")
        self.window.geometry("400x300")
        self.window.resizable(0, 0)

        self.options = ("rock", "paper", "scissors")

        self.create_widgets()

    def create_widgets(self):
        self.label = tk.Label(self.window, text="Choose Rock, Paper, or Scissors:", font=("Calibri", 16))
        self.label.pack(pady=20)

        self.button_frame = tk.Frame(self.window)
        self.button_frame.pack()

        self.rock_button = tk.Button(self.button_frame, text="Rock", font=("Calibri", 14),
                                     command=lambda: self.play("rock"))
        self.rock_button.grid(row=0, column=0, padx=10)

        self.paper_button = tk.Button(self.button_frame, text="Paper", font=("Calibri", 14),
                                      command=lambda: self.play("paper"))
        self.paper_button.grid(row=0, column=1, padx=10)

        self.scissors_button = tk.Button(self.button_frame, text="Scissors", font=("Calibri", 14),
                                         command=lambda: self.play("scissors"))
        self.scissors_button.grid(row=0, column=2, padx=10)

        self.result_label = tk.Label(self.window, text="", font=("Calibri", 16))
        self.result_label.pack(pady=20)

        self.play_again_button = tk.Button(self.window, text="Play Again", font=("Calibri", 14),
                                           command=self.reset_game)
        self.play_again_button.pack(pady=10)
        self.play_again_button.config(state="disabled")

        self.quit_button = tk.Button(self.window, text="Quit", font=("Calibri", 14), command=self.window.quit)
        self.quit_button.pack(pady=10)

    def play(self, player_choice):
        computer_choice = random.choice(self.options)

        result_text = f"Player: {player_choice}\nComputer: {computer_choice}\n"
        if player_choice == computer_choice:
            result_text += "It's a tie!"
        elif (player_choice == "rock" and computer_choice == "scissors") or \
                (player_choice == "paper" and computer_choice == "rock") or \
                (player_choice == "scissors" and computer_choice == "paper"):
            result_text += "You win!"
        else:
            result_text += "Computer wins, you lose!"

        self.result_label.config(text=result_text)
        self.play_again_button.config(state="normal")

    def reset_game(self):
        self.result_label.config(text="")
        self.play_again_button.config(state="disabled")

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    game = RockPaperScissors()
    game.run()
