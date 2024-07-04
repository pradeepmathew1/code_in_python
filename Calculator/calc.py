import tkinter as tk

# Defining font styles
LARGE_FONT = ("Arial", 40, "bold")
MEDIUM_FONT = ("Arial", 18)
BUTTON_FONT = ("Arial", 24, "bold")
DEFAULT_FONT = ("Arial", 22)

# Defining color scheme
BACKGROUND_COLOR = "#E8E8E8"
BUTTON_COLOR = "#FFFFFF"
ACTIVE_BUTTON_COLOR = "#B0E0E6"
DISPLAY_COLOR = "#DCDCDC"
TEXT_COLOR = "#000000"

class SimpleCalculator:
    def __init__(self):
        self.window = tk.Tk()
        self.window.geometry("400x600")
        self.window.resizable(0, 0)
        self.window.title("Simple Calculator")

        self.current_input = ""
        self.total_input = ""
        self.display_frame = self.create_display_frame()

        self.total_label, self.current_label = self.create_display_labels()

        self.digits = {
            7: (1, 1), 8: (1, 2), 9: (1, 3),
            4: (2, 1), 5: (2, 2), 6: (2, 3),
            1: (3, 1), 2: (3, 2), 3: (3, 3),
            0: (4, 2), '.': (4, 1)
        }
        self.operations = {"/": "÷", "*": "×", "-": "-", "+": "+"}
        self.buttons_frame = self.create_buttons_frame()
#These methods all used to take inout from the keyboard
        self.setup_grid_config()
        self.create_digit_buttons()
        self.create_operation_buttons()
        self.create_special_buttons()
        self.bind_keyboard_events()
# In this method, we have just included the digits, the operations that we need to include, inorder to take input from keyboard using keyboard_events method, etc.

    def setup_grid_config(self):
        self.buttons_frame.rowconfigure(0, weight=1)
        for x in range(1, 5):
            self.buttons_frame.rowconfigure(x, weight=1)
            self.buttons_frame.columnconfigure(x, weight=1)

    def bind_keyboard_events(self):
        self.window.bind("<Return>", lambda event: self.evaluate())
        for key in self.digits:
            self.window.bind(str(key), lambda event, digit=key: self.add_digit(digit))

        for key in self.operations:
            self.window.bind(key, lambda event, operator=key: self.add_operator(operator))
# These methods are for clearing the input, or taking the arthimethic operators like sqrt, square, etc for complex problems
    def create_special_buttons(self):
        self.create_clear_button()
        self.create_equals_button()
        self.create_square_button()
        self.create_sqrt_button()

    def create_display_labels(self):
        total_label = tk.Label(self.display_frame, text=self.total_input, anchor=tk.E, bg=DISPLAY_COLOR,
                               fg=TEXT_COLOR, padx=24, font=MEDIUM_FONT)
        total_label.pack(expand=True, fill='both')

        current_label = tk.Label(self.display_frame, text=self.current_input, anchor=tk.E, bg=DISPLAY_COLOR,
                                 fg=TEXT_COLOR, padx=24, font=LARGE_FONT)
        current_label.pack(expand=True, fill='both')

        return total_label, current_label
#This method used to display the application
    def create_display_frame(self):
        frame = tk.Frame(self.window, height=150, bg=DISPLAY_COLOR)
        frame.pack(expand=True, fill="both")
        return frame
#This takes the input
    def add_digit(self, digit):
        self.current_input += str(digit)
        self.update_current_label()

    def create_digit_buttons(self):
        for digit, grid_value in self.digits.items():
            button = tk.Button(self.buttons_frame, text=str(digit), bg=BUTTON_COLOR, fg=TEXT_COLOR, font=BUTTON_FONT,
                               borderwidth=0, command=lambda x=digit: self.add_digit(x))
            button.grid(row=grid_value[0], column=grid_value[1], sticky=tk.NSEW)

    def add_operator(self, operator):
        self.current_input += operator
        self.total_input += self.current_input
        self.current_input = ""
        self.update_total_label()
        self.update_current_label()

    def create_operation_buttons(self):
        i = 0
        for operator, symbol in self.operations.items():
            button = tk.Button(self.buttons_frame, text=symbol, bg=BACKGROUND_COLOR, fg=TEXT_COLOR, font=DEFAULT_FONT,
                               borderwidth=0, command=lambda x=operator: self.add_operator(x))
            button.grid(row=i, column=4, sticky=tk.NSEW)
            i += 1

    def clear(self):
        self.current_input = ""
        self.total_input = ""
        self.update_current_label()
        self.update_total_label()

    def create_clear_button(self):
        button = tk.Button(self.buttons_frame, text="C", bg=BACKGROUND_COLOR, fg=TEXT_COLOR, font=DEFAULT_FONT,
                           borderwidth=0, command=self.clear)
        button.grid(row=0, column=1, sticky=tk.NSEW)

    def square(self):
        self.current_input = str(eval(f"{self.current_input}**2"))
        self.update_current_label()

    def create_square_button(self):
        button = tk.Button(self.buttons_frame, text="x²", bg=BACKGROUND_COLOR, fg=TEXT_COLOR, font=DEFAULT_FONT,
                           borderwidth=0, command=self.square)
        button.grid(row=0, column=2, sticky=tk.NSEW)

    def sqrt(self):
        self.current_input = str(eval(f"{self.current_input}**0.5"))
        self.update_current_label()

    def create_sqrt_button(self):
        button = tk.Button(self.buttons_frame, text="√x", bg=BACKGROUND_COLOR, fg=TEXT_COLOR, font=DEFAULT_FONT,
                           borderwidth=0, command=self.sqrt)
        button.grid(row=0, column=3, sticky=tk.NSEW)
# This triggerd to evaluate the expression
    def evaluate(self):
        self.total_input += self.current_input
        self.update_total_label()
        try:
            self.current_input = str(eval(self.total_input))
            self.total_input = ""
        except Exception as e:
            self.current_input = "Error"
        finally:
            self.update_current_label()
#This equals button says to program that to work on calculations on the expresion
    def create_equals_button(self):
        button = tk.Button(self.buttons_frame, text="=", bg=ACTIVE_BUTTON_COLOR, fg=TEXT_COLOR, font=DEFAULT_FONT,
                           borderwidth=0, command=self.evaluate)
        button.grid(row=4, column=3, columnspan=2, sticky=tk.NSEW)

    def create_buttons_frame(self):
        frame = tk.Frame(self.window)
        frame.pack(expand=True, fill="both")
        return frame

    def update_total_label(self):
        expression = self.total_input
        for operator, symbol in self.operations.items():
            expression = expression.replace(operator, f' {symbol} ')
        self.total_label.config(text=expression)
#Updating the label
    def update_current_label(self):
        self.current_label.config(text=self.current_input[:11])

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    calculator_app = SimpleCalculator()
    calculator_app.run()
