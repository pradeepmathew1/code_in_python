import tkinter as tk
import string
import random
# These are the style pattern that we follow inorder to create GUI
LARGE_FONT_STYLE = ("Calibri", 24, "bold")
SMALL_FONT_STYLE = ("Calibri", 18)
BUTTON_FONT_STYLE = ("Calibri", 14, "bold")
DEFAULT_FONT_STYLE = ("Calibri", 16)

OFF_WHITE = "#F8FAFF"
WHITE = "#FFFFFF"
LIGHT_BLUE = "#CCEDFF"
LIGHT_GRAY = "#F5F5F5"
LABEL_COLOR = "#25265E"

class PasswordGenerator:
    def __init__(self):
        self.window = tk.Tk()
        self.window.geometry("400x400")
        self.window.resizable(0, 0)
        self.window.title("Password Generator")

        self.length_var = tk.IntVar()
        self.password_var = tk.StringVar()

        self.create_widgets()
# Widget related code block
    def create_widgets(self):
        self.display_frame = self.create_display_frame()
        self.entry_frame = self.create_entry_frame()
        self.buttons_frame = self.create_buttons_frame()

        self.create_display_labels()
        self.create_entry()
        self.create_generate_button()
        self.create_copy_button()
#These methods returns the frame which finally all combined is -> Application
    def create_display_frame(self):
        frame = tk.Frame(self.window, height=100, bg=LIGHT_GRAY)
        frame.pack(expand=True, fill="both")
        return frame

    def create_entry_frame(self):
        frame = tk.Frame(self.window, height=100, bg=WHITE)
        frame.pack(expand=True, fill="both")
        return frame

    def create_buttons_frame(self):
        frame = tk.Frame(self.window, height=100, bg=OFF_WHITE)
        frame.pack(expand=True, fill="both")
        return frame

    def create_display_labels(self):
        label = tk.Label(self.display_frame, text="Generated Password", anchor=tk.CENTER, bg=LIGHT_GRAY,
                         fg=LABEL_COLOR, font=LARGE_FONT_STYLE)
        label.pack(expand=True, fill='both')

        self.password_display = tk.Entry(self.display_frame, textvariable=self.password_var, font=LARGE_FONT_STYLE,
                                         bg=LIGHT_GRAY, fg=LABEL_COLOR, justify='center', state='readonly', borderwidth=0)
        self.password_display.pack(expand=True, fill='both')

    def create_entry(self):
        label = tk.Label(self.entry_frame, text="Enter Password Length:", font=DEFAULT_FONT_STYLE, bg=WHITE, fg=LABEL_COLOR)
        label.pack(pady=10)

        length_entry = tk.Entry(self.entry_frame, textvariable=self.length_var, font=DEFAULT_FONT_STYLE, justify='center', borderwidth=1)
        length_entry.pack(pady=10)

    def create_generate_button(self):
        button = tk.Button(self.buttons_frame, text="Generate Password", bg=LIGHT_BLUE, fg=LABEL_COLOR, font=BUTTON_FONT_STYLE,
                           borderwidth=0, command=self.generate_password)
        button.pack(expand=True, fill='both', pady=10)

    def create_copy_button(self):
        button = tk.Button(self.buttons_frame, text="Copy to Clipboard", bg=LIGHT_BLUE, fg=LABEL_COLOR, font=BUTTON_FONT_STYLE,
                           borderwidth=0, command=self.copy_to_clipboard)
        button.pack(expand=True, fill='both', pady=10)

    def generate_password(self):
        try:
            length = self.length_var.get()
            self.password_var.set(self.create_password(length))
        except ValueError as e:
            self.password_var.set("Error")

    def create_password(self, length):
        if length < 4:
            raise ValueError("Password length should be at least 4 characters.")

        letters = string.ascii_letters
        digits = string.digits
        punctuation = string.punctuation

        # Ensuring the password includes at least one letter, one digit, and one punctuation mark
        password = [
            random.choice(letters),
            random.choice(digits),
            random.choice(punctuation)
        ]

        all_chars = letters + digits + punctuation
        password += random.choices(all_chars, k=length - 3)

        random.shuffle(password)
        return ''.join(password)

    def copy_to_clipboard(self):
        self.window.clipboard_clear()
        self.window.clipboard_append(self.password_var.get())
        self.window.update()  # now it stays on the clipboard after the window is closed

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = PasswordGenerator()
    app.run()
