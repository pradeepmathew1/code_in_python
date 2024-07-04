import string
import random

def generate_password(length): # method to generate a random password
    # Define character sets
    letters = string.ascii_letters
    digits = string.digits
    punctuation = string.punctuation

    # Ensuring that password includes at least one character from each set for providing better password
    password = [
        random.choice(letters),
        random.choice(digits),
        random.choice(punctuation)
    ]

    # Filling the rest of the password length with a mix of all characters
    all_chars = letters + digits + punctuation
    password += random.choices(all_chars, k=length - 3)

    # Shuffling to prevent any predictable patterns
    random.shuffle(password)
    return ''.join(password)

if __name__ == "__main__":
    try:
        passlen = int(input("Enter the password length: "))
        generated_password = generate_password(passlen)
        print("Generated password is ->")
        print(generated_password)
    except ValueError as e:
        print(f"Error: {e}")
