# encrypt_weights.py
import os
import base64
import getpass
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

# --- Configuration ---
ORIGINAL_MODEL_FILENAME = 'region_segmentation_model.pth'  # <<<--- IMPORTANT: Set to your actual model filename (.pt or .pth)
ENCRYPTED_OUTPUT_FILENAME = 'encrypted_model.bin' # Output file name (will be embedded)
SALT_SIZE = 16  # Standard salt size
# Use a high number of iterations to make brute-forcing harder
PBKDF2_ITERATIONS = 390000
# --- End Configuration ---

def encrypt_file(input_filename, output_filename, password):
    """Encrypts a file using a password-derived key and saves salt + encrypted data."""
    if not os.path.exists(input_filename):
        print(f"Error: Input file not found: {input_filename}")
        return

    # 1. Generate a random salt
    salt = os.urandom(SALT_SIZE)

    # 2. Derive a key from the password and salt using PBKDF2
    print(f"Deriving key (using {PBKDF2_ITERATIONS} iterations)... This may take a moment.")
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,  # Fernet key size
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
        backend=default_backend()
    )
    # Fernet keys must be url-safe base64 encoded
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    print("Key derived.")

    # 3. Create Fernet cipher
    cipher_suite = Fernet(key)

    # 4. Read the original file content
    try:
        with open(input_filename, 'rb') as f_in:
            file_data = f_in.read()
        print(f"Read {len(file_data)} bytes from {input_filename}")
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    # 5. Encrypt the data
    try:
        encrypted_data = cipher_suite.encrypt(file_data)
        print("Data encrypted.")
    except Exception as e:
        print(f"Error during encryption: {e}")
        return

    # 6. Write the salt FOLLOWED BY the encrypted data to the output file
    try:
        with open(output_filename, 'wb') as f_out:
            f_out.write(salt)             # Write salt first
            f_out.write(encrypted_data)   # Then write encrypted data
        print(f"Encrypted data saved to {output_filename} (Salt included)")
        print("IMPORTANT: Keep this password safe! You'll need it to run the application.")
    except Exception as e:
        print(f"Error writing encrypted file: {e}")

if __name__ == "__main__":
    print(f"This script will encrypt '{ORIGINAL_MODEL_FILENAME}' into '{ENCRYPTED_OUTPUT_FILENAME}'.")
    password = getpass.getpass("Enter a strong password for encryption: ")
    password_confirm = getpass.getpass("Confirm password: ")

    if password == password_confirm:
        if password: # Ensure password is not empty
             encrypt_file(ORIGINAL_MODEL_FILENAME, ENCRYPTED_OUTPUT_FILENAME, password)
        else:
             print("Error: Password cannot be empty.")
    else:
        print("Error: Passwords do not match.")

