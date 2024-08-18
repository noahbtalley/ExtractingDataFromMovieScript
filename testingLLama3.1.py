from PIL import Image
import pytesseract
import os
import openai
import json
import cv2
import platform
import re

# Set up LM Studio client to point to the local server
client = openai.OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def setup_tesseract():
    system = platform.system()
    if system == "Windows":
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    elif system == "Darwin":  # macOS
        pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
    else:
        raise OSError("Unsupported operating system")

setup_tesseract()

SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}

def preprocess_image(image_path):
    try:
        image = Image.open(image_path)
        gray = image.convert('L')  # Convert image to grayscale for better OCR performance
        return gray
    except Exception as e:
        raise RuntimeError(f"An error occurred while preprocessing the image: {str(e)}")

def detect_low_quality_image(image_path):
    try:
        # Load image using OpenCV
        image = cv2.imread(image_path)
        
        # Check the resolution of the image
        height, width = image.shape[:2]
        if height < 500 or width < 500:  # Threshold is 500x500
            return True, "Image resolution is too low."
        
        return False, "Image quality is sufficient."
    
    except Exception as e:
        raise RuntimeError(f"An error occurred while checking image quality: {str(e)}")

def extract_text_from_image(image_path):
    try:
        is_low_quality, reason = detect_low_quality_image(image_path)
        if is_low_quality:
            raise ValueError(f"Low quality image detected: {reason}")

        image = preprocess_image(image_path)
        text = pytesseract.image_to_string(image)
        if not text.strip():
            raise ValueError("No text could be extracted from the image. It might be too low-quality or empty.")
        return text
    except Exception as e:
        raise RuntimeError(f"An error occurred while extracting text from the image: {str(e)}")

def generate_custom_prompt(extracted_text):
    try:
        prompt = f"""
        Extract and structure the key information in JSON format. Ensure that all characters mentioned in the scene are included in the 'characters' field, \
        along with both their actions and dialogues. Keep scene descriptions concise, focusing only on the most important actions and settings. \
        If a scene or any other description is not included in the text, omit the related field.

        Example output format:

        {{
            "scenes": [
                {{
                    "location": "Scene location",
                    "time": "Scene time",
                    "characters": [
                        {{
                            "name": "Character Name",
                            "actions": [
                                "Action 1",
                                "Action 2"
                            ],
                            "dialogue": "Character's dialogue here"
                        }},
                        {{
                            "name": "Another Character",
                            "actions": [
                                "Action 1",
                                "Action 2"
                            ],
                            "dialogue": "Their dialogue here"
                        }}
                    ]
                }}
            ]
        }}

        Text: {extracted_text}
        """
        return prompt
    except Exception as e:
        raise RuntimeError(f"An error occurred while generating the prompt: {str(e)}")

def get_llama_response(prompt):
    try:
        completion = client.chat.completions.create(
            model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
            messages=[
                {"role": "system", "content": "You are an expert at reading movie scripts and organizing them into a structured JSON format. \
                Extract character names, dialogue, and scene descriptions efficiently, and omit unnecessary information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        
        # Accessing the content correctly
        message_content = completion.choices[0].message.content.strip()
        return message_content

    except Exception as e:
        raise RuntimeError(f"An error occurred while communicating with Llama 3.1 via LM Studio: {str(e)}")

def extract_json_from_response(response_content):
    try:
        # Use regular expression to find the JSON block within the response, ignoring any non-JSON prefixes like "json"
        json_match = re.search(r"\{.*\}", response_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0).strip()
            return json_str
        else:
            raise ValueError("No JSON block found in the response content.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while extracting JSON from the response: {str(e)}")

def save_response_as_json(response_content, filename):
    try:
        if not response_content or response_content.strip() == "":
            raise ValueError("Received empty response content. Cannot save empty JSON.")
        
        # Extract the JSON part from the response
        json_str = extract_json_from_response(response_content)

        try:
            # Attempt to parse the JSON string
            response_json = json.loads(json_str)
        except json.JSONDecodeError as e:
            # If parsing fails, log the response content for debugging
            print(f"Failed to decode JSON for {filename}. Content: {json_str}")
            raise e
        
        # Save the JSON object as a file
        with open(filename, 'w') as json_file:
            json.dump(response_json, json_file, indent=4)

        print(f"Response successfully saved as {filename}")

    except ValueError as e:
        print(f"An error occurred: {str(e)}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"An error occurred while decoding JSON: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while saving the JSON file: {str(e)}")

def process_all_images_in_folder(folder_path):
    # Create a list of all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is an image based on supported extensions
        if os.path.splitext(filename)[1].lower() in SUPPORTED_EXTENSIONS:
            image_path = os.path.join(folder_path, filename)
            print(f"Processing file: {image_path}")
            llama_response = main(image_path)
            
            if llama_response:
                json_filename = f"{os.path.splitext(filename)[0]}.json"
                save_response_as_json(llama_response, json_filename)

def main(image_path):
    try:
        extracted_text = extract_text_from_image(image_path)
        if not extracted_text:
            print(f"No text extracted from {image_path}. Skipping.")
            return None
        
        prompt = generate_custom_prompt(extracted_text)
        llama_response = get_llama_response(prompt)
        
        if not llama_response or llama_response.strip() == "":
            print(f"No response from Llama for {image_path}. Skipping.")
            return None
        
        return llama_response
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    folder_path = "."  # Set folder path to current directory
    process_all_images_in_folder(folder_path)
