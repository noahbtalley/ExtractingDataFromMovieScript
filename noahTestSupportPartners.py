#This program is designed to take a screenshot of a script and extract all information from it,
#including scene, location, and character's actions/dialogue. Chat-GPT will output the information as a JSON and save it to the current directory.
#Script extraction was chosen to keep with the theme..

#Written by Noah Talley on 8/16/2024 for Support Partners test.

#Note: image file must be rasterized--no vector images--.jpg, .png, .bmp, .tiff, and .gif are all acceptable formats.
#      also checks for edge cases such as improper file types and too small of an image < 500x500.

#      TO RUN THIS PROGRAM: Add your OpenAI key (line 23) and make sure program is in the same folder as images.



from PIL import Image
import pytesseract
import os
from openai import OpenAI
import json
import cv2
import platform

# add in personal key within the quotation marks to use
client = OpenAI(
    api_key="INSERT YOUR OPENAI KEY HERE"
)

def setup_tesseract():
    system = platform.system()
    if system == "Windows":
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    elif system == "Darwin": # macbook
        pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
    else:
        raise OSError("Unsupported operating system")

setup_tesseract()

SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}

def preprocess_image(image_path):
    try:
        image = Image.open(image_path)
        gray = image.convert('L')  # make image black and white for increased reliability with optical character recognition
        return gray
    except Exception as e:
        raise RuntimeError(f"An error occurred while preprocessing the image: {str(e)}")

def detect_low_quality_image(image_path):
    try:
        # load image using openCV
        image = cv2.imread(image_path)
        
        # checks the resolution of the image (hence no vectors)
        height, width = image.shape[:2]
        if height < 500 or width < 500:  # threshold is 500x500
            return True, "Image resolution is too low."
        
        return False, "Image quality is sufficient."
    
    except Exception as e:
        raise RuntimeError(f"An error occurred while checking image quality: {str(e)}")

def extract_text_from_image(image_path):
    try:
        is_low_quality, reason = detect_low_quality_image(image_path)       # before it extracts the text it will check if the quality was too low.. if not it will continue.
        if is_low_quality:
            raise ValueError(f"Low quality image detected: {reason}")

        image = preprocess_image(image_path)  # uses previously defined function to 'pre-process' image.
        text = pytesseract.image_to_string(image)
        if not text.strip():
            raise ValueError("No text could be extracted from the image. It might be too low-quality or empty.")
        return text
    except Exception as e:
        raise RuntimeError(f"An error occurred while extracting text from the image: {str(e)}")
    
# custom prompt from the user specifying that it shall extract and structure the inforation in json format and to focus on the most important actions and settings. provides example.
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
    
#created this prompt telling chat gpt it is an expert at reading movies. the user prompt also includes an example of how the json format should look.
def get_gpt_response(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert at reading movie scripts and organizing them into a structured JSON format. \
                 Extract character names, dialogue, and scene descriptions efficiently, and omit unnecessary information."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=750,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        raise RuntimeError(f"An error occurred while communicating with the GPT API: {str(e)}")
    
def save_response_as_json(response_content, filename):
    try:
        # converts the gpt response to a json format
        response_json = json.loads(response_content)

        # save this object as a file
        with open(filename, 'w') as json_file:
            json.dump(response_json, json_file, indent=4)

        print(f"Response successfully saved as {filename}")

    except Exception as e:
        raise RuntimeError(f"An error occurred while saving the JSON file: {str(e)}")


def process_all_images_in_folder(folder_path):
    # creates list of all files in folder
    for filename in os.listdir(folder_path):
        # checks if the file is an image according to supported extensions
        if os.path.splitext(filename)[1].lower() in SUPPORTED_EXTENSIONS:
            image_path = os.path.join(folder_path, filename)
            print(f"Processing file: {image_path}")
            gpt_response = main(image_path)
            
            if gpt_response:
                json_filename = f"{os.path.splitext(filename)[0]}.json"
                save_response_as_json(gpt_response, json_filename)

def main(image_path):

    #this function extracts the text, generates the gpt prompt, sends it to GPT-4 (or whatever model.. just change it out in the function),
    # and returns a structured json response

    try:
        extracted_text = extract_text_from_image(image_path)
        prompt = generate_custom_prompt(extracted_text)
        gpt_response = get_gpt_response(prompt)
        return gpt_response
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    folder_path = "."  # folder path set to current directory.
    process_all_images_in_folder(folder_path)
