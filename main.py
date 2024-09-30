import os
import time
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from file_utils import (
    display_directory_tree,
    collect_file_paths,
    separate_files_by_type,
    read_text_file,
    read_pdf_file,
    read_docx_file,
    sanitize_filename,
    create_folder
)

from nexa.gguf import NexaVLMInference, NexaTextInference
import contextlib
import sys

# Global variables to hold the models
image_inference = None
text_inference = None

@contextlib.contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def initialize_models():
    """Initialize the models if they haven't been initialized yet."""
    global image_inference, text_inference
    if image_inference is None or text_inference is None:
        with suppress_stdout_stderr():
            # Initialize the models
            model_path = "llava-v1.6-vicuna-7b:q4_0"
            model_path_text = "gemma-2-2b-instruct:q4_0"

            # Initialize the image inference model
            image_inference = NexaVLMInference(
                model_path=model_path,
                local_path=None,
                stop_words=[],
                temperature=0.3,
                max_new_tokens=256,
                top_k=3,
                top_p=0.2,
                profiling=False
            )

            # Initialize the text inference model
            text_inference = NexaTextInference(
                model_path=model_path_text,
                local_path=None,
                stop_words=[],
                temperature=0.5,
                max_new_tokens=256,
                top_k=3,
                top_p=0.3,
                profiling=False
            )

def get_text_from_generator(generator):
    """Extract text from the generator response."""
    response_text = ""
    try:
        while True:
            response = next(generator)
            choices = response.get('choices', [])
            for choice in choices:
                delta = choice.get('delta', {})
                if 'content' in delta:
                    response_text += delta['content']
    except StopIteration:
        pass
    return response_text

def generate_image_metadata(image_path):
    """Generate description, folder name, and filename for an image file."""
    initialize_models()

    # Generate description
    description_prompt = "Please provide a detailed description of this image, focusing on the main subject and any important details."
    description_generator = image_inference._chat(description_prompt, image_path)
    description = get_text_from_generator(description_generator).strip()

    # Generate filename
    filename_prompt = f"""Based on the description below, generate a specific and descriptive filename (2-4 words) for the image.
Do not include any data type words like 'image', 'jpg', 'png', etc. Use only letters and connect words with underscores.

Description: {description}

Example:
Description: A photo of a sunset over the mountains.
Filename: sunset_over_mountains

Now generate the filename.

Filename:"""
    filename_response = text_inference.create_completion(filename_prompt)
    filename = filename_response['choices'][0]['text'].strip()
    filename = filename.replace('Filename:', '').strip()
    sanitized_filename = sanitize_filename(filename)

    if not sanitized_filename:
        sanitized_filename = 'untitled_image'

    # Generate folder name from description
    foldername_prompt = f"""Based on the description below, generate a general category or theme (1-2 words) for this image.
This will be used as the folder name. Do not include specific details or words from the filename.

Description: {description}

Example:
Description: A photo of a sunset over the mountains.
Category: landscapes

Now generate the category.

Category:"""
    foldername_response = text_inference.create_completion(foldername_prompt)
    foldername = foldername_response['choices'][0]['text'].strip()
    foldername = foldername.replace('Category:', '').strip()
    sanitized_foldername = sanitize_filename(foldername)

    if not sanitized_foldername:
        sanitized_foldername = 'images'

    return sanitized_foldername, sanitized_filename, description

def process_single_image(image_path):
    """Process a single image file to generate metadata."""
    try:
        foldername, filename, description = generate_image_metadata(image_path)
        return {
            'file_path': image_path,
            'foldername': foldername,
            'filename': filename,
            'description': description
        }
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def process_image_files(image_paths):
    """Process image files using multiprocessing."""
    with Pool(cpu_count()) as pool:
        data_list = list(tqdm(pool.imap(process_single_image, image_paths), total=len(image_paths), desc="Processing Images"))
    
    # Filter out None values (failed processings) and print results
    data_list = [data for data in data_list if data is not None]
    for data in data_list:
        print(f"File: {data['file_path']}")
        print(f"Description: {data['description']}")
        print(f"Folder name: {data['foldername']}")
        print(f"Generated filename: {data['filename']}")
        print("-" * 50)
    
    return data_list

def generate_text_metadata(input_text):
    """Generate description, folder name, and filename for a text document."""
    initialize_models()

    # Generate description
    description_prompt = f"""Provide a concise and accurate summary of the following text, focusing on the main ideas and key details.
Limit your summary to a maximum of 150 words.

Text: {input_text[:1000]}  # Limit input text to first 1000 characters

Summary:"""
    description_response = text_inference.create_completion(description_prompt)
    description = description_response['choices'][0]['text'].strip()

    # Generate filename
    filename_prompt = f"""Based on the summary below, generate a specific and descriptive filename (2-4 words) for the document.
Do not include any data type words like 'text', 'document', 'pdf', etc. Use only letters and connect words with underscores.

Summary: {description}

Example:
Summary: A research paper on the fundamentals of string theory.
Filename: string_theory_fundamentals

Now generate the filename.

Filename:"""
    filename_response = text_inference.create_completion(filename_prompt)
    filename = filename_response['choices'][0]['text'].strip()
    filename = filename.replace('Filename:', '').strip()
    sanitized_filename = sanitize_filename(filename)

    if not sanitized_filename:
        sanitized_filename = 'untitled_document'

    # Generate folder name from summary
    foldername_prompt = f"""Based on the summary below, generate a general category or theme (1-2 words) for this document.
This will be used as the folder name. Do not include specific details or words from the filename.

Summary: {description}

Example:
Summary: A research paper on the fundamentals of string theory.
Category: physics

Now generate the category.

Category:"""
    foldername_response = text_inference.create_completion(foldername_prompt)
    foldername = foldername_response['choices'][0]['text'].strip()
    foldername = foldername.replace('Category:', '').strip()
    sanitized_foldername = sanitize_filename(foldername)

    if not sanitized_foldername:
        sanitized_foldername = 'documents'

    return sanitized_foldername, sanitized_filename, description

def process_single_text_file(args):
    """Process a single text file to generate metadata."""
    file_path, text = args
    try:
        foldername, filename, description = generate_text_metadata(text)
        return {
            'file_path': file_path,
            'foldername': foldername,
            'filename': filename,
            'description': description
        }
    except Exception as e:
        print(f"Error processing text file {file_path}: {str(e)}")
        return None

def process_text_files(text_tuples):
    """Process text files using multiprocessing."""
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_single_text_file, text_tuples), total=len(text_tuples), desc="Processing Text Files"))
    
    # Filter out None values (failed processings) and print results
    results = [result for result in results if result is not None]
    for result in results:
        print(f"File: {result['file_path']}")
        print(f"Description: {result['description']}")
        print(f"Folder name: {result['foldername']}")
        print(f"Generated filename: {result['filename']}")
        print("-" * 50)
    
    return results

def copy_and_rename_files(data_list, new_path, renamed_files, processed_files):
    """Copy and rename files based on generated metadata."""
    for data in data_list:
        file_path = data['file_path']
        if file_path in processed_files:
            continue
        processed_files.add(file_path)

        # Use folder name which generated from the description
        dir_path = create_folder(new_path, data['foldername'])

        # Use filename which generated from the  description
        new_file_name = data['filename'] + os.path.splitext(file_path)[1]
        new_file_path = os.path.join(dir_path, new_file_name)

        # Handle duplicates
        counter = 1
        while new_file_path in renamed_files or os.path.exists(new_file_path):
            new_file_name = f"{data['filename']}_{counter}" + os.path.splitext(file_path)[1]
            new_file_path = os.path.join(dir_path, new_file_name)
            counter += 1

        shutil.copy2(file_path, new_file_path)
        renamed_files.add(new_file_path)
        print(f"Copied and renamed to: {new_file_path}")
        print("-" * 50)

def save_to_json(data, output_path, file_type):
    """Save processed data to a JSON file."""
    json_file = os.path.join(output_path, f"processed_{file_type}.json")
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)

def load_from_json(output_path, file_type):
    """Load processed data from a JSON file if it exists."""
    json_file = os.path.join(output_path, f"processed_{file_type}.json")
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            return json.load(f)
    return []

def main():
    # Paths configuration
    print("-" * 50)
    input_path = input("Enter the path of the directory you want to organize: ").strip()
    if not os.path.exists(input_path):
        print(f"Input path {input_path} does not exist. Please create it and add the necessary files.")
        return

    # Confirm successful input path
    print(f"Input path successfully uploaded: {input_path}")
    print("-" * 50)

    # Default output path is a folder named "organized_folder" in the same directory as the input path
    output_path = input("Enter the path to store organized files and folders (press Enter to use 'organized_folder' in the input directory): ").strip()
    if not output_path:
        # Get the parent directory of the input path and append 'organized_folder'
        output_path = os.path.join(os.path.dirname(input_path), 'organized_folder')

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Confirm successful output path
    print(f"Output path successfully upload: {output_path}")
    print("-" * 50)

    # Start processing files
    start_time = time.time()
    file_paths = collect_file_paths(input_path)
    end_time = time.time()

    print(f"Time taken to load file paths: {end_time - start_time:.2f} seconds")
    print("-" * 50)
    print("Directory tree before renaming:")
    display_directory_tree(input_path)

    print("*" * 50)
    print("The file upload was successful. It will take some minutes.")
    print("*" * 50)

    # Separate files by type
    image_files, text_files = separate_files_by_type(file_paths)

    # Load any existing processed data
    existing_data_images = load_from_json(output_path, "images")
    existing_data_texts = load_from_json(output_path, "texts")

    # Create sets of already processed files
    processed_image_files = set(item['file_path'] for item in existing_data_images)
    processed_text_files = set(item['file_path'] for item in existing_data_texts)

    # Filter out already processed files
    image_files = [f for f in image_files if f not in processed_image_files]
    text_files = [f for f in text_files if f not in processed_text_files]

    # Process remaining image files
    new_data_images = process_image_files(image_files)
    data_images = existing_data_images + new_data_images
    save_to_json(data_images, output_path, "images")
    print(f"Processed {len(new_data_images)} new image files. Total: {len(data_images)}. Results saved to JSON.")

    # Prepare text tuples for processing
    text_tuples = []
    for fp in text_files:
        ext = os.path.splitext(fp.lower())[1]
        if ext == '.txt':
            text_content = read_text_file(fp)
        elif ext == '.docx':
            text_content = read_docx_file(fp)
        elif ext == '.pdf':
            text_content = read_pdf_file(fp)
        else:
            print(f"Unsupported text file format: {fp}")
            continue
        text_tuples.append((fp, text_content))

    # Process text files
    new_data_texts = process_text_files(text_tuples)
    data_texts = existing_data_texts + new_data_texts
    save_to_json(data_texts, output_path, "texts")
    print(f"Processed {len(new_data_texts)} new text files. Total: {len(data_texts)}. Results saved to JSON.")

    # Prepare for copying and renaming
    renamed_files = set()
    processed_files = set()

    # Copy and rename image files
    copy_and_rename_files(data_images, output_path, renamed_files, processed_files)

    # Copy and rename text files
    copy_and_rename_files(data_texts, output_path, renamed_files, processed_files)

    print("-" * 50)
    print("The folder contents are renamed and cleaned up successfully.")
    print("-" * 50)
    print("Directory tree after copying and renaming:")
    display_directory_tree(output_path)
    print("-" * 50)
    print("The folder contents are renamed and cleaned up successfully.")
    print("-" * 50)
    print("Directory tree after copying and renaming:")
    display_directory_tree(output_path)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total processing time: {total_time:.2f} seconds")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Progress has been saved.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Progress up to this point has been saved.")
    finally:
        print("You can run the program again to continue from where it left off.")