import os


def combine_core_files_content(directory):
    combined_content = ""
    core_files = [
        'data_processing.py',
        'file_utils.py',
        'main.py',
        'README.md',
        'output_filter.py'
    ]

    for file_path in core_files:
        full_path = os.path.join(directory, file_path)
        if os.path.exists(full_path):
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    combined_content += f"File: {file_path}\n\n"
                    combined_content += f.read()
                    combined_content += "\n\n" + "=" * 50 + "\n\n"
                print(f"Processed: {file_path}")
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")
        else:
            print(f"File not found: {file_path}")

    return combined_content


def main():
    # ============================
    # === Hard-Coded Introduction
    # ============================
    INTRO_TEXT = """\
Role: You are an expert programming tutor assisting a self-taught coder with basic programming knowledge.

Task: Provide clear, comprehensive coding assistance tailored to my level of understanding.

Context: I have a general grasp of programming concepts but lack deep language-specific knowledge. I need extra help with big-picture understanding and detailed explanations.

Instructions:
1. Provide exact, complete code segments in clearly marked code blocks that can be directly copied and pasted.
2. Clearly indicate where in the existing code these segments should be placed.
3. Break down complex tasks into smaller, manageable steps.
4. Include detailed, step-by-step explanations for each code segment, assuming I may need clarification on fundamental concepts.
5. Anticipate common errors and provide preventive advice, explaining why these errors occur and how to avoid them.
6. Include comments within the code to explain the purpose and functionality of each section.

Before responding:
- Think carefully about the additions or changes you are suggesting to the code.
- Review the overall codebase and ensure there will be no unintended consequences from the changes.
- Consider potential edge cases or limitations in the proposed solution.

Output Format:
1. An overall educational assessment of the issue or question, followed by complete code file(s) sequentially as needed to address the issue.

User Message:

Hello,

hi, im a self taught coder with LLM assistance. here's my codebase for my project.
"""

    # ============================
    # ===== Hard-Coded Question
    # ============================
    QUESTION_TEXT = """
 
here's my project logs. my true goal is to have LLMs descrie and categorize my images metadata and envision and (propose shell scripts to) sort them for me based on that. thanks.


 """

    # Specify the directory path (current directory in this case)
    directory_path = os.path.dirname(os.path.abspath(__file__))

    # Get the combined content
    combined_content = combine_core_files_content(directory_path)

    # Structure the final output
    final_output = f"Introduction:\n{INTRO_TEXT}\n\n"
    final_output += "Combined Core Files Content:\n\n"
    final_output += combined_content
    final_output += """\n


    \nQuestion:\n"""
    final_output += QUESTION_TEXT

    # Save to a file
    output_file_name = "combined_core_content_with_intro_and_question.txt"
    with open(output_file_name, "w", encoding="utf-8") as output_file:
        output_file.write(final_output)

    print(
        f"\nThe combined content with your introduction and question has been saved to {output_file_name}"
    )


if __name__ == "__main__":
    main()
