import pandas as pd
import os
from openai import OpenAI

# Set your GPT-4 API key here
# openai.api_key = 'OPEN_AI_API_KEY'

def read_translation(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def evaluate_translation(translation):
    prompt = f"""
    Please evaluate the following Turkish to English translation:
    {translation}

    Provide an evaluation based on the following criteria:
    1. Accuracy: How accurate is the translation compared to the original text?
    2. Fluency: How natural and fluent is the English translation?
    3. Completeness: Does the translation convey the full meaning of the original text?
    4. Grammar and Syntax: Are there any grammatical or syntactical errors?
    5. Overall Impression: General comments on the quality of the translation.

    Provide scores for each criterion from 1 to 10, where 10 is the best. Also, include detailed comments for each criterion.
    """

    # Initialize the OpenAI client
    client = OpenAI(api_key=os.environ.get("OPEN_AI_API_KEY"))

    # Create the chat completion request
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for evaluating translations."},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the evaluation content from the response
    return completion.choices[0].message.content

def save_evaluation_to_new_excel(evaluation, excel_file):
    # Create a DataFrame with the evaluation data
    df = pd.DataFrame([evaluation])

    # Save the DataFrame to a new Excel file
    df.to_excel(excel_file, index=False, engine='openpyxl')

# File paths
translation_file = 'deepseek-ai_deepseek-coder-33b-instruct.txt'
excel_file = 'evaluations.xlsx'  # Changed to a new file name

# Read the translated text
translated_text = read_translation(translation_file)

# Evaluate the translation using GPT-4
evaluation_result = evaluate_translation(translated_text)

# Prepare the evaluation data to save
evaluation_data = {
    'Translation': translated_text,
    'Evaluation': evaluation_result
}

# Save the evaluation result to a new Excel file
save_evaluation_to_new_excel(evaluation_data, excel_file)

print("Evaluation completed and saved to a new Excel file.")
