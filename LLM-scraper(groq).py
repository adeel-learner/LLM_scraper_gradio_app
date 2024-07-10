import os
import csv
import json
import gradio as gr
from groq import Groq
from langchain_community.document_loaders import WebBaseLoader

# Specify the CSV file path
csv_file_path = 'output.csv'

def get_api_key():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    return api_key

def process_url(url, model_name):
    loader = WebBaseLoader(url)
    data = loader.load()[0].page_content
    text_content = ' '.join(data.split())

    prompt = "Just extract four things i.e title, price, brand, and retailer from the following context in JSON format."
    script_content = f"{prompt}:\n ```{text_content}```"

    client = Groq(api_key=get_api_key())
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful data format assistant that gives outputs only in JSON."},
            {"role": "user", "content": script_content},
        ],
        model=model_name,
        temperature=0,
        response_format={"type": "json_object"}
    )

    output_response = chat_completion.choices[0].message.content
    json_data = json.loads(output_response)
    retailer = json_data.get('retailer', '')

    # Pretty-print the JSON data
    pretty_output_response = json.dumps(json_data, indent=4)

    return text_content, pretty_output_response, retailer

def save_results_to_csv(text_content, output_response, retailer):
    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
        csv_headers = ['input', 'output', 'retailer']
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        if not file_exists or os.stat(csv_file_path).st_size == 0:
            writer.writeheader()
        writer.writerow({'input': text_content, 'output': output_response, 'retailer': retailer})

def load_and_display_webpage_content(url, file, model_name, save_to_csv_flag):
    try:
        if url:
            text_content, output_response, retailer = process_url(url, model_name)
            if save_to_csv_flag:
                save_results_to_csv(text_content, output_response, retailer)
            return text_content, output_response

        elif file:
            with open(file.name, mode='r', encoding='ISO-8859-1') as file:
                csv_reader = csv.DictReader(file)
                all_results = ""
                count = 0
                for index, row in enumerate(csv_reader):
                    if index < 1:   # 0-based index
                        continue
                    url = row['link-href']
                    text_content, output_response, retailer = process_url(url, model_name)
                    count += 1
                    if save_to_csv_flag:
                        save_results_to_csv(text_content, output_response, retailer)
                    all_results += f"URL: {url}\nContent: {text_content}\nOutput: {output_response}\n\n"
                    counting = f"Total URLs processed: {count}"
                return counting, all_results
    except Exception as e:
        return str(e), ""

# Create Gradio interface
url_input = gr.Textbox(label="Enter URL", type="text")
file_upload = gr.File(label="Upload CSV File", file_types=[".csv"])
dropdown = gr.Dropdown(
    choices=["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it", "gemma2-9b-it"],
    label="Select Model",
    value="gemma2-9b-it"
)
save_to_csv_checkbox = gr.Checkbox(label="Save response to CSV")
output_text = gr.Textbox(label="Webpage Content", show_copy_button=True)
final_output = gr.Textbox(label="Final Output", show_copy_button=True)

gr.Interface(
    fn=load_and_display_webpage_content,
    inputs=[url_input, file_upload, dropdown, save_to_csv_checkbox],
    outputs=[output_text, final_output],
    title="Webpage Content Viewer",
).launch()
