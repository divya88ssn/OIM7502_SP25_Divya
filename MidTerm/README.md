Overview:

This project demonstrates prompt engineering using Gradio and the Hugging Face Inference API. 
It allows users to experiment with prompt styles, adjust generation parameters, and observe how prompt engineering influences large language model (LLM) responses in real-time.

Libraries Used:

✅ Gradio
Gradio is a Python library for building interactive web-based demos for machine learning models and data science applications. 
It provides easy-to-use components like chatbots, sliders, and dropdowns, with the ability to share apps via public links.
https://www.gradio.app/docs


✅ Hugging Face Hub (huggingface_hub)
huggingface_hub enables seamless interaction with the Hugging Face Model Hub. 
It provides tools for downloading models, datasets, and leveraging Hugging Face’s hosted inference APIs to generate outputs without requiring local model deployment.
https://huggingface.co/docs


✅ Hugging Face Inference Client
InferenceClient lets you easily query models hosted on Hugging Face’s Inference API from Python. 
It supports tasks like text generation, summarization, and more without requiring local deployment.
https://huggingface.co/docs/api-inference/index

Requirements:
Python Version: 3.8 or higher
Install the Required Libraries:
pip install gradio==4.29.0 huggingface_hub==0.23.2


How This Demo App Works:
The app connects to Hugging Face's Inference API using a hardcoded token (no additional setup required).
Users can select a model (GPT-2 Small or GPT-2 Medium) from a dropdown.
Users type a message into the chatbot input box.

They can adjust generation parameters:
temperature: randomness in generation
top_p: nucleus sampling
max_tokens: response length limit

The app maintains a conversation history (context window) and returns responses based on the selected model and parameters.
Runs locally and can be shared remotely via Gradio’s public link feature.


Running the Demo:
Open the jupyter notebook submitted in PyCharm or Jupyter Notebook.
Execute the script (or run all cells if using a notebook).
Launch the Gradio app with:
demo.launch(share=True)


Accessing the Gradio Demo: https://www.gradio.app/guides/quickstart#sharing-your-demo
When you run the app, Gradio will provide:
A local URL such as:
http://127.0.0.1:7860
A public shareable link (if share=True is enabled), such as:
https://xxxxxx.gradio.live

Gradio Features Used in This Demo
1. Blocks API
gr.Blocks() enables building flexible, multi-component UIs.

Allows arranging elements (dropdowns, sliders, chatbot) with clean layout control.

2. Chatbot Component
gr.Chatbot() displays conversation history in a chat bubble format.

Supports stateful conversations by maintaining history using gr.State().

3. Dropdown Menu
gr.Dropdown() allows users to select between different models dynamically.

4. Textbox Input
gr.Textbox() collects user messages.

The .submit() method triggers the model inference when the user presses Enter.

5. Sliders
gr.Slider() lets users control generation parameters like:

max_tokens: limits response length.

temperature: controls randomness.

top_p: controls nucleus sampling.

6. Buttons
gr.Button() clears the chat history and resets the state.

7. State Management
gr.State() keeps track of chat history between user inputs and responses.

Maintains context awareness by passing the full history to the model.

8. Launch & Sharing
demo.launch(share=True) runs the app locally and generates a publicly accessible URL, enabling easy sharing for demos and feedback collection.


