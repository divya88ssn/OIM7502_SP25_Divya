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
This demo uses Gradio to create an interactive chatbot for LLM exploration. It showcases the following core features:

✅ 1. Blocks API for Custom Layouts
Gradio’s Blocks API allows flexible arrangement of components to create a custom UI workflow.
In this demo:

A dropdown menu lets you switch between different models (GPT-2 Small, GPT-2 Medium).

Sliders let users control generation parameters (temperature, top_p, max_tokens), affecting how the LLM responds.

A chatbot window displays multi-turn conversations with the selected model.

The Blocks API gives full control over layout and event handling, ideal for custom LLM tools and prompt engineering playgrounds.

✅ 2. Interactive LLM Chatbot Demo
This chatbot demonstrates how you can:

Send user prompts and view model-generated responses.

Maintain conversation context using gr.State() for multi-turn interaction with the model.

Experiment with different prompts and models, useful for prompt engineering and model testing.

✅ 3. Easy Sharing & Deployment
Run locally on your laptop (localhost:7860), useful for personal testing.

Generate a public link instantly by launching with share=True.
This makes model sharing simple, enabling feedback collection or demoing your chatbot without deploying to a server.
