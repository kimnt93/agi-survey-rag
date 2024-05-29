# RAG Chatbot Basic

This project demonstrates how to build a basic Retrieval-Augmented Generation (RAG) chatbot. It uses [LlamaParse](https://cloud.llamaindex.ai/) for parsing PDF and integrates with the Groq API for low latency inference. The demo PDF document is from the ICLR 2024 Workshop on ["How Far Are We From AGI"](https://github.com/ulab-uiuc/AGI-survey).

## Features

- **PDF Parsing**: Utilizes [LlamaParse](https://cloud.llamaindex.ai/) to parse PDF documents. Access the API at [LlamaCloud](https://cloud.llamaindex.ai/api-key).
- **Groq API Integration**: Leverages the Groq API for fast and efficient inference. Access the API at [Groq API](https://console.groq.com/keys).

## Requirements

To set up and run the project, you need the following dependencies and environment variables.

### Dependencies

Install the required Python packages by running:

```bash
pip install -r requirements.txt
```

### Environment Variables

Make sure to set up the following environment variables in your shell or in an `.env` file:

```bash
export LLAMA_CLOUD_API_KEY=<YOUR_LLAMA_CLOUD_API_KEY>
export GROQ_API_KEY=<YOUR_GROQ_API_KEY>
```

Replace `<YOUR_LLAMA_CLOUD_API_KEY>` and `<YOUR_GROQ_API_KEY>` with your actual API keys.

## Usage

After setting up the environment variables and installing the dependencies, you can run the chatbot using Chainlit:

```bash
chainlit run main.py
```

