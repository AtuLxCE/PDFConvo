# PDFConvo

Ask questions across multiple PDF documents. This Streamlit prototype retrieves relevant text with FAISS, passes it to a language model with conversation history, and reads answers aloud using text-to-speech.

[Watch the demo](https://youtu.be/0QLXdarbhqw) · [Portfolio](https://atulshreewastav.com.np/#projects)

## How it works

1. Upload PDFs in the sidebar.
2. Extract text with PyPDF2 and split it into overlapping chunks.
3. Embed the chunks and build an in-memory FAISS index.
4. Retrieve relevant chunks for each question and generate a conversational answer.
5. Play the answer as audio using gTTS.

**Stack:** Python, Streamlit, LangChain, FAISS, Google PaLM, PyPDF2, gTTS.

## Project status

This is an older prototype using LangChain 0.0.348 and legacy Google PaLM integrations. The provider integration needs updating before a current end-to-end run can be expected. The demo shows the original application.

## Local development

After updating the provider integration and checking its required credentials:

```bash
python -m venv .venv
# Activate the virtual environment for your shell.
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

`app.py` loads environment variables from a local `.env` file. Keep credentials out of source control. Upload your PDFs before submitting a question.

## Repository guide

- `app.py`: PDF processing, retrieval, chat interface, and speech output.
- `htmlTemplates.py`: Chat styles and message templates.
- `requirements.txt`: Original pinned dependencies.
- `sample.pdf`: Example document.

## Limits

PDF extraction expects embedded text; scanned documents require an OCR step. Retrieved context can improve relevance but does not guarantee a correct answer. Document text is sent to the configured model and embedding provider; answer text is sent to gTTS for audio generation.

## License

See [LICENSE](LICENSE).
