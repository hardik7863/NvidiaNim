# Nvidia NIM 🚀

This project is a Streamlit application that uses NVIDIA AI endpoints to analyze and answer questions based on US Census documents.

## Project Structure 📁

```
.env
.gitignore
app1.py
requirements.txt
testapp.py
us_census/
    acsbr-015.pdf
    acsbr-016.pdf
    acsbr-017.pdf
    p70-178.pdf
```

## Setup 🛠️

1. **Clone the repository:**
    ```sh
    git clone https://github.com/hardik7863/NvidiaNim.git
    cd <repository-directory>
    ```

2. **Create a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up environment variables:**
    - Create a `.env` file in the root directory with the following content:
        ```env
        NVIDIA_API_KEY="your-nvidia-api-key"
        ```

## Running the Application ▶️

To run the Streamlit application, use the following command:
```sh
streamlit run app1.py
```

## Files 📄

- **app1.py**: The main Streamlit application file.
- **testapp.py**: A script to test the NVIDIA API integration.
- **requirements.txt**: A list of dependencies required for the project.
- **us_census/**: A directory containing US Census PDF documents.

## Usage 💡

1. Open the Streamlit application in your browser.
2. Click on the "Documents Embedding" button to create the vector store database.
3. Enter your question in the text input field and get answers based on the provided context from the US Census documents.

## Deployment 🌐

You can deploy this application using Streamlit Sharing, Heroku, or any other platform that supports Python web applications. For example, to deploy on Streamlit Sharing:

1. Push your code to a GitHub repository.
2. Go to [Streamlit Sharing](https://share.streamlit.io/).
3. Connect your GitHub repository and deploy the application.

**Deployment Link:** [Nvidia NIM on Streamlit](https://hardik7863-nvidianim-app1-2aih8a.streamlit.app/)

## License 📜

This project is licensed under the MIT License.