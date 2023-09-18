# langchain-streamlit-app

Web applications using [Streamlit](https://streamlit.io/) and [LangChain](https://www.langchain.com/). The contents of this repository are based on the files in [naotaka1128/ai_app_book](https://github.com/naotaka1128/ai_app_book).

# LICENSE

* tests/cheat_sheet_examples.md: please see [KDnuggets: Terms of Service](https://www.kdnuggets.com/terms-of-service.html).
* The other codes: Apache License, Version 2.0.

# Requirements

* Python >= 3.8.1
* modules: please see `requirements.txt`.

If you want to know specific moduls required by each script, please see `Requirements` shown in the doc_string on the top of each script.

# Installation

You can clone/download this repository and test the scripts.

## One method

1. clone this repository and move to the cloned directory:   
    ```sh
    git clone https://github.com/Surpris/langchain-streamlit-app.git ./langchain-streamlit-app && cd ./langchain-streamlit-app
    ```
2. install necessary modules for each script:   
    ```sh
    pip -r ./requirements.txt
    ```
3. run the target script:   
    ```python
    streamlit run ./chapters/XX.py
    ```

## Remarks on using the scripts in this repository

You have to set your OpenAI key somewhere, e.g., your envronmental PATH.

For UNIX:

```sh
export OPENAI_API_KEY=<your-openai-key>
```

For Windows:

```sh
set OPENAI_API_KEY=<your-openai-key>
```

Another way is to set your key to `openai.api_key`:

```python
import openai
openai.api_key = '<your-openai-key>'
```

## Configuration of streamlit

You can set the configuration of streamlit in `.streamlit/config.toml`. Please check [the Official page](https://docs.streamlit.io/library/advanced-features/configuration) for more details.

# References

* [Build and Learn: AI App Development for Beginners - Unleashing ChatGPT API with LangChain & Streamlit](https://zenn.dev/ml_bear/books/d1f060a3f166a5)
* [GitHub: naotaka1128/ai_app_book](https://github.com/naotaka1128/ai_app_book)
* [Configuration - Streamlit Docs](https://docs.streamlit.io/library/advanced-features/configuration)
