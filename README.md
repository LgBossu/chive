> **Source availability:** This repository is publicly available for portfolio and code-review purposes. It is not open-source software, and no general permission is granted to reuse, modify, redistribute, or deploy its contents. See [LICENSE](LICENSE.txt).


# CHIVE

> Chat History Indexing & Visualization Environment

This project is designed to handle post-processing tasks for ChatGPT outputs. Below, you'll find the necessary information to set up and run the project.
Insofar, this is a private prototype, justifying my rather loose structuring of the project and approximate documentation.

*Note that this project is to be considered an early prototype. Versioning can be tracked in the `pyproject.toml`, with a now unified versioning scheme. We are currently at version `alpha-3.∞`, preparing migration to versions `alpha-4`.*

## Directory Structure

The project roughly follows the directory structure below:

```
/main/
├── README.md          # Project documentation
├── env/               # Environment variables dir
├── .venv/             # Virtual environment (created using Poetry)
├── code/              # Source code for the project
├── data/              # Directory for input/output data files
└── logs/              # Directory to store logging files
```

## Setup Instructions

1. **Clone the Repository**  
    Clone the repository to your local machine.

2. **Install Dependencies**  
    This project uses [Poetry](https://python-poetry.org/) for dependency management. Ensure Poetry is installed on your system, then run:
    ```bash
    poetry install
    ```

3. **Set Up Environment Variables**  
    Create a `path.env` file in the root directory to define necessary environment variables. Example:
    ```
    DATA_DIR=~/chatgpt-postprocess/data
    ```

4. **Activate the Virtual Environment**  
    Activate the Poetry-managed virtual environment:
    ```bash
    poetry shell
    ```

5. **Run the Project**  
    Execute the main script or desired functionality from the `code/` directory. Probably.


## Notes

- Ensure the `path.env` file is not included in version control for security reasons.
- The `data/` directory is used for input/output files and should be structured as required by the project.
