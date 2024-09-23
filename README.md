# Arxiv2Arxode

> The aim of this project is to automate code generation based off of information internalized from artificial intelligence and machine learning research papers. In hopes of exploring the capabilities of current state-of-the-art models and how to pose them as task solving agents.

----------
## Known Issues
- Failure to stop and remove docker containers after execution, you will find that there will be a lot (emphasis on a lot) of "zombie" containers that are running in the background eating up your memory. This ***can*** be fixed by running `docker stop $(docker ps -a -q)` and `docker rm $(docker ps -a -q)`. This does not work too well if you have other containers running that you want to keep, however, I must provide a solution to my contrived problem 💀.

- If using the GPT3 family of models one can run into token limit issues, there is a hacky fix under the retrieval agent's `_get_context` function, it is commented out when using GPT4.
## Installation

#### Prerequisites:
- [Anaconda](https://docs.anaconda.com/free/anaconda/install/) is preferred for environment and dependency management.
- Having [Docker](https://docs.docker.com/get-docker/) installed is also required for running the vector database.
- Optionally have jupyter notebook and the IPython kernel installed to run the [demo.ipynb](./demo.ipynb) file for interactive use.
    - Within the below created conda env run `conda install anaconda::ipykernel anaconda::jupyter -y`

----
1. Create a new conda environment with python 3.11 and activate it:</br>
`conda create -n testenv python=3.11 && conda activate testenv`
2. Install poetry within the newly created conda environment:</br>
`conda install poetry -y`
3. From the root of the project, run `poetry install` to install project dependencies.
4. Finally while still in the root of the directory run `pip install -e .` to install `arxiv2arxode` as an editable package.

#### Environment Configuration

Before running the project, create a `.env` file at the root of the project directory with the following content to configure your environment variables:

>Please note that these are the default values that are set up in the [`docker-compose.yml`](./docker-compose.yml) file. **For testing purposes and ease of use, it is recommended to use the same values. However, you can change the values to suit your environment.**

```plaintext
# Default sample env file...
OPENAI_API_KEY=<your_openai_api_key>
ANTHROPIC_API_KEY=<your_anthropic_api_key>
PGVECTOR_DRIVER=psycopg2
PGV_PORT=5432
PGV_USER=pgvector_user
PGV_PASSWORD=pgvector_password
PGV_DATABASE=pgvector_database
```
4. Create the necessary `.env` file at the root of the project directory.
5. Run the following command to start the pgvector database:</br>
`docker-compose up -d`
6. You can now view and run the [demo.py](./demo.py) via the following command: `./run.sh demo.py`. Alternatively, if you have the necessary environment for running Jupyter notebooks take a look at the [demo.ipynb](./notebooks/demo.ipynb) file.


<img src="./imgs/DALL·E 2023-12-16 14.50.png" > </img>

According to DALL-E, the word Arxiv2Arxode is represented by the image above.
