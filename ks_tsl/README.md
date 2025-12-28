This folder archives some files and a few steps during kserve building process.

You'll need to have kserve(v0.14.0) and other dependencies already installed in  your system.

Note that *minikube* is more preferred than *kind*, since the former provides better CUDA GPU support.

## model loading
* models-pvc.yaml: prepare for pvc format, which will be used later.
* model-loader.yaml: load qwen, mistral, and sign_vectors.pkl in pvc.

## tsl-semantic-service
* app.py: a kserve version of our semantic search main program.
* dockerfile: main program is expected to be wrapped up in docker.
* requirements.txt: used when building dockerfile.

## others
* models_list.txt: specifies the version of llm models we used.
* semantic-isvc.yaml: the main inference service which functions the program and loading prepared models.
* sign_vectors.pkl: same as those seen in other directories

## environments
the virtual environment used during building. (in uv)

note that this is for reference only.

usage: `uv sync`
