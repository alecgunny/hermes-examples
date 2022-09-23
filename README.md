# `hermes` inference examples
This repository contains some example notebooks of how to use the [`hermes`](https://github.com/ML4GW/hermes) libraries for deploying as-a-service deep learning inference applications in service of gravitational wave physics applications.
Example notebooks can be found in the [`book`](./book) directory.
At present there is only one notebook containing real examples, `hermes-examples.ipynb`, but over time the steps in this example will be broken up into sub-notebooks and expanded upon to show how to increase the scale of your deployment.


## Reading and running the examples
The best place to read these notebooks is on this repo's [GitHub Page](https://alecgunny.github.io/hermes-examples/hermes-example.html).
If you'd like to run the notebooks yourself, ensure that you have [Poetry](https://python-poetry.org/) installed in your environment. Then start by cloning this repo and initialize the `hermes` submodule
```console
git clone --recurse-submodules git@github.com:alecgunny/hermes-examples.git
```
Then all you have to do is move to this directory and install the project and its dependencies
```console
cd hermes-examples
poetry install -E notebook
```
Once the project installs, you can launch the notebook server with
```console
poetry run jupyter notebook
```
If you're running on a remote cluster, I would recommend [port forwarding](https://help.ubuntu.com/community/SSH/OpenSSH/PortForwarding) your local port `8888` to `localhost:8888` (or whichever port is available) on the remote server, then running the notebook server with a couple simplifying arguments:
```console
poetry run jupyter notebook --no-browser --ip 0.0.0.0 --NotebookApp.token=''
```
Then you should be able to connect to the notebook server at your local `localhost:8888`.
