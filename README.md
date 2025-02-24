# Granite IO Processing

## Introduction

Granite IO Processing is a framework which enables you to transform how a user calls or infers an IBM Granite model and how the output from the model is returned to the user. In other words, the framework allows you to extend the functionality of calling the model.

## Getting Started

### Requirements

1. **🍎 Apple M1/M2/M3 Mac or 🐧 Linux system**
2. Python 3.10 to Python 3.11

### Installation

We recommend using a Python virtual environment with Python 3.10+. Here is how to setup a virtual environment using [Python venv](https://docs.python.org/3/library/venv.html):

```
python3 -m venv granite_io_venv
source granite_io_venv/bin/activate
```

> [!TIP]
> If you use [pyenv](https://github.com/pyenv/pyenv), [Conda Miniforge](https://github.com/conda-forge/miniforge) or other such tools for Python version management, create the virtual environment with that tool instead of venv. Otherwise, you may have issues with installed packages not being found as they are linked to your Python version management tool and not `venv`.

You can install the Granite IO Processor as follows:

#### From Source

To install from source(GitHub Repository):

```shell
python3 -m venv granite_io_venv
source granite_io_venv/bin/activate
git clone https://github.com/ibm-granite/granite-io
cd granite-io
pip install -e .
```

### Try It Out!

To help you get up and running as quickly as possible with the Granite IO Processing framework, check out the following resources which demonstrate how to use the framework:

- Jupyter notebook tutorials:
  - [IO](./notebooks/io.ipynb)

## Contributing

Check out our [contributing guide](CONTRIBUTING.md) to learn how to contribute.
