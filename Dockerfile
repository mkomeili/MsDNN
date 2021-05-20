# Arguments to select python version, cuda version etc.
# These were added here using a shell script that generated this file. Changing this file is not recommended.
ARG base_image=tensorflow/tensorflow:latest-gpu-py3

FROM ${base_image}

ARG python_version=3.7.2
# DON'T CHANGE THIS ARG
ARG username=python_user

# Add this to prevent prompts during build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update

# install pyenv pre-requisites
RUN apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev python-openssl git graphviz

RUN useradd -m ${username}
USER ${username}

# get pyenv
RUN git clone https://github.com/pyenv/pyenv.git /home/${username}/.pyenv

# Set up environment variables for pyenv
ENV HOME /home/${username}
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# Install specific python version
RUN pyenv install ${python_version}
RUN pyenv global ${python_version}
RUN pyenv rehash

# Upgrade pip
RUN pip install -U pip

# Check if system python was updated properly
# ADD ./src/info.py .
# RUN python info.py
# CMD ["python", "src/info.py"]

# Install jupyter and jupyter lab
RUN pip install jupyter --use-feature=2020-resolver
RUN pip install jupyterlab --use-feature=2020-resolver

# Set up port for jupyterlab
EXPOSE 8888

# change working directory inside container
WORKDIR /home/${username}/app/

# Project specific stuff
COPY requirements.txt .
RUN pip install -U pip
RUN pip install autopep8 --use-feature=2020-resolver
RUN pip install sklearn --use-feature=2020-resolver
RUN pip install tensorflow-gpu --use-feature=2020-resolver
RUN pip install pandas --use-feature=2020-resolver
RUN pip install matplotlib --use-feature=2020-resolver
RUN pip install --upgrade prompt_toolkit --use-feature=2020-resolver
RUN pip install google-cloud-storage --use-feature=2020-resolver
RUN pip install google-auth --use-feature=2020-resolver
RUN pip install google-api-python-client --use-feature=2020-resolver
RUN pip install google-cloud-resource-manager --use-feature=2020-resolver
RUN pip install bs4 --use-feature=2020-resolver
RUN pip install pydot --use-feature=2020-resolver
RUN pip install graphviz --use-feature=2020-resolver

RUN jupyter nbextension enable scratchpad/main
RUN jupyter nbextension enable toc2/main
RUN jupyter nbextension enable varInspector/main
RUN jupyter nbextension enable code_prettify/autopep8
RUN jupyter nbextension enable splitcell/splitcell
RUN jupyter nbextension enable spellchecker/main
RUN jupyter nbextension enable move_selected_cells/main

RUN pip install -r requirements.txt --use-feature=2020-resolver

USER root
# Run jupyter notebook. Customize the external ip here
CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--port", "8888", "--no-browser", "--allow-root"]
# CMD [ "jupyter", "notebook", "--no-browser"]
