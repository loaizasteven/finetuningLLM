#!/bin/bash    

# Run bash functions from terminal with
# . ./agent_build_script.sh && func

# Repo finetuningLLM
# env
DIRECTORY="finetune_env"
source $HOME/git/finetuningLLM/$DIRECTORY/bin/activate
function activate () {
  source $HOME/git/finetuningLLM/$DIRECTORY/bin/activate

  # logging info
  basename $VIRTUAL_ENV
  # pip installation
  pip install --upgrade pip
  echo "pip installation in quiet mode..."
  pip install -q -r requirements.txt --upgrade-strategy only-if-needed
}

function unittest() {
  pytest --doctest-modules >&1
}

if [ -d "$DIRECTORY" ]; then
    unittest
else
    echo "Creating venv"
    /usr/local/bin/python3 -m venv "$DIRECTORY"

    activate
fi