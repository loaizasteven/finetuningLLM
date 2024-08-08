#!/usr/bin/python    

# Repo AI-Agent-in-LangGraph
# env
DIRECTORY="finetune_env"

function activate () {
  source $HOME/git/finetuningLLM/$DIRECTORY/bin/activate
}

if [ -d "$DIRECTORY" ]; then
    activate
else
    echo "Creating venv"
    /usr/local/bin/python3 -m venv "$DIRECTORY"
fi