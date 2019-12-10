echo "If you already ran this script, you don't need to do so again"
echo "Start the environment by running:  $ pipenv run jupyter-notebook"
echo "Or start a sandboxed session with: $ pipenv shell"

# setup python 3.8 env
command -v pipenv &>/dev/null         || exit 1
pipenv --venv || pipenv --python 3.8  || exit 1

# math, statistics, data science
pipenv install "Cython"               || exit 1
pipenv install "numpy"                || exit 1
pipenv install "scipy"                || exit 1
pipenv install "pandas"               || exit 1
pipenv install "scikit-learn"         || exit 1

# notebook, plots
pipenv install "unidecode"            || exit 1
pipenv install "matplotlib"           || exit 1
pipenv install "colorspacious"        || exit 1
pipenv install "jupyter"              || exit 1
pipenv install "notebook"             || exit 1
pipenv install "ipywidgets"           || exit 1
pipenv install "plotly"               || exit 1

# pomegranate for hidden markov models
pipenv install "joblib"               || exit 1
pipenv install "pomegranate"          || exit 1

# jupyter themes
pipenv install "jupyterthemes" && pipenv run \
jt -t monokai -fs 10 -tf sourcesans -tfs 11 -nf source -nfs 12

pipenv run jupyter-notebook

# reset theme
command -v pipenv &>/dev/null  && jt -r

exit
