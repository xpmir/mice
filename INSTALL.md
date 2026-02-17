# Installation steps 

To get started with the project requirements and uv, you need to follow the steps below:
1. Make sure that your Java version is at least 21. You can check it by running `java -version`. If it's okay, then you can move to the step 3. directly.

2. To get java version 21, you need to download the corresponding [openjdk version](https://openjdk.org/install/) and unzip it under your root directory for example. Then, to make sure your system targets it, you need to modify your `.bashrc` file (located under your root) with the following two lines:
```
export JAVA_HOME=~/jdk-21
export PATH=$JAVA_HOME/bin:$PATH
```

Once it's changed, run `source .bashrc` in your terminal and double-check that the changes are effective by running again `java -version` and `which java` to make sure your system knows to use the newly installed jdk and has changed the version accordignly.

3. Once Java is setup, you can setup the virtual environment with uv. Make sure you build a virtual env based on Python 3.10 at least (or run `uv python install 3.10` if you don't already have this version of Python) by running `uv venv --python 3.10.xx`.

4. Install the project requirements by running `uv pip install pyproject.toml`.

5. Validate that everything is correctly setup by running `uv run experimaestro run-experiment --workdir /path/to/workdir/ --file /path/to/franken_minilm/src/ir_training/experiment.py /path/to/franken_minilm/src/ir_training/bm25_minilm_baseline.yaml --run-mode DRY_RUN`.