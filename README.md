# Keras 3 + MLflow — an MLOps learning project

**Part 1** (`part1/notebooks/hub.ipynb`) builds a **train → compare → promote → serve → observe**
loop on MNIST with Keras 3, MLflow as tracking server and registry, and a FastAPI server that finds
its model through the `@champion` alias alone.
**Part 2** (`part2/notebooks/hub.ipynb`) moves to credit-card fraud and closes the three loops Part 1
cannot teach — an application that keeps answering when the endpoint times out, a human review queue
that buys the labels production never gives you, and a DVC-hashed dataset that makes a retrained
model traceable to exact bytes — then wraps the lot in Airflow.

Start at `slides/`, then run the notebooks in order.
**Before you clone: install Docker** — Engine with the Compose v2 plugin, running as your own user.
Then `pip install -r requirements-host.txt` into a virtualenv and open the Part 1 notebook at its
first cell.
