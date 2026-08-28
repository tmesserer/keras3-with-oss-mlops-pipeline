# Keras 3 + MLflow — an MLOps learning project

A hands-on **train → compare → promote → serve → observe** loop that runs entirely on one laptop
under Docker Compose, taught as two notebooks: **Part 1** (`part1/notebooks/hub.ipynb`) builds the
minimal loop on MNIST with Keras 3 on the torch backend, MLflow as tracking server and registry, and
a FastAPI server that finds its model through the `@champion` alias alone; **Part 2**
(`part2/notebooks/hub.ipynb`) moves to credit-card fraud and closes the three loops Part 1 cannot
teach — an application that must keep answering when the endpoint times out, a human review queue
that buys the labels production never gives you, and a DVC-hashed derived dataset that makes a
retrained model traceable to exact bytes — then wraps the lot in Airflow. Both parts share one
MLflow registry and one Postgres, and everything you need to know is in the notebooks: read them in
order, run the cells, and don't skip the manual steps. **Before you clone: install Docker or
Podman.** You need Docker Engine with the Compose v2 plugin running as your own user, or rootless
Podman with its user socket switched on (`systemctl --user enable --now podman.socket`) — the
notebooks detect which engine you have and configure themselves for it. It is the one thing this
project does not install for you. Linux, macOS (Docker Desktop or Podman Desktop) and Windows
through WSL2 all work; on WSL2, keep the clone inside the Linux filesystem, not under `/mnt/c`. To
set up: create a virtualenv and `pip install -r requirements-host.txt` into it (clients only — Keras
and torch live exclusively inside the containers), then open the Part 1 notebook and start at its
first cell — that cell writes `.env` from `.env.example` with the ids and the engine settings this
machine needs, and leaves an existing `.env` alone.
