"""ScienceWorld integration for the text-protocol evolver (SCIWORLD_SPEC.md).

Unlike TextCraft (in-process) and ALFWorld/WebShop (their own env servers), the
ScienceWorld environment runs in a separate py3.8 process tree
(``benchmarks/sciworld/env_server.py``) because ``scienceworld==1.2.3`` needs
Python 3.8 + a JVM via py4j.  The thin client wrapper here (``env.py``) mirrors
the ``reset(task_id) -> (obs, info)`` / ``step(action) -> (obs, reward, done,
info)`` contract the runner expects.
"""
