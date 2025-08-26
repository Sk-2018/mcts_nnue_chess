# MCST + Stockfish NNUE UCI chess engine
Python implementation of MCTS + NNUE chess engine (very basic)

# LIVE CODING STREAMS
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/6ZxRgwwBkgU/0.jpg)](https://www.youtube.com/watch?v=6ZxRgwwBkgU&list=PLmN0neTso3Jx8FQSmq9Iab7q7zILYgKjt)

## Self Learning

A minimal self-play training loop is available in `src/self_learning.py`. It
plays games against itself using MCTS and stores simple centipawn evaluations
for encountered positions. Run it with:

```bash
python src/self_learning.py --games 10 --iterations 50
```

Learned values are persisted to `data/selfplay.json` and will be used to
override NNUE evaluations on subsequent runs.

## NNUE Library

The engine relies on a small C++ helper library for probing Stockfish NNUE
networks. A prebuilt Linux version (`libnnueprobe.so`) is provided in `src/`.
When running on other platforms the code now looks for `libnnueprobe.dll` on
Windows or `libnnueprobe.dylib` on macOS. Ensure the appropriate library for
your platform is available in the `src/` directory or update `src/mcts.py` to
point to a compatible build. On Windows the file should be named
`libnnueprobe.dll` and reside alongside the Python sources so it can be loaded
via `ctypes.WinDLL`. Make sure to use a 64-bit build if you are running a
64-bit version of Python.

## AlphaZero-style search

`src/alpha_zero.py` contains a minimal AlphaZero-inspired MCTS. It expects a
network that returns a policy over legal moves and a value in [-1, 1] for a
given position. A tiny trainable network using PyTorch now provides the value
head while keeping a uniform policy for simplicity:

```bash
python - <<'PY'
from state import State
from alpha_zero import AlphaZeroMCTS, TinyNetwork

net = TinyNetwork()
searcher = AlphaZeroMCTS(net, n_simulations=10)
move = searcher.search(State())
print(move)
PY
```

This demonstrates the PUCT-based MCTS loop.

### Reinforcement learning

`AlphaZeroTrainer` implements a very small self-play reinforcement learning
loop that updates the value network from game outcomes. Running the module as a
script performs one training step and prints the training loss:

```bash
python src/alpha_zero.py
```

This is intentionally lightweight and meant for experimentation rather than
playing strength.
