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
