# parallel-trpo

## Acknowledge
[kvfrans/parallel-trpo](https://github.com/kvfrans/parallel-trpo)

[ilyasu123/trpo](https://github.com/ilyasu123/trpo)

[wojzaremba/trpo](https://github.com/wojzaremba/trpo)

## Citation
```
@article{DBLP:journals/corr/SchulmanLMJA15,
  author    = {John Schulman and
               Sergey Levine and
               Philipp Moritz and
               Michael I. Jordan and
               Pieter Abbeel},
  title     = {Trust Region Policy Optimization},
  journal   = {CoRR},
  volume    = {abs/1502.05477},
  year      = {2015},
  url       = {http://arxiv.org/abs/1502.05477},
  timestamp = {Wed, 07 Jun 2017 14:42:34 +0200},
  biburl    = {http://dblp.uni-trier.de/rec/bib/journals/corr/SchulmanLMJA15},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
```
## Contributions
- Update for TensorFlow 1.3
- Fix some bugs. In file `main.py`, Add
```
history["maxkl"] = []
```

## About

A parallel implementation of Trust Region Policy Optimization (TRPO) on environments from OpenAI Gym.

Now includes hyperparaemter adaptation as well! For more info, check [Kevin Frans' post on this project](http://kvfrans.com/speeding-up-trpo-through-parallelization-and-parameter-adaptation/).

Kevin Frans is working towards the ideas at [this openAI research request](https://openai.com/requests-for-research/#parallel-trpo).
The code is based off of [this implementation](https://github.com/ilyasu123/trpo).

Kevin Frans is currently working together with [Danijar](https://github.com/danijar) on writing an updated version of [this preliminary paper,](http://kvfrans.com/static/trpo.pdf) describing the multiple actors setup.

How to run:
```
# This just runs a simple training on Reacher-v1.
python main.py

# For the commands used to recreate results, check trials.txt

```
Parameters:
```
--task: what gym environment to run on
--timesteps_per_batch: how many timesteps for each policy iteration
--n_iter: number of iterations
--gamma: discount factor for future rewards_1
--max_kl: maximum KL divergence between new and old policy
--cg_damping: damp on the KL constraint (ratio of original gradient to use)
--num_threads: how many async threads to use
--monitor: whether to monitor progress for publishing results to gym or not
```

## Requirements

- TensorFlow >= 1.3
- Python 2.7
