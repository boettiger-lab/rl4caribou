# rl4caribou

Using *deep reinforcement learning* techniques to optimize Caribou conservation policies.
Our apporach is based on the methods developed in the
[approx-model-or-approx-soln](https://github.com/boettiger-lab/approx-model-or-approx-soln) project:

[![DOI](https://zenodo.org/badge/572256056.svg)](https://zenodo.org/badge/latestdoi/572256056) 

## Problem statement

Caribous are facing increased rates of predation by wolves due to the presence of *linear features* in
their habitat---mainly seismic lines and roads which facilitate wolves' access to high elevations during
winter.
Due to this, Caribou numbers have dwindled over the past decades.
Policy makers now face a question of resource allocation: how to best use finite financial resources in
order to restore Caribou populations?

We consider three types of actions available to policy makers: 
1. **Seismic line restoration.** Restoring seismic lines to forest is an expensive policy which takes several years to have its effect felt, however it can greatly reduce the wolf-caribou predation rate on the long run.
2. **Wolf culling.** A second, more direct, option is to cull wolves to bring their population density down.
3. **Moose culling.** A third option is to cull moose---wolves' main source of food. Reducing moose population density effectively reduces the carrying capacity for the wolf population.
