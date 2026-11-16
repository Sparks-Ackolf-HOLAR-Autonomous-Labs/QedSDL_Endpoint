# P55 econ re-analysis, G1 (existing outputs, 20 seeds, one look at 0.25 unless stated)

L = lost + c_false * false stops + c_time * furnace runs (c_lost = 1). 'reanneal' adds one anneal per false stop.
Endpoints use the never-stop (gp) trajectory: stopping changes what the planner sees next, so stop-all and the
oracle are bounds, not runs. Stop-all also gives the planner no fidelity data.

## Frontier (means over 20 seeds)

| arm | lost | false stops | furnace runs |
|---|---|---|---|
| gp | 7.75 | 0.00 | 30.00 |
| cbo | 4.35 | 0.00 | 30.00 |
| gp+fly_best | 3.30 | 1.80 | 24.90 |
| gp+fly_best_shuf | 3.15 | 2.10 | 25.09 |
| gp+logreg | 3.90 | 2.50 | 24.60 |
| gp+fly_best_rand | 3.50 | 2.90 | 24.49 |
| gp+knn | 3.90 | 3.25 | 24.04 |
| gp+fly | 3.55 | 4.05 | 23.59 |
| stop-all (gp traj.) | 0.00 | 22.25 | 7.50 |
| oracle hindsight (lower bound) | 0.00 | 0.00 | 24.19 |

Pareto set on (lost, false stops): cbo, gp+fly_best, gp+fly_best_shuf, stop-all (gp traj.)

## Who beats the connectome model (gp+fly_best) and where

Differences are other arm minus gp+fly_best (means). Other arm wins where d_lost + c_false d_false + c_time d_furn < 0.

| arm | d_lost | d_false | d_furnace | wins at c_time=0 if c_false | wins at c_false=0 if c_time |
|---|---|---|---|---|---|
| gp | +4.45 | -1.80 | +5.10 | > 2.472 | never |
| cbo | +1.05 | -1.80 | +5.10 | > 0.583 | never |
| gp+fly | +0.25 | +2.25 | -1.31 | never | > 0.190 |
| gp+fly_best_shuf | -0.15 | +0.30 | +0.19 | < 0.500 | < 0.800 |
| gp+fly_best_rand | +0.20 | +1.10 | -0.41 | never | > 0.485 |
| gp+knn | +0.60 | +1.45 | -0.86 | never | > 0.696 |
| gp+logreg | +0.60 | +0.70 | -0.30 | never | > 2.000 |
| stop-all (gp traj.) | -3.30 | +20.45 | -17.40 | < 0.161 | always |

## Ranking at named ratios (S4 arms + stop-all)

**c_false 0.096 (Janus share), c_time 0**: stop-all (gp traj.) 2.14; gp+fly_best_shuf 3.35; gp+fly_best 3.47; gp+fly_best_rand 3.78; gp+fly 3.94 (oracle 0.00)
  fly_best behind: stop-all (gp traj.) (fly +1.34, 4/16 seeds, p=0.000388), gp+fly_best_shuf (fly +0.12, 3/3 seeds, p=0.6)

**c_false 0.096, c_time 0.1, reanneal**: stop-all (gp traj.) 5.11; gp+fly_best_shuf 6.07; gp+fly_best 6.14; gp+fly_best_rand 6.52; gp+fly 6.70 (oracle 2.42)
  fly_best behind: stop-all (gp traj.) (fly +1.03, 4/16 seeds, p=0.000586), gp+fly_best_shuf (fly +0.07, 3/3 seeds, p=0.753)

**c_false 0.25, c_time 0**: gp+fly_best_shuf 3.67; gp+fly_best 3.75; gp+fly_best_rand 4.22; cbo 4.35; gp+logreg 4.53 (oracle 0.00)
  fly_best behind: gp+fly_best_shuf (fly +0.07, 3/3 seeds, p=0.75)

**c_false 0.5, c_time 0**: gp+fly_best 4.20; gp+fly_best_shuf 4.20; cbo 4.35; gp+fly_best_rand 4.95; gp+logreg 5.15 (oracle 0.00)
  fly_best behind: none

**c_false 1, c_time 0**: cbo 4.35; gp+fly_best 5.10; gp+fly_best_shuf 5.25; gp+fly_best_rand 6.40; gp+logreg 6.40 (oracle 0.00)
  fly_best behind: cbo (fly +0.75, 7/11 seeds, p=0.228)

**c_false 0.5, c_time 0.5**: stop-all (gp traj.) 14.88; gp+fly_best 16.65; gp+fly_best_shuf 16.74; gp+fly_best_rand 17.19; gp+fly 17.37 (oracle 12.09)
  fly_best behind: stop-all (gp traj.) (fly +1.77, 3/17 seeds, p=0.000272)

**c_false 0, c_time 1**: stop-all (gp traj.) 7.50; gp+fly 27.14; gp+knn 27.94; gp+fly_best_rand 27.99; gp+fly_best 28.20 (oracle 24.19)
  fly_best behind: stop-all (gp traj.) (fly +20.70, 0/20 seeds, p=8.82e-05), gp+fly (fly +1.06, 7/10 seeds, p=0.155), gp+knn (fly +0.26, 8/8 seeds, p=0.568), gp+fly_best_rand (fly +0.21, 6/6 seeds, p=0.582)

**c_false 1, c_time 1**: stop-all (gp traj.) 29.75; gp+fly_best 30.00; gp+fly_best_shuf 30.34; gp+fly_best_rand 30.89; gp+logreg 31.00 (oracle 24.19)
  fly_best behind: stop-all (gp traj.) (fly +0.25, 8/11 seeds, p=0.702)

Winner share of the grid c_false 0-2 x c_time 0-1 (with stop-all, reanneal=False): gp+fly_best 50%, stop-all (gp traj.) 30%, cbo 18%, gp+fly_best_shuf 2%
Winner share of the grid c_false 0-2 x c_time 0-1 (with stop-all, reanneal=True): gp+fly_best 66%, cbo 28%, stop-all (gp traj.) 5%, gp+fly_best_shuf 2%
Winner share of the grid c_false 0-2 x c_time 0-1 (learned stoppers and never-stop only, reanneal=False): gp+fly_best 65%, cbo 18%, gp+fly 10%, gp+fly_best_shuf 8%
Winner share of the grid c_false 0-2 x c_time 0-1 (learned stoppers and never-stop only, reanneal=True): gp+fly_best 68%, cbo 28%, gp+fly_best_shuf 4%

## probe time (one look at f)

| arm | lost | false | furnace |
|---|---|---|---|
| gp (never stop, S4) | 7.75 | 0.00 | 30.00 |
| gp+fly@0.1 | 3.55 | 4.05 | 22.53 |
| gp+fly@0.25 | 3.55 | 4.05 | 23.59 |
| gp+fly@0.5 | 3.50 | 4.15 | 25.68 |
| gp+fly@0.75 | 3.45 | 3.45 | 27.95 |
| gp+fly_best@0.1 | 3.65 | 2.70 | 23.43 |
| gp+fly_best@0.25 | 3.30 | 1.80 | 24.90 |
| gp+fly_best@0.5 | 3.40 | 2.05 | 26.73 |
| gp+fly_best@0.75 | 3.40 | 2.25 | 28.30 |
| gp+knn@0.1 | 4.20 | 3.70 | 22.44 |
| gp+knn@0.25 | 3.90 | 3.25 | 24.04 |
| gp+knn@0.5 | 3.95 | 2.70 | 26.32 |
| gp+knn@0.75 | 3.75 | 2.60 | 28.20 |
| gp+logreg@0.1 | 3.90 | 2.65 | 23.52 |
| gp+logreg@0.25 | 3.90 | 2.50 | 24.60 |
| gp+logreg@0.5 | 3.75 | 2.25 | 26.38 |
| gp+logreg@0.75 | 3.50 | 1.50 | 28.52 |
| gp@0.25 | 7.75 | 0.00 | 30.00 |
| stop-all (gp traj.) | 0.00 | 22.25 | 7.50 |

c_false 0.096 (Janus share), c_time 0: stop-all (gp traj.) 2.14; gp+fly_best@0.25 3.47; gp+fly_best@0.5 3.60; gp+fly_best@0.75 3.62
c_false 0.096, c_time 0.1, reanneal: stop-all (gp traj.) 5.11; gp+fly_best@0.25 6.14; gp+fly_best@0.5 6.47; gp+fly_best@0.1 6.52
c_false 0.25, c_time 0: gp+fly_best@0.25 3.75; gp+logreg@0.75 3.88; gp+fly_best@0.5 3.91; gp+fly_best@0.75 3.96
c_false 0.5, c_time 0: gp+fly_best@0.25 4.20; gp+logreg@0.75 4.25; gp+fly_best@0.5 4.42; gp+fly_best@0.75 4.53
c_false 1, c_time 0: gp+logreg@0.75 5.00; gp+fly_best@0.25 5.10; gp+fly_best@0.5 5.45; gp+fly_best@0.75 5.65
c_false 0.5, c_time 0.5: stop-all (gp traj.) 14.88; gp+fly_best@0.25 16.65; gp+fly_best@0.1 16.71; gp+fly@0.1 16.84
c_false 0, c_time 1: stop-all (gp traj.) 7.50; gp+fly@0.1 26.08; gp+knn@0.1 26.64; gp+fly_best@0.1 27.08
c_false 1, c_time 1: stop-all (gp traj.) 29.75; gp+fly_best@0.1 29.78; gp+fly_best@0.25 30.00; gp+logreg@0.1 30.07

## any-time rule

| arm | lost | false | furnace |
|---|---|---|---|
| gp | 7.75 | 0.00 | 30.00 |
| gp (never stop, S4) | 7.75 | 0.00 | 30.00 |
| gp+fly | 2.65 | 5.95 | 20.70 |
| gp+fly_best | 2.60 | 4.65 | 22.08 |
| gp+knn | 2.30 | 7.70 | 19.35 |
| gp+logreg | 2.35 | 5.60 | 21.20 |
| gp+mlp | 2.35 | 7.05 | 20.07 |
| stop-all (gp traj.) | 0.00 | 22.25 | 7.50 |

c_false 0.096 (Janus share), c_time 0: stop-all (gp traj.) 2.14; gp+logreg 2.89; gp+mlp 3.03; gp+knn 3.04
c_false 0.096, c_time 0.1, reanneal: stop-all (gp traj.) 5.11; gp+logreg 5.57; gp+fly_best 5.72; gp+mlp 5.74
c_false 0.25, c_time 0: gp+logreg 3.75; gp+fly_best 3.76; gp+mlp 4.11; gp+fly 4.14
c_false 0.5, c_time 0: gp+fly_best 4.92; gp+logreg 5.15; gp+fly 5.62; gp+mlp 5.88
c_false 1, c_time 0: gp+fly_best 7.25; gp 7.75; gp (never stop, S4) 7.75; gp+logreg 7.95
c_false 0.5, c_time 0.5: stop-all (gp traj.) 14.88; gp+logreg 15.75; gp+knn 15.82; gp+mlp 15.91
c_false 0, c_time 1: stop-all (gp traj.) 7.50; gp+knn 21.65; gp+mlp 22.43; gp+fly 23.35
c_false 1, c_time 1: gp+logreg 29.16; gp+fly 29.30; gp+fly_best 29.33; gp+knn 29.35

## tuned variants

| arm | lost | false | furnace |
|---|---|---|---|
| fly_fair_t1:fly | 3.55 | 4.05 | 23.59 |
| fly_fair_t1:logreg | 3.40 | 2.60 | 24.71 |
| gp (never stop, S4) | 7.75 | 0.00 | 30.00 |
| nn_e1:gp+logreg | 3.90 | 2.50 | 24.60 |
| nn_e1:gp+mlp32 | 3.95 | 2.30 | 24.68 |
| nn_e1:gp+mlp64x2 | 3.60 | 2.10 | 24.86 |
| nn_fair_j1:((64, 64), 0.0001) | 3.60 | 2.10 | 24.86 |
| nn_fair_j1:logreg | 3.90 | 2.50 | 24.60 |
| stop-all (gp traj.) | 0.00 | 22.25 | 7.50 |

c_false 0.096 (Janus share), c_time 0: stop-all (gp traj.) 2.14; fly_fair_t1:logreg 3.65; nn_e1:gp+mlp64x2 3.80; nn_fair_j1:((64, 64), 0.0001) 3.80
c_false 0.096, c_time 0.1, reanneal: stop-all (gp traj.) 5.11; fly_fair_t1:logreg 6.38; nn_e1:gp+mlp64x2 6.50; nn_fair_j1:((64, 64), 0.0001) 6.50
c_false 0.25, c_time 0: fly_fair_t1:logreg 4.05; nn_e1:gp+mlp64x2 4.12; nn_fair_j1:((64, 64), 0.0001) 4.12; nn_e1:gp+logreg 4.53
c_false 0.5, c_time 0: nn_e1:gp+mlp64x2 4.65; nn_fair_j1:((64, 64), 0.0001) 4.65; fly_fair_t1:logreg 4.70; nn_e1:gp+mlp32 5.10
c_false 1, c_time 0: nn_e1:gp+mlp64x2 5.70; nn_fair_j1:((64, 64), 0.0001) 5.70; fly_fair_t1:logreg 6.00; nn_e1:gp+mlp32 6.25
c_false 0.5, c_time 0.5: stop-all (gp traj.) 14.88; fly_fair_t1:logreg 17.06; nn_e1:gp+mlp64x2 17.08; nn_fair_j1:((64, 64), 0.0001) 17.08
c_false 0, c_time 1: stop-all (gp traj.) 7.50; fly_fair_t1:fly 27.14; fly_fair_t1:logreg 28.11; nn_e1:gp+mlp64x2 28.46
c_false 1, c_time 1: stop-all (gp traj.) 29.75; nn_e1:gp+mlp64x2 30.56; nn_fair_j1:((64, 64), 0.0001) 30.56; fly_fair_t1:logreg 30.71

## m2: value of knowing the purge rate (S20 slow purge, 20 seeds x 30 flakes)

oracle_k knows the rate and loses 0, so it is an oracle, not a method. VoI = rule's losses - oracle's losses.

| rule | VoI, 3 calibration flakes | VoI, 10 calibration flakes |
|---|---|---|
| seq_kc | 2.35 | 1.00 |
| snap_ridge | 3.15 | 1.90 |
| snap_kc | 2.45 | 1.90 |
| snap_mlp | 3.00 | 2.10 |
| lstm_tuned | 4.15 | 2.90 |
| now_ridge | 4.60 | 3.35 |
| lstm | 3.95 | 3.80 |
| seq_ridge | 7.25 | 4.65 |
| trend_rule | 12.45 | 10.90 |

## m1: optimal stopping

The one-look rule stops at the first checkpoint whose score crosses a threshold: a myopic one-step rule.
Under L the problem is a finite-horizon optimal stopping problem (Wald 1947; Chow, Robbins & Siegmund 1971),
solved by backward induction on the posterior that the flake is failing. The any-time and probe-time files
only sample this; none of the arms is the Bayes-optimal stopper, and the oracle row bounds what any can reach.
