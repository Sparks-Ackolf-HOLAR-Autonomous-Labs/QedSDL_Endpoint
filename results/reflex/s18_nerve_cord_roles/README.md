# S18: what the nerve cord does, as a template for an instrument-side controller

Structure-only check on BANC v888 (female brain and nerve cord), pairs with at least 3 synapses.
Code: `src/reflex/s18_nerve_cord_roles.py BANC_DIR OUT_JSON` (2 s). Numbers: `banc_roles.json`.
MaleCNS v1.0 checked 12 Oct 2026 (`malecns_roles.json`, same script with `malecns`): 703 of 707 motor neurons reached from nerve-cord sensors in at most 2 steps without the brain; descending input 8.6 % of motor-neuron input (median descending neuron reaches 268 motor neurons); ascending output onto descending neurons 11.1 %, onto the mushroom body 0.03 %. Same three roles as BANC. Control tests S18a-c: `../s18_control/`.

| Role in the animal | Measure | BANC v888 |
|---|---|---|
| Local reflexes | motor neurons | 805 |
| | motor-neuron input from nerve-cord interneurons | 70.2 % |
| | motor-neuron input from nerve-cord interneurons that receive nerve-cord sensory input | 55.2 % |
| | motor neurons reached from nerve-cord sensors in at most 2 steps without the brain | 699 of 805 |
| Command bottleneck | descending neurons / central-brain intrinsic neurons | 1,329 / 31,879 (4 %) |
| | motor-neuron input directly from descending neurons | 9.6 % |
| | motor neurons one descending neuron reaches in at most 2 steps (median, 90th percentile) | 172, 446 |
| State report | ascending neurons / nerve-cord sensory neurons | 2,371 / 7,845 |
| | ascending output staying in the nerve cord | 41.4 % |
| | ascending output onto descending neurons (short loop) | 6.6 % |
| | ascending output onto the mushroom body | 0.016 % |

28 % of ascending output goes to partners without annotation, so the brain-area shares are lower bounds.

Reading: the nerve cord is a controller, not a memory. The brain sends a small set of commands down;
the nerve cord turns each one into a coordinated pattern over many motor neurons and corrects it with its
own sensors without waiting for the brain; ascending neurons send a compressed status report up, part of
which feeds straight back onto the command neurons. The mushroom body, which the connectome model copies, receives
almost none of it, which is why adding the nerve cord to that model changed nothing (S13, S14).

S18a-c were run on 12 Oct 2026; see `../s18_control/README.md`.
