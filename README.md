# RepuLink

![Repulink](./figure/RepuLink.png)

**RepuLink** is a network-based reputation system that combines traditional forward propagation with a novel backward propagation mechanism to adjust reputations based on both interactions and endorsements. The algorithm leverages endorsement penalty and reward signals to provide a more robust, calibrated, and explainable reputation evaluation in decentralized networks.


## Overview

In decentralized networks, establishing trust between unknown or partially trusted nodes is a key challenge. **Repulink** introduces a two-layer reputation model that:

1. **Integrates multiple types of feedback:**  
   - **Interaction data:** Captures individual feedback (positive/negative) between nodes.
   - **Endorsement data:** Captures social trust signals through endorsements.

2. **Applies backward propagation mechanisms:**  
   - **Penalty Propagation (BEPP):** Propagates negative feedback from misbehaving nodes back to endorsers.
   - **Reward Propagation (BERP):** Rewards endorsers when the nodes they endorse perform well.

3. **Calibrates final reputations:**  
   - Combines forward reputation computed via power iteration with backward adjustments.
   - Maintains normalized reputation distributions to reflect network performance.

## Usage

The simulation runs via the main driver script `main.py`, which loads processed data, instantiates managers, performs reputation computation, and records node histories.

To run the simulation and save output to a file, you can either use command-line redirection or configure logging in the script.

Example using command-line redirection:

```bash
python3 repulink_example_full.py 
```
Check the results saved under the folder `repulink_multislot_core_modules`.
## Configuration

All configurable parameters are stored in `config.py`, including:

- `EPSILON`: Small constant to prevent division by zero.
- `ALPHA`: Weight between local trust and endorsement contribution.
- `LAMBDA`: Reward sensitivity coefficient.
- `BETA`: Penalty sensitivity coefficient.
- `GAMMA`: Penalty (reward) discount factor.
- `MAX_ITER`: Maximum iterations for convergence.
- `CONVERGENCE_THRESHOLD`: Convergence threshold for the reputation vector.

Feel free to adjust these parameters as needed for experimentation and simulation tuning.
