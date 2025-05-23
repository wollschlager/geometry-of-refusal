# The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence

This repository contains the code for the experiments presented in our paper: [The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence](https://arxiv.org/abs/2502.17420).

## Overview

The code is primarily designed for VSCode interactive mode (similar to Jupyter notebooks) but also mostly supports command-line execution. We use [Weights & Biases](https://wandb.ai/) for experiment logging and result storage.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/wollschlager/topdown-interpretability.git
    cd topdown-interpretability
    ```

2.  **Set up the environment:**

    ```bash
    # Create the conda environment
    conda create -n rdo python=3.10
    conda activate rdo
    pip install -r requirements.txt
    ```

## Configuration (.env)

This project uses a `.env` file in the root directory to manage configuration settings. You should take a look at this file and change the parameters to suit your needs.

## Usage

### Computing DIM Directions

   *   Navigate to the `refusal_direction` subdirectory.
   *   Run the pipeline script:
       ```bash
       cd refusal_direction
       python -m pipeline.run_pipeline --model_path google/gemma-2-2b-it
       cd ..
       ```
     *(Note: This part has minimal differences from [https://github.com/andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction))*

**2. Compute the Refusal Direction Optimization (RDO) directions, cones, or rep. ind. directions:**

   *   Train the RDO direction:

       ```bash
       python rdo.py --train_direction --model google/gemma-2-2b-it
       ```
   *   Train a direction orthogonal to the main RDO direction:

       ```bash
       python rdo.py --train_orthogonal_direction --model google/gemma-2-2b-it
       ```
   *   Train a direction independent from the DIM direction:

       ```bash
       python rdo.py --train_independent_direction --model google/gemma-2-2b-it
       ```
   *   Train a refusal cone (example with dimension 2):

       ```bash
       # Adjust min/max_cone_dim, n_sample as needed
       python rdo.py --train_cone --min_cone_dim 2 --max_cone_dim 2 --n_sample 16 --model google/gemma-2-2b-it
       ```

### Evaluating Directions

Use the `run_rdo_pipeline.py` script to evaluate the computed RDO, Cone, or other directions stored in a Weights & Biases run.

*   **Evaluate a specific W&B run (e.g., RDO direction):**

    ```bash
    # Replace 'fast-shadow-11' with your W&B run display name
    python -m pipeline.run_rdo_pipeline --wandb_run fast-shadow-11
    ```
*   **Evaluate the basis vectors of a refusal cone (e.g., dimension 2):**

    ```bash
    # Replace 'dim_2' with your W&B run display name corresponding to the cone
    python -m pipeline.run_rdo_pipeline --wandb_run dim_2
    ```
*   **After evaluating the basis vectors, you can evaluate the performance of Monte-Carlo samples in the cone:**

    ```bash
    # Replace 'dim_2' with your W&B run display name corresponding to the cone
    python -m pipeline.run_rdo_samples --wandb_run dim_2 --sample_start_idx 0 --sample_end_idx 8 # evaluates sample 0 to 7, by default 512 samples are saved during basis vector evaluation
    ```

## Analysis

The `cosinesims_analysis.py` script can be used to analyze the cosine similarities between the computed DIM direction and rep. ind. directions.
