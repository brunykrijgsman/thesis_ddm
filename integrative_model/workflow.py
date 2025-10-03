import bayesflow as bf
from bayesflow.simulators import make_simulator
from bayesflow.adapters import Adapter
import numpy as np

from integrative_model.simulation import prior, likelihood

def create_workflow():
    """
    Creates and configures the BayesFlow workflow for the integrative DDM.
    
    Returns
    -------
    workflow : bayesflow.BasicWorkflow
        The configured BayesFlow workflow.
    simulator : bayesflow.simulators.Simulator
        The configured simulator.
    adapter : bayesflow.adapters.Adapter
        The configured adapter.
    """
    
    # Meta function to generate random number of trials per trial (uniform distribution between 30 and 1000)
    def meta():
        n_obs = np.random.randint(30, 1001)  # inclusive of 30, exclusive of 1001
        return dict(n_obs=n_obs)             # Uniform distribution between 30 and 1000 --> Experimental context 
    # return dict(n_obs=100) 

    simulator = make_simulator([prior, likelihood], meta_fn=meta)

    # SetTransformer handles variable-length sets naturally 
    summary_network = bf.networks.SetTransformer(summary_dim=8)
    inference_network = bf.networks.CouplingFlow()
    
    adapter = (
        Adapter()
        .broadcast("n_obs", to="choicert") # broadcast n_obs to trial-level data (choicert)
        .as_set(["choicert", "z"]) # treat choicert and z as sets (variable length)
        .standardize(exclude=["n_obs"]) # standardize all except n_obs
        .convert_dtype("float64", "float32") # convert data type
        .concatenate(["alpha", "tau", "beta", "mu_delta", "eta_delta", "gamma", "sigma"], into="inference_variables")
        .concatenate(["choicert", "z"], into="summary_variables")
        .rename("n_obs", "inference_conditions") # treat n_obs as conditioning variable
    )

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        inference_network=inference_network,
        summary_network=summary_network,
    )

    return workflow, simulator, adapter 