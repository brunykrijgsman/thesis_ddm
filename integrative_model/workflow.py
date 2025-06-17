import bayesflow as bf
from bayesflow.simulators import make_simulator
from bayesflow.adapters import Adapter

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
    
    def meta():
        return dict(n_obs=100)

    simulator = make_simulator([prior, likelihood], meta_fn=meta)

    summary_network = bf.networks.SetTransformer(summary_dim=8)
    inference_network = bf.networks.CouplingFlow()
    adapter = (
        Adapter()
        .broadcast("n_obs", to="choicert")    
        .as_set(["choicert", "z"])
        .standardize(exclude=["n_obs"])
        .convert_dtype("float64", "float32")
        .concatenate(["alpha", "tau", "beta", "mu_delta", "eta_delta", "gamma", "sigma"], into="inference_variables")
        .concatenate(["choicert", "z"], into="summary_variables")
        .rename("n_obs", "inference_conditions")
    )

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        inference_network=inference_network,
        summary_network=summary_network,
    )

    return workflow, simulator, adapter 