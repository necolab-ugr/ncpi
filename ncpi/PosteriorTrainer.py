import torch
from sbi.inference import NPE, NLE, NRE
from sbi.neural_nets import posterior_nn, likelihood_nn, classifier_nn
from sbi.utils import BoxUniform

def train_posterior(
    theta_train,
    x_train,
    prior=None,
    inference_type="npe",
    model="nsf",
    hidden_features=50,
    num_transforms=5,
    training_batch_size=100,
    learning_rate=0.001
):
    """
    Entrena un posterior dado theta y x usando sbi.NPE, NLE o NRE.

    Devuelve:
    - posterior: objeto posterior con .sample() y .sample_batched()
    - inference: objeto de inferencia para reutilizar si se desea
    """
    inference_type = inference_type.lower()

    if prior is None:
        theta_min = theta_train.min(dim=0).values
        theta_max = theta_train.max(dim=0).values
        prior = BoxUniform(low=theta_min, high=theta_max)

    if inference_type == "npe":
        estimator_fn = posterior_nn(model=model, hidden_features=hidden_features, num_transforms=num_transforms)
        inference = NPE(prior=prior, density_estimator=estimator_fn)

    elif inference_type == "nle":
        estimator_fn = likelihood_nn(model=model, hidden_features=hidden_features, num_transforms=num_transforms)
        inference = NLE(prior=prior, density_estimator=estimator_fn)

    elif inference_type == "nre":
        estimator_fn = classifier_nn(model=model, hidden_features=hidden_features)
        inference = NRE(prior=prior, ratio_estimator=estimator_fn)

    else:
        raise ValueError(f"Inference type '{inference_type}' no reconocido. Usa 'npe', 'nle' o 'nre'.")

    inference = inference.append_simulations(theta=theta_train, x=x_train)
    trained = inference.train(training_batch_size=training_batch_size, learning_rate=learning_rate)
    posterior = inference.build_posterior(trained)

    return posterior, inference
