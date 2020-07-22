"""
A programatic way of specifying the models and serialize them as JSON
"""

import json
from typing import Dict

from src.conf import Model

# Define models
models: Dict[str, Model] = dict()

# Model 1: compare normals vs patients - major categorical factors + sex + age
categories = ["sex", "COVID19"]  # "patient",
continuous = ["age"]
technical = [
    "processing_batch_continuous"
]  # "processing_batch_continuous", "processing_batch_categorical"
variables = categories + continuous + technical
formula = None
model_name = "1-general"
model = Model(
    covariates=variables,
    categories=categories,
    continuous=continuous,
    formula=formula,
)
models[model_name] = model


# Model 2: look deep into patients
categories = [
    "sex",
    "COVID19",
    "severity_group",
    "hospitalization",
    "intubation",
    "death",
    "diabetes",
    "obesity",
    "hypertension",
]
continuous = ["age", "time_symptoms"]  # "bmi", "time_symptoms"]
technical = ["processing_batch_continuous"]
variables = categories + continuous + technical
formula = None
model_name = "2-covid"
model = Model(
    covariates=variables,
    categories=categories,
    continuous=continuous,
    formula=formula,
)
models[model_name] = model


# Model 3: look at changes in treatment
categories = [
    "severe",  # <- take only severe patients
    "sex",
    "tocilizumab",
]
continuous = [
    "age",
]
technical = [
    # "processing_batch_continuous"
]
variables = categories + continuous + technical
formula = None
model_name = "3-treatment"
model = Model(
    covariates=variables,
    categories=categories,
    continuous=continuous,
    formula=formula,
)
models[model_name] = model


# Model 4+: interactions of sex with other factors
f = ["severity_group", "death", "hospitalization", "intubation", "tocilizumab"]
for factor in f:
    categories = [
        factor,
        "sex",
    ]
    continuous = []
    technical = []
    variables = categories + continuous + technical
    model_name = f"4-interaction_sex_{factor}"
    model = Model(
        covariates=variables,
        categories=categories,
        continuous=continuous,
        formula=f"~ sex * {factor}",
    )
    models[model_name] = model

json.dump(models, open("metadata/model_specifications.json", "w"))

# Model 5: look at temporal changes in severe patients
categories = [
    "severe",  # <- take only severe patients
    "sex",
    "tocilizumab",
]
continuous = ["age", "time_symptoms"]
technical = [
    # "processing_batch_continuous"
]
variables = categories + continuous + technical
formula = None
model_name = "5-time"
model = Model(
    covariates=variables,
    categories=categories,
    continuous=continuous,
    formula=formula,
)
models[model_name] = model
