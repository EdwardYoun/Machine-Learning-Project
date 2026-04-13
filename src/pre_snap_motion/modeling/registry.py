from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


def _stringify_categories(values):
    return values.astype(str)


def linear_preprocessor(
    numeric_columns: list[str], categorical_columns: list[str]
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "stringify",
                            FunctionTransformer(_stringify_categories),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )


def tree_preprocessor(
    numeric_columns: list[str], categorical_columns: list[str]
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "stringify",
                            FunctionTransformer(_stringify_categories),
                        ),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )


def build_classification_model(
    name: str, numeric_columns: list[str], categorical_columns: list[str], random_state: int
) -> Pipeline:
    if name == "logistic_regression":
        estimator = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        )
        preprocessor = linear_preprocessor(numeric_columns, categorical_columns)
    elif name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=1,
        )
        preprocessor = tree_preprocessor(numeric_columns, categorical_columns)
    elif name == "gradient_boosting":
        estimator = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=300,
            random_state=random_state,
        )
        preprocessor = tree_preprocessor(numeric_columns, categorical_columns)
    else:
        raise ValueError(f"Unknown classification model: {name}")

    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def build_regression_model(
    name: str, numeric_columns: list[str], categorical_columns: list[str], random_state: int
) -> Pipeline:
    if name == "ridge_regression":
        estimator = Ridge(alpha=1.0)
        preprocessor = linear_preprocessor(numeric_columns, categorical_columns)
    elif name == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=1,
        )
        preprocessor = tree_preprocessor(numeric_columns, categorical_columns)
    elif name == "gradient_boosting":
        estimator = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=6,
            max_iter=300,
            random_state=random_state,
        )
        preprocessor = tree_preprocessor(numeric_columns, categorical_columns)
    else:
        raise ValueError(f"Unknown regression model: {name}")

    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
