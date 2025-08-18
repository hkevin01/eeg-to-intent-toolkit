from __future__ import annotations

import numpy as np


def csp_lda(x: np.ndarray, y: np.ndarray, n_components: int = 6):
    """CSP + LDA pipeline using pyriemann if available.

    Returns a fitted pipeline object; raises if optional deps missing.
    """
    try:  # pragma: no cover
        from pyriemann.estimation import Covariances
        from pyriemann.spatialfilters import CSP
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.pipeline import Pipeline
    except Exception as e:  # pragma: no cover
        msg = "pyriemann and scikit-learn are required for csp_lda"
        raise RuntimeError(msg) from e

    pipe = Pipeline(
        steps=[
            ("cov", Covariances(estimator="oas")),
            ("csp", CSP(n_components=n_components)),
            ("clf", LinearDiscriminantAnalysis()),
        ]
    )
    return pipe.fit(x, y)
