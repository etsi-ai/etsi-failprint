import pandas as pd
import shap

def explain_failures(model, X_train, X_failures):
    explainer = shap.KernelExplainer(model.predict, X_train)
    shap_values = explainer.shap_values(X_failures)

    mean_shap_values = pd.DataFrame(
        abs(shap_values).mean(axis=0),
        index=X_failures.columns,
        columns=['mean_abs_shap_value']
    ).sort_values(by='mean_abs_shap_value', ascending=False)

    return mean_shap_values
