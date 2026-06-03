import lightgbm as lgb
import xgboost as xgb
from src.config import config

def get_model(model_type: str = "lightgbm"):
    """
    Factory function to initialize classifiers with cost-sensitive parameters.
    """
    if model_type == "lightgbm":
        return lgb.LGBMClassifier(**config.LGBM_PARAMS)
    elif model_type == "xgboost":
        # Adjust XGBoost params to match scale_pos_weight
        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "learning_rate": 0.05,
            "max_depth": 6,
            "scale_pos_weight": config.LGBM_PARAMS["scale_pos_weight"],
            "random_state": 42,
            "n_estimators": 500
        }
        return xgb.XGBClassifier(**xgb_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
