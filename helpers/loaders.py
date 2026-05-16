"""Cached resource loaders for the Streamlit app."""
import logging
import pickle

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


@st.cache_resource
def load_models() -> dict:
    try:
        logger.info("Loading models...")
        with open("model/lightgbm_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("model/calibrator.pkl", "rb") as f:
            calibrator = pickle.load(f)
        with open("model/feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)
        logger.info("Models loaded: %d trees, %d features", model.num_trees(), len(feature_names))
        return {"model": model, "calibrator": calibrator, "feature_names": feature_names, "success": True}
    except Exception as exc:
        logger.error("Error loading models: %s", exc)
        return {"success": False, "error": str(exc)}


@st.cache_resource
def load_data() -> dict:
    try:
        logger.info("Loading data...")
        with open("data/features_test.pkl", "rb") as f:
            X_test, y_test, _ = pickle.load(f)
        with open("data/ieee_prepared.pkl", "rb") as f:
            data_dict = pickle.load(f)
        logger.info("Data loaded: %d test samples", len(X_test))
        return {"X_test": X_test, "y_test": y_test, "test_raw": data_dict["test"], "success": True}
    except Exception as exc:
        logger.error("Error loading data: %s", exc)
        return {"success": False, "error": str(exc)}


@st.cache_resource
def load_feature_importance() -> "pd.DataFrame | None":
    try:
        return pd.read_csv("model/feature_importance.csv")
    except Exception as exc:
        logger.error("Error loading feature importance: %s", exc)
        return None
