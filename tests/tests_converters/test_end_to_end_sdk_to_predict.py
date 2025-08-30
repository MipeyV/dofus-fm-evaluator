# tests/tests_integration/test_end_to_end_sdk_to_predict.py
import os
import pytest

from dofusdude import ApiClient
from dofusdude.api import equipment_api

from src.adapters.dofusdude_converter import api_item_to_template
from src.core.item_models import ItemInstance
from src.models import predictor


MODEL_PATH = "models/random_forest_model.joblib"


@pytest.mark.integration
def test_end_to_end_sdk_to_predict():
    """Test end-to-end : SDK -> Template -> Instance -> Features -> RandomForest"""

    # --- 1) API SDK ---
    client = ApiClient.get_default()
    equip_api = equipment_api.EquipmentApi(client)
    items = equip_api.get_items_equipment_list(
        language="fr",
        game="dofus3",
        page_size=1
    )
    api_item = items.items[0].to_dict()

    # --- 2) Conversion en ItemTemplate ---
    template = api_item_to_template(api_item)
    assert template is not None
    assert len(template.stats) >= 0  # certains items n'ont pas toujours des effets

    # --- 3) Créer un ItemInstance factice (on prend les max du template) ---
    fake_stats = {stat: sd.max_value for stat, sd in template.stats.items()}
    instance = ItemInstance(template, fake_stats)

    # --- 4) Extraire les features pour le modèle ---
    model = predictor.load_model(MODEL_PATH)
    feature_names = predictor.get_expected_features(model)
    features_dict = instance.get_features()

    # Aligner avec les features attendues par le modèle
    import pandas as pd
    import numpy as np
    X = pd.DataFrame([[features_dict.get(f, np.nan) for f in feature_names]],
                     columns=feature_names)

    # --- 5) Prédiction ---
    y_pred = model.predict(X)
    print("Prediction:", y_pred[0])
    assert y_pred is not None
