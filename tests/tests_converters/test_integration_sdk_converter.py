# tests/test_converters/test_integration_sdk_converter.py
from dofusdude import ApiClient
from dofusdude.api import equipment_api
from src.adapters.dofusdude_converter import api_item_to_template
from src.core.item_models import ItemTemplate


def test_sdk_to_template_integration():
    """Test d'intégration : API officielle -> converter -> ItemTemplate"""

    # Initialisation du client officiel Dofusdude
    client = ApiClient.get_default()
    equip_api = equipment_api.EquipmentApi(client)

    # Récupère un petit échantillon (1 item)
    items = equip_api.get_items_equipment_list(
        language="fr",
        game="dofus3",
        page_size=1
    )

    api_item = items.items[0].to_dict()

    # Debug : affiche le JSON brut de l'item
    print("RAW ITEM JSON:", api_item)

    # Conversion vers ItemTemplate
    template = api_item_to_template(api_item)

    # --- Vérifications de cohérence ---
    assert isinstance(template, ItemTemplate)
    assert template.name == api_item["name"]
    assert isinstance(template.stats, dict)

    # Vérifie que si l'API fournit des effets, ils sont bien convertis
    if api_item.get("effects"):
        assert len(template.stats) > 0, "L'item avec des effects doit avoir au moins une stat"
