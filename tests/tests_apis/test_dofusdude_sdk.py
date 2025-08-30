def test_dofusdude_sdk_fetch():
    from dofusdude import ApiClient
    from dofusdude.api import equipment_api
    import json

    client = ApiClient.get_default()
    equip_api = equipment_api.EquipmentApi(client)

    # On récupère juste une petite page
    items = equip_api.get_items_equipment_list(
        language="fr",
        game="dofus3",
        page_size=5
    )
    print(f"✅ {len(items.items)} items récupérés")

    for it in items.items:
        print(f"- id={it.ankama_id} | name={it.name} | lvl={it.level} | type={it.type.name}")

    # Exemple dump JSON complet sur un item
    first_id = items.items[0].ankama_id
    item = equip_api.get_items_equipment_single(
        language="fr",
        game="dofus3",
        ankama_id=first_id
    )
    item_json = item.to_dict()
    item_json.pop("conditions", None)  # on vire la clé qui pose problème
    print(json.dumps(item_json, indent=2, ensure_ascii=False))