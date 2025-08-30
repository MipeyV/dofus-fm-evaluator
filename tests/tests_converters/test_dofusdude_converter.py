# tests/test_dofusdude_converter.py
import pytest
from src.adapters.dofusdude_converter import api_item_to_template

def test_api_item_to_template_basic():
    fake_item = {
        "name": "Épée de Boisaille",
        "effects": [
            {
                "int_minimum": 8,
                "int_maximum": 10,
                "type": {"name": "dommage_neutre"}
            },
            {
                "int_minimum": 7,
                "int_maximum": 10,
                "type": {"name": "force"}
            }
        ]
    }

    template = api_item_to_template(fake_item)

    assert template.name == "Épée de Boisaille"
    assert "dommage_neutre" in template.stats
    assert "force" in template.stats

    dom_stat = template.stats["dommage_neutre"]
    assert dom_stat.min_value == 8
    assert dom_stat.max_value == 10
    assert dom_stat.weight > 0