import streamlit.delta_generator as delta_generator
from streamlit.delta_generator import DeltaGenerator, BlockType

def _fake_check_nested_element_violation(
    dg: DeltaGenerator, block_type: str | None, ancestor_block_types: list[BlockType]
) -> None:
    return None

delta_generator._check_nested_element_violation = _fake_check_nested_element_violation
