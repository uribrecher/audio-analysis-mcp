"""List all available Demucs models and their properties."""
from demucs.pretrained import get_model

models = ["htdemucs", "htdemucs_ft", "htdemucs_6s", "hdemucs_mmi", "mdx", "mdx_extra"]
for name in models:
    try:
        m = get_model(name)
        sources = m.sources if hasattr(m, "sources") else list(m.models[0].sources)
        is_bag = hasattr(m, "models")
        num_sub = len(m.models) if is_bag else 1
        print(f"{name:20s} stems={sources}  bag={is_bag} sub_models={num_sub}")
    except Exception as e:
        print(f"{name:20s} ERROR: {e}")
