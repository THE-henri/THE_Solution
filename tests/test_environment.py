from core.constants import load_defaults

defaults = load_defaults()
print("Loaded defaults:")
for k, v in defaults.items():
    print(f"{k}: {v}")

