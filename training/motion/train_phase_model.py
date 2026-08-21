from __future__ import annotations

import sys

from stable_phase_training_v3 import main as train_phase_main
from train_attack_ready import train_character_attack_ready


def _argument_value(name: str) -> str | None:
    try:
        index = sys.argv.index(name)
    except ValueError:
        return None
    if index + 1 >= len(sys.argv):
        return None
    return sys.argv[index + 1]


def main() -> int:
    code = int(train_phase_main() or 0)
    if code != 0:
        return code

    character = _argument_value("--character")
    if not character:
        return 0

    print("phase training complete; learning ATTACK/CHAIN_READY from recorded human input...")
    try:
        result = train_character_attack_ready(character)
    except Exception as exc:
        # Phase training is already valid. READY is a downstream layer, so a READY
        # failure must not make the recorder report that the phase model was lost.
        print(f"WARN ATTACK READY training failed: {exc}")
        return 0

    if result.get("ready_model_ready"):
        print("ATTACK/CHAIN_READY layer READY.")
    else:
        print(
            "ATTACK/CHAIN_READY layer not ready yet; phase model remains valid and "
            "future recordings will retry automatically."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
