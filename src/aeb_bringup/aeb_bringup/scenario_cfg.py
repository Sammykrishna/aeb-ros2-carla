import numpy as np

TARGET_MODELS = ["vehicle.audi.a2", "vehicle.audi.tt", "vehicle.citroen.c3", "vehicle.nissan.micra",
                 "vehicle.seat.leon", "vehicle.toyota.prius"]


def jitter_cfg(cfg: dict, rng: np.random.Generator) -> dict:
    out = dict(cfg, ego_kmh=cfg["ego_kmh"] + rng.uniform(-1.0, 1.0))
    if "brake_after_s" in cfg:
        out["brake_after_s"] = cfg["brake_after_s"] + rng.uniform(-0.2, 0.2)
    if cfg.get("no_target"):
        return out
    if cfg["target_kmh"] > 0:
        out["target_kmh"] = cfg["target_kmh"] + rng.uniform(-1.0, 1.0)
    out["gap_m"] = cfg["gap_m"] * rng.uniform(0.95, 1.05)
    out["lateral_offset_m"] = cfg.get("lateral_offset_m", 0.0) + rng.uniform(-0.2, 0.2)
    out["target_model"] = str(rng.choice(TARGET_MODELS))
    return out
