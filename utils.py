import yaml

def _convert_types(d):
    # 재귀적으로 dict의 float string을 float으로 변환
    if isinstance(d, dict):
        return {k: _convert_types(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [_convert_types(x) for x in d]
    elif isinstance(d, str):
        # 1e-5, 0.001 등 float string을 float으로 변환
        try:
            if d.lower() in ("true", "false"):  # bool은 건드리지 않음
                return d
            if "." in d or "e" in d.lower():
                return float(d)
            return int(d)
        except Exception:
            return d
    else:
        return d

def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return _convert_types(cfg)
