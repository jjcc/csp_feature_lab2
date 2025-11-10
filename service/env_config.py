import os
import yaml
from dotenv import load_dotenv

class ConfigLoader:
    def __init__(self, config_path='config.yaml', fallback_to_env=True):
        self.config_path = config_path
        self.fallback_to_env = fallback_to_env
        self._config = None
        self._env = None

    def _load_yaml_config(self):
        """Load YAML configuration file."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}

    def _load_env_config(self):
        """Load environment variables as fallback."""
        if self.fallback_to_env:
            load_dotenv('.env', override=False)
            return dict(os.environ)
        return {}

    def _flatten_config(self, config, prefix='', separator='_'):
        """Flatten nested YAML config to match env variable style."""
        flattened = {}
        for key, value in config.items():
            new_key = f"{prefix}{separator}{key}" if prefix else key
            if isinstance(value, dict):
                flattened.update(self._flatten_config(value, new_key.upper(), separator))
            else:
                flattened[new_key.upper()] = str(value) if value is not None else ''
        return flattened

    def get(self, key, default=None):
        """Get configuration value by key, with fallback to env variables."""
        if self._config is None:
            yaml_config = self._load_yaml_config()
            self._config = self._flatten_config(yaml_config)

        #if self._env is None:
        #    self._env = self._load_env_config()

        # Try YAML config first, then env variables, then default
        #return self._config.get(key, self._env.get(key, default))
        return self._config.get(key, None)

    def get_section(self, section_name):
        """Get an entire configuration section."""
        if self._config is None:
            yaml_config = self._load_yaml_config()
            self._config = self._flatten_config(yaml_config)

        return {k: v for k, v in self._config.items() if k.startswith(f"{section_name.upper()}_")}
    
    def get_common_configs_raw(self):
        """Get common configuration parameters."""
        yaml_config = self._load_yaml_config()
        return yaml_config.get('common_configs', {})
    

    def get_derived_file(self, basic_csv):
        # derive macro csv and output csv from basic csv
        # get the section after "trades_raw_"
        if "trades_raw_" in basic_csv:
            section = basic_csv.split("trades_raw_")[-1].split(".csv")[0]
            macro_csv = f"trades_with_gex_macro_{section}.csv"
            output_csv = f"labeled_trades_{section}.csv"
            return macro_csv, output_csv
        else:
            return None, None

# Global config instance
config = ConfigLoader()

def getenv(key, default=None):
    """Drop-in replacement for os.getenv that uses YAML config first."""
    return config.get(key, default)

def load_env(dotenv_path: str = None):
    load_dotenv(dotenv_path or os.environ.get('DOTENV_PATH', '.env'), override=False)
    env = {k:v for k,v in os.environ.items()}
    params = {}
    def maybe_read_yaml(path_key, default_path):
        p = env.get(path_key, default_path)
        if p and os.path.exists(p):
            with open(p, 'r') as fh:
                return yaml.safe_load(fh) or {}
        return {}
    params.update(maybe_read_yaml('THRESHOLDS_FILE', './configs/thresholds_branch_b.yaml'))
    meta_extra = maybe_read_yaml('THRESHOLDS_META_FILE', './configs/thresholds_meta.yaml')
    params.update(meta_extra or {})
    for k in ['LAMBDA_CVAR','KELLY_FRAC','MAX_SIZE','TAU_ACCEPT','T_TAIL','M_TAIL','T_WIN_P','T_U','TAU_BINS']:
        v = env.get(k)
        if v not in (None, '', 'None'):
            try:
                params[k.lower()] = float(v)
            except:
                params[k.lower()] = v
    if 't_tail' not in params and env.get('T_TAIL'): params['t_tail'] = float(env['T_TAIL'])
    if 'm_tail' not in params and env.get('M_TAIL'): params['m_tail'] = float(env['M_TAIL'])
    if 't_win_P' not in params and env.get('T_WIN_P'): params['t_win_P'] = float(env['T_WIN_P'])
    if 't_U' not in params and env.get('T_U'): params['t_U'] = float(env['T_U'])
    if 'lambda_cvar' not in params and env.get('LAMBDA_CVAR'): params['lambda_cvar'] = float(env['LAMBDA_CVAR'])
    if 'kelly_frac' not in params and env.get('KELLY_FRAC'): params['kelly_frac'] = float(env['KELLY_FRAC'])
    if 'max_size' not in params and env.get('MAX_SIZE'): params['max_size'] = float(env['MAX_SIZE'])
    if 'tau_accept' not in params and env.get('TAU_ACCEPT'): params['tau_accept'] = float(env['TAU_ACCEPT'])
    if 'tau_bins' not in params and env.get('TAU_BINS'): params['tau_bins'] = float(env['TAU_BINS'])
    return env, params
