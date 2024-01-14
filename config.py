from dynaconf import Dynaconf

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_file="settings.toml",
    root_path='.',
    environments=True
)