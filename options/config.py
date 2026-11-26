import argparse
import difflib

import yaml


# Options that get_model() fills with loaded models at runtime, so they can't come from a config file
RUNTIME_OPTIONS = {'LGradGenerativeModel', 'diffusion', 'diffusionModel', 'direArgs', 'defakeClipEncode', 'defakeBlip'}


def convert_value(action, value):
    """Convert a YAML value the way argparse converts the same value typed on the command line."""
    if value is None:
        return None

    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)) or action.type is bool:
        if not isinstance(value, bool):
            raise ValueError(f"'{action.dest}' must be true or false, got {value!r}")
        return value

    if isinstance(value, (bool, list, dict)):
        raise ValueError(f"'{action.dest}' must be a single value, got {value!r}")

    convert = action.type or str
    try:
        return convert(str(value))
    except ValueError:
        raise ValueError(f"'{action.dest}' must be of type {convert.__name__}, got {value!r}")


def check_entries(key, entries, fields):
    """Check a list option such as 'datasets'. fields is (types of all allowed fields, set of required fields)."""
    types, required = fields

    if not isinstance(entries, list):
        raise ValueError(f"'{key}' must be a list")

    for i, entry in enumerate(entries):
        where = f'{key}[{i}]'
        if not isinstance(entry, dict):
            raise ValueError(f'{where} must be a mapping of field names to values')

        unknown = sorted(set(entry) - set(types))
        if unknown:
            raise ValueError(f'{where}: unknown field(s) {unknown}. Valid fields: {sorted(types)}')

        missing = sorted(required - set(entry))
        if missing:
            raise ValueError(f'{where}: missing required field(s) {missing}')

        for name, value in entry.items():
            if value is not None and (isinstance(value, bool) or not isinstance(value, types[name])):
                raise ValueError(f'{where}: {name} must be of type {types[name].__name__}, got {value!r}')


def command_line_options(parser, args=None):
    """The names of the options given on the command line, as opposed to left at their defaults."""
    not_given = object()
    namespace = argparse.Namespace(**{action.dest: not_given for action in parser._actions if action.dest != 'help'})
    # argparse only fills in defaults for attributes the namespace doesn't have yet, so every option
    # still holding not_given afterwards was not on the command line
    given, _ = parser.parse_known_args(args, namespace=namespace)
    return {name for name, value in vars(given).items() if value is not not_given}


def parse_with_config(parser, lists=None, replace_groups=(), args=None):
    """
    Parse command-line options, taking defaults from an optional YAML file given with --config.

    Precedence: built-in defaults < values in the YAML file < flags on the command line.
    lists maps the names of list options that only exist in YAML (e.g. 'datasets') to the fields
    check_entries() expects. replace_groups are sets of option names that belong together: if any of them
    is given on the command line, the file's values for all of them are ignored, e.g. --dataPath on the
    command line replaces a dataRoot set in the file. Returns the parsed options and a dict with the
    lists found in the file.
    """
    lists = lists or {}

    parser.add_argument('--config', type=str, default=None, help='YAML file with option values; command-line flags override it')
    known, _ = parser.parse_known_args(args)
    if known.config is None:
        return parser.parse_args(args), {}

    on_command_line = command_line_options(parser, args)
    replaced = set().union(*[group for group in replace_groups if group & on_command_line])

    try:
        with open(known.config) as f:
            config = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
        parser.error(f'could not read config file {known.config}: {e}')

    if not isinstance(config, dict):
        parser.error(f'{known.config}: expected option names and values, e.g. "modelName: UnivFD"')

    actions = {
        action.dest: action for action in parser._actions
        if action.dest not in ('help', 'config') and action.dest not in RUNTIME_OPTIONS
    }

    defaults, found_lists, ignored = {}, {}, []
    for key, value in config.items():
        try:
            if key in replaced and key not in on_command_line:
                if value is not None:
                    ignored.append(key)
            elif key in lists:
                check_entries(key, value, lists[key])
                found_lists[key] = value
            elif key in actions:
                defaults[key] = convert_value(actions[key], value)
            else:
                valid = list(actions) + list(lists)
                close = difflib.get_close_matches(str(key), valid, n=1)
                hint = f"did you mean '{close[0]}'?" if close else 'valid options: ' +', '.join(sorted(valid))
                raise ValueError(f"unknown option '{key}', {hint}")
        except ValueError as e:
            parser.error(f'{known.config}: {e}')

    if ignored:
        print(f"Note: {', '.join(ignored)} from {known.config} ignored, because the command line gives "
              f"{', '.join(sorted(name for name in on_command_line if name in replaced))} instead")

    parser.set_defaults(**defaults)

    return parser.parse_args(args), found_lists
