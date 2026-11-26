"""
Decide which datasets to evaluate on, find their images and check them before any model is loaded.

A dataset entry is a dict describing one group of images that gets its own row in the results:
    real_path, fake_path   folders holding the real and the generated images (can be the same folder)
    source                 name of the dataset the images come from, e.g. wang2020
    generative_model       what made the fake images, e.g. biggan
    family                 type of generator, e.g. gan; looked up in configs/families.yaml when not given
"""
import os
import re

import yaml

from dataset.dataset import get_list


REAL_MARKER = '0_real'
FAKE_MARKER = '1_fake'

# A marker counts only as a whole part of a name: a folder called 0_real, or 0_real between separators
# such as _ - . in a file name (img_7_0_real.png). It does not count when glued to other letters or
# digits, so img_10_real.png, whose 0 belongs to 10, is not labelled real by accident.
MARKER_PATTERNS = {marker: re.compile(rf'(?<![A-Za-z0-9]){marker}(?![A-Za-z0-9])') for marker in (REAL_MARKER, FAKE_MARKER)}

# Options that say where the images are. If any of them is given on the command line, all of them
# (and the 'datasets' list) are ignored in the config file, so the command line fully replaces it.
DATA_OPTIONS = {'realPath', 'fakePath', 'dataPath', 'dataRoot', 'datasets'}

FAMILIES_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'families.yaml')

# Fields of each entry in the 'datasets' list of a config file: (types, required fields)
DATASET_FIELDS = (
    dict(real_path=str, fake_path=str, source=str, family=str, generative_model=str),
    {'real_path', 'fake_path'},
)


def load_families(path=FAMILIES_FILE):
    """Read the family table, which lists the generators of each family, into a generator -> family dict."""
    try:
        with open(path) as f:
            table = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
        raise SystemExit(f'Could not read the families file {path}: {e}')

    if not isinstance(table, dict):
        raise SystemExit(f'{path}: expected family names, each followed by a list of generator names')

    families = {}
    for family, generators in table.items():
        if not isinstance(generators, list) or not all(isinstance(g, str) for g in generators):
            raise SystemExit(f'{path}: {family} must be a list of generator names, e.g. [biggan, stylegan]')
        for generator in generators:
            key = generator.lower()
            if key in families and families[key] != str(family):
                raise SystemExit(f"{path}: '{generator}' is listed under both {families[key]} and {family}")
            families[key] = str(family)

    return families


def is_same_folder(path_a, path_b):
    return os.path.realpath(path_a) == os.path.realpath(path_b)


def marked(images, folder, marker):
    """The images whose path below folder has marker as a whole folder name or part of the file name."""
    return [path for path in images if MARKER_PATTERNS[marker].search(os.path.relpath(path, folder))]


def pick(folder, marker, opposite, kind):
    """
    The images of one side (real or fake) of a dataset whose real and fake images are in different folders:
    the images named with marker if there are any, otherwise all images in the folder.
    Returns the images, notes about images left out, and problems that should stop the run.
    """
    images = get_list(folder)
    chosen = marked(images, folder, marker)

    if chosen:
        notes = []
        if len(chosen) < len(images):
            notes.append(f'{kind} folder {folder}: using {len(chosen)} of {len(images)} images, '
                         f'only those named {marker}')
        return chosen, notes, []

    if marked(images, folder, opposite):
        return images, [], [f'the {kind} folder {folder} contains images named {opposite} but none named {marker}. '
                            f'Are the real and fake folders swapped?']

    return images, [], []


def list_images(real_path, fake_path):
    """
    Find the real and the fake images of a dataset entry. Returns real images, fake images, notes, problems.

    When real_path and fake_path are the same folder, images are told apart by their path: real ones
    have 0_real as a folder name or part of a file name below it, fake ones 1_fake. When they are different
    folders, every image under real_path is real and every image under fake_path is fake, unless that folder
    uses the 0_real/1_fake naming, in which case only the images named that way are used.
    """
    if is_same_folder(real_path, fake_path):
        images = get_list(real_path)
        real_images, fake_images = marked(images, real_path, REAL_MARKER), marked(images, real_path, FAKE_MARKER)

        notes = []
        unused = len(images) - len(set(real_images) | set(fake_images))
        if unused and (real_images or fake_images):
            notes.append(f'{unused} of {len(images)} images are named neither {REAL_MARKER} nor {FAKE_MARKER} '
                         f'and are not used')
        return real_images, fake_images, notes, []

    real_images, real_notes, real_problems = pick(real_path, REAL_MARKER, FAKE_MARKER, 'real')
    fake_images, fake_notes, fake_problems = pick(fake_path, FAKE_MARKER, REAL_MARKER, 'fake')
    return real_images, fake_images, real_notes + fake_notes, real_problems + fake_problems


def find_generators(root, source=None):
    """One entry per subfolder of root, named after the subfolder. source defaults to the name of root."""
    if not os.path.isdir(root):
        raise SystemExit(f'Data folder not found: {root}')

    subfolders = sorted(name for name in os.listdir(root)
                        if not name.startswith('.') and os.path.isdir(os.path.join(root, name)))

    if REAL_MARKER in subfolders or FAKE_MARKER in subfolders:
        raise SystemExit(
            f'{root} contains {REAL_MARKER}/{FAKE_MARKER} folders directly, so it holds the images of a single '
            f'generator. Use --dataPath {root} instead, or give --dataRoot the folder above it.'
        )
    if not subfolders:
        raise SystemExit(f'{root} has no subfolders. --dataRoot expects one subfolder per generator, '
                         f'each containing {REAL_MARKER} and {FAKE_MARKER} folders.')

    source = source or os.path.basename(os.path.normpath(os.path.abspath(root)))

    return [
        dict(real_path=os.path.join(root, name), fake_path=os.path.join(root, name),
             source=source, generative_model=name, family=None)
        for name in subfolders
    ]


def describe(entry):
    """Short name of an entry for messages, e.g. wang2020/biggan."""
    if entry.get('generative_model'):
        return '/'.join(str(part) for part in (entry.get('source'), entry['generative_model']) if part)
    if is_same_folder(entry['real_path'], entry['fake_path']):
        return entry['real_path']
    return f"{entry['real_path']} + {entry['fake_path']}"


def check_datasets(datasets):
    """
    Find the images of every entry and collect what is wrong, so all problems are reported at once.
    Returns notes (worth knowing, e.g. images left out) and problems (the run should not start).
    """
    notes, problems = [], []

    for entry in datasets:
        name = describe(entry)
        # Messages that name their folder only name the entry when that adds something
        prefix = f'{name}: ' if len(datasets) > 1 or entry.get('generative_model') else ''
        entry['real_images'], entry['fake_images'] = [], []

        missing = [path for path in dict.fromkeys([entry['real_path'], entry['fake_path']]) if not os.path.isdir(path)]
        if missing:
            problems.append(f"{prefix}folder not found: {', '.join(missing)}")
            continue

        real_images, fake_images, entry_notes, entry_problems = list_images(entry['real_path'], entry['fake_path'])
        entry['real_images'], entry['fake_images'] = real_images, fake_images
        notes += [prefix + note for note in entry_notes]
        problems += [prefix + problem for problem in entry_problems]

        both = sorted(set(real_images) & set(fake_images))
        if both:
            if is_same_folder(entry['real_path'], entry['fake_path']):
                reason = f'Their path contains both {REAL_MARKER} and {FAKE_MARKER}.'
            else:
                reason = 'One of the real and fake folders is inside the other.'
            problems.append(f'{name}: {len(both)} images count as both real and fake, e.g. {both[0]}. {reason}')

        if not real_images or not fake_images:
            message = f'{name}: found {len(real_images)} real and {len(fake_images)} fake images.'
            if is_same_folder(entry['real_path'], entry['fake_path']):
                message += (f' A folder holding both kinds needs a {REAL_MARKER} and a {FAKE_MARKER} folder '
                            f'inside it (they can be nested, e.g. car/{REAL_MARKER}).')
            else:
                message += ' Check that the folder holds .png, .jpg, .jpeg or .bmp images.'
            problems.append(message)

    return notes, problems


def print_summary(datasets, max_sample=None):
    columns = ['Source', 'Generative model', 'Family', 'Real', 'Fake']
    rows = [[str(entry.get('source') or '-'), str(entry.get('generative_model') or '-'), str(entry.get('family') or '-'),
             str(len(entry['real_images'])), str(len(entry['fake_images']))] for entry in datasets]
    rows.append(['Total', '', '', str(sum(len(e['real_images']) for e in datasets)),
                 str(sum(len(e['fake_images']) for e in datasets))])

    widths = [max(len(row[i]) for row in rows + [columns]) for i in range(len(columns))]

    def line(row):
        return '  '.join(cell.ljust(w) if i < 3 else cell.rjust(w) for i, (cell, w) in enumerate(zip(row, widths)))

    print(f'Datasets found: {len(datasets)}')
    print(line(columns))
    print('-' * len(line(columns)))
    for row in rows[:-1]:
        print(line(row))
    print('-' * len(line(columns)))
    print(line(rows[-1]))
    if max_sample is not None:
        print(f'Using at most {max_sample} real and {max_sample} fake images from each (maxSample)')
    print()


def build_datasets(opt, config_datasets=None):
    """
    Select the datasets to evaluate, find their images and check them.

    Only one way of giving the data is used, in this order: --realPath with --fakePath, --dataPath,
    --dataRoot, the 'datasets' list of a config file. With --dataRoot, --source sets the source name;
    with a 'datasets' list, it keeps only the entries with that source.
    """
    if (opt.realPath is None) != (opt.fakePath is None):
        raise SystemExit('--realPath and --fakePath must be given together')

    given = [name for name, value in [('--realPath/--fakePath', opt.realPath), ('--dataPath', opt.dataPath),
                                      ('--dataRoot', opt.dataRoot), ("the 'datasets' list", config_datasets)]
             if value is not None]
    if len(given) > 1:
        print(f"Note: using {given[0]}; ignoring {', '.join(given[1:])}")

    if opt.realPath is not None:
        datasets = [dict(real_path=opt.realPath, fake_path=opt.fakePath, source=opt.source,
                         generative_model=opt.generativeModel, family=opt.family)]
    elif opt.dataPath is not None:
        datasets = [dict(real_path=opt.dataPath, fake_path=opt.dataPath, source=opt.source,
                         generative_model=opt.generativeModel, family=opt.family)]
    elif opt.dataRoot is not None:
        datasets = find_generators(opt.dataRoot, opt.source)
    elif config_datasets is not None:
        datasets = [dict(entry) for entry in config_datasets
                    if opt.source is None or entry.get('source') == opt.source]
        if not datasets:
            raise SystemExit(f"No datasets to evaluate: no entry of the 'datasets' list has source={opt.source}")
    else:
        raise SystemExit(
            'No data given. Use one of:\n'
            '  --dataRoot <folder>    one subfolder per generator, each with 0_real and 1_fake folders\n'
            '  --dataPath <folder>    a folder with 0_real and 1_fake folders, scored as one dataset\n'
            '  --realPath <folder> --fakePath <folder>    any folder of real and any folder of fake images\n'
            "  a 'datasets' list in a config file given with --config (see configs/evaluate.yaml)"
        )

    families = load_families()
    for entry in datasets:
        if entry.get('family') is None and entry.get('generative_model'):
            entry['family'] = families.get(entry['generative_model'].lower(), 'unknown')

    notes, problems = check_datasets(datasets)
    print_summary(datasets, opt.maxSample)
    if notes:
        print('Notes:\n  - ' + '\n  - '.join(notes) + '\n')
    if problems:
        raise SystemExit('Problems with the datasets:\n  - ' + '\n  - '.join(problems))

    return datasets
