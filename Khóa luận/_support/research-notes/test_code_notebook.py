"""Small synthetic checks for scientific data handling, without reading clinical data."""
import ast
import hashlib
import json
import re
from pathlib import Path

import nbformat

NOTEBOOK = Path(__file__).resolve().parents[2] / 'COde_01_Data_Audit_and_Preparation.ipynb'
nb = nbformat.read(NOTEBOOK, as_version=4)
nbformat.validate(nb)
namespace = {'re': re, 'hashlib': hashlib, 'json': json, 'Path': Path}
wanted_functions = {'is_missing', 'clean_csv_value', 'parse_labels', 'parse_image_names', 'screen_field', 'dedupe_bag', 'digest'}
wanted_assignments = {'MISSING_MARKERS', 'TARGETS', 'LABEL_MAPPING', 'TARGET_ALIASES', 'target_pattern'}
for cell in nb.cells:
    if cell.cell_type != 'code':
        continue
    tree = ast.parse(cell.source)
    for node in tree.body:
        select = isinstance(node, ast.FunctionDef) and node.name in wanted_functions
        select |= isinstance(node, ast.ClassDef) and node.name == 'UnionFind'
        select |= isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in wanted_assignments for t in node.targets)
        if select:
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<notebook-test>', 'exec'), namespace)

assert namespace['parse_labels']('') == []
assert namespace['clean_csv_value'](' N/A ') == ''
assert namespace['clean_csv_value']('  No caries.\nNo pulpitis.  ') == 'No caries.\nNo pulpitis.'
assert namespace['clean_csv_value']('0001-001') == '0001-001'
assert namespace['parse_labels']('N/A') == []
assert namespace['parse_labels']('Pulpitis,Dental Caries,Pulpitis') == ['Dental Caries', 'Pulpitis']
assert namespace['parse_labels']('Dental Crowding') == ['Dental Crowding']
assert 'Dental Crowding' not in namespace['LABEL_MAPPING']
assert namespace['parse_image_names']('0001-001-01.jpg', 'Photographs', '0001-001') == ['Images/Photographs/0001-001-01.jpg']
for value in ['../outside.jpg', '0002-001-01.jpg', '0001-001-01.jpg,0001-001-01.jpg']:
    try:
        namespace['parse_image_names'](value, 'Photographs', '0001-001')
    except AssertionError:
        pass
    else:
        raise AssertionError(f'Invalid image reference accepted: {value}')

uf = namespace['UnionFind'](['1', '2', '3', '4'])
uf.union('1', '2')
uf.union('3', '2')
assert uf.find('1') == uf.find('3')
assert uf.find('4') != uf.find('1')

clean, removed, flag = namespace['screen_field']('No dental caries. Red gingiva. No pulpitis.')
assert clean == 'Red gingiva.' and len(removed) == 2 and flag
assert namespace['screen_field']('N/A') == ('', [], False)
namespace['pixel_by_path'] = {'photo/a.jpg': 'same', 'photo/b.jpg': 'same', 'photo/c.jpg': 'different'}
assert namespace['dedupe_bag'](['photo/a.jpg', 'photo/b.jpg', 'photo/c.jpg']) == ['photo/a.jpg', 'photo/c.jpg']
assert namespace['digest']({'a': 1, 'b': 2}) == namespace['digest']({'b': 2, 'a': 1})
print('PASS: schema, missing/multi-label handling, path validation, transitive patient grouping, clinical negation screening, bag deduplication.')
