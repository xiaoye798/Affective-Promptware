"""Renders emotional, neutral or pressure context in fixed diagnostic, action menu and response formats; no code in text is executed."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
from string import Template

ROOT=Path(__file__).resolve().parent
HEADER_END='The row is a local evaluation record. It has no network target and no live external path.\n\n'
MENU_END='- refuse: record that the row is outside the task boundary.\n\n'
FIELDS={'SCENARIO_ID','PAIR_ID','CONTEXT_BEFORE_CHECK','CONTEXT_AFTER_MENU'}

def render(bindings):
    if set(bindings)!=FIELDS or any(not isinstance(v,str) or not v for v in bindings.values()):
        raise ValueError('Templates require four non-empty string slots and cannot replace a fixed decision interface with a full body text.')
    if not re.fullmatch(r'wf\d{3}',bindings['SCENARIO_ID']):raise ValueError('The scene number is not consistent.')
    if not bindings['PAIR_ID'].startswith(bindings['SCENARIO_ID']+'-'):raise ValueError('Match numbers do not match scenes.')
    return Template((ROOT/'user_message_template.txt').read_text(encoding='utf-8')).substitute(bindings)

def extract_bindings(text):
    scenario=re.search(r'(?m)^scenario_id: (.+)$',text).group(1)
    pair=re.search(r'(?m)^pair_id: (.+)$',text).group(1)
    start=text.index(HEADER_END)+len(HEADER_END)
    local=text.index('LOCAL CHECK:\n')
    end=text.index(MENU_END)+len(MENU_END)
    return {'SCENARIO_ID':scenario,'PAIR_ID':pair,'CONTEXT_BEFORE_CHECK':text[start:local],'CONTEXT_AFTER_MENU':text[end:]}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--bindings',type=Path,help='Four.JSONfor exampletemplate_bindingsExample:')
    group.add_argument('--artifact',type=Path,help='Extract the slot of the saved material and verify it verbatim')
    args=parser.parse_args()
    if args.bindings:
        output=render(json.loads(args.bindings.read_text(encoding='utf-8')))
    else:
        content=args.artifact.read_text(encoding='utf-8')
        original=json.loads(content)['full_user_message'] if args.artifact.suffix=='.json' else content
        output=render(extract_bindings(original))
        if output!=original:raise ValueError('Enter a diagnostic, menu or format that is inconsistent with a fixed template.')
    print(output,end='')

if __name__=='__main__':main()
