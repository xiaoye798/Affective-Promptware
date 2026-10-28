"""以固定诊断、动作菜单和响应格式渲染情绪、中性或压力上下文；不执行文本中的代码。"""
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
        raise ValueError('模板需要四个非空字符串槽位，不能用完整正文替代固定决策界面。')
    if not re.fullmatch(r'wf\d{3}',bindings['SCENARIO_ID']):raise ValueError('场景编号不合规。')
    if not bindings['PAIR_ID'].startswith(bindings['SCENARIO_ID']+'-'):raise ValueError('配对编号与场景不一致。')
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
    group.add_argument('--bindings',type=Path,help='四槽位JSON，例如template_bindings中的示例')
    group.add_argument('--artifact',type=Path,help='提取已保存材料的槽位并逐字校验')
    args=parser.parse_args()
    if args.bindings:
        output=render(json.loads(args.bindings.read_text(encoding='utf-8')))
    else:
        content=args.artifact.read_text(encoding='utf-8')
        original=json.loads(content)['full_user_message'] if args.artifact.suffix=='.json' else content
        output=render(extract_bindings(original))
        if output!=original:raise ValueError('输入包含与固定模板不一致的诊断、菜单或格式。')
    print(output,end='')

if __name__=='__main__':main()
