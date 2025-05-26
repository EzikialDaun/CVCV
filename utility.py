# 파일 이름 유틸리티

import re


def natural_key(filename):
    # 숫자와 문자를 나눠서 정렬 키 생성: '10.png' → ['10', '.png']
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', filename)]
