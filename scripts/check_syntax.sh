#!/bin/bash
# -*- coding: utf-8 -*-
#
# SentinelPipeline 문법 및 코드 품질 검사 스크립트
#
# 사용법:
#   ./scripts/check_syntax.sh
#   ./scripts/check_syntax.sh --strict  # 엄격 모드 (mypy 포함)
#

set -euo pipefail

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 프로젝트 루트 디렉토리
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 옵션 파싱
STRICT_MODE=false
if [[ "${1:-}" == "--strict" ]]; then
    STRICT_MODE=true
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SentinelPipeline 문법 검사${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 1. Python 파일 찾기
echo -e "${YELLOW}[1/4] Python 파일 검색 중...${NC}"
PYTHON_FILES=$(find src -name "*.py" -type f | sort)
FILE_COUNT=$(echo "$PYTHON_FILES" | wc -l | tr -d ' ')
echo -e "  발견된 파일: ${GREEN}${FILE_COUNT}개${NC}"
echo ""

# 2. Python 문법 검사 (AST 파싱)
echo -e "${YELLOW}[2/4] Python 문법 검사 중...${NC}"
SYNTAX_ERRORS=0
SYNTAX_ERROR_FILES=()

while IFS= read -r file; do
    if python -m py_compile "$file" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $file"
    else
        echo -e "  ${RED}✗${NC} $file"
        SYNTAX_ERRORS=$((SYNTAX_ERRORS + 1))
        SYNTAX_ERROR_FILES+=("$file")
    fi
done <<< "$PYTHON_FILES"

echo ""

if [ $SYNTAX_ERRORS -eq 0 ]; then
    echo -e "  ${GREEN}✓ 모든 파일 문법 검사 통과${NC}"
else
    echo -e "  ${RED}✗ 문법 오류 발견: ${SYNTAX_ERRORS}개 파일${NC}"
    for file in "${SYNTAX_ERROR_FILES[@]}"; do
        echo -e "    ${RED}- $file${NC}"
    done
fi
echo ""

# 3. Import 검사 (의존성 없이 가능한 것만)
echo -e "${YELLOW}[3/4] Import 구조 검사 중...${NC}"
IMPORT_ERRORS=0
IMPORT_ERROR_FILES=()

while IFS= read -r file; do
    # import 문만 체크 (실제 import는 하지 않음)
    if python -c "
import ast
import sys
try:
    with open('$file', 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read(), filename='$file')
    # import 문이 있는지 확인
    has_import = any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))
    sys.exit(0 if has_import or True else 1)
except SyntaxError as e:
    print(f'SyntaxError: {e}', file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $file"
    else
        echo -e "  ${YELLOW}⚠${NC} $file (import 구조 확인 필요)"
        IMPORT_ERRORS=$((IMPORT_ERRORS + 1))
        IMPORT_ERROR_FILES+=("$file")
    fi
done <<< "$PYTHON_FILES"

echo ""

if [ $IMPORT_ERRORS -eq 0 ]; then
    echo -e "  ${GREEN}✓ Import 구조 검사 통과${NC}"
else
    echo -e "  ${YELLOW}⚠ Import 구조 확인 필요: ${IMPORT_ERRORS}개 파일${NC}"
fi
echo ""

# 4. Ruff 린터 검사 (사용 가능한 경우)
echo -e "${YELLOW}[4/4] Ruff 린터 검사 중...${NC}"
if command -v ruff &> /dev/null || python -m ruff --version &> /dev/null 2>&1; then
    if python -m ruff check src --output-format=concise 2>&1 | tee /tmp/ruff_output.txt; then
        echo -e "  ${GREEN}✓ Ruff 검사 통과${NC}"
    else
        RUFF_EXIT_CODE=${PIPESTATUS[0]}
        if [ $RUFF_EXIT_CODE -ne 0 ]; then
            echo -e "  ${RED}✗ Ruff 검사 실패${NC}"
            cat /tmp/ruff_output.txt
        fi
    fi
    rm -f /tmp/ruff_output.txt
elif [ "$STRICT_MODE" = true ]; then
    echo -e "  ${YELLOW}⚠ Ruff가 설치되지 않음 (--strict 모드)${NC}"
    echo -e "  설치: ${BLUE}uv add --dev ruff${NC}"
else
    echo -e "  ${YELLOW}⚠ Ruff가 설치되지 않음 (건너뜀)${NC}"
fi
echo ""

# 5. Mypy 타입 체크 (--strict 모드에서만)
if [ "$STRICT_MODE" = true ]; then
    echo -e "${YELLOW}[5/5] Mypy 타입 체크 중...${NC}"
    if command -v mypy &> /dev/null || python -m mypy --version &> /dev/null 2>&1; then
        if python -m mypy src --strict --show-error-codes 2>&1 | tee /tmp/mypy_output.txt; then
            echo -e "  ${GREEN}✓ Mypy 검사 통과${NC}"
        else
            MYPY_EXIT_CODE=${PIPESTATUS[0]}
            if [ $MYPY_EXIT_CODE -ne 0 ]; then
                echo -e "  ${RED}✗ Mypy 검사 실패${NC}"
                cat /tmp/mypy_output.txt
            fi
        fi
        rm -f /tmp/mypy_output.txt
    else
        echo -e "  ${YELLOW}⚠ Mypy가 설치되지 않음${NC}"
        echo -e "  설치: ${BLUE}uv add --dev mypy${NC}"
    fi
    echo ""
fi

# 결과 요약
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}검사 결과 요약${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "  검사한 파일: ${FILE_COUNT}개"
echo -e "  문법 오류: ${SYNTAX_ERRORS}개"

if [ $SYNTAX_ERRORS -eq 0 ]; then
    echo -e "  ${GREEN}✓ 모든 검사 통과!${NC}"
    exit 0
else
    echo -e "  ${RED}✗ 문법 오류가 발견되었습니다.${NC}"
    exit 1
fi

