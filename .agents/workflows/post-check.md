---
description: Post-check — Run after every session to validate SDK health
---

# Post-Check Workflow

Run this after every coding session to validate Quanta SDK health.

## Steps

// turbo-all

1. Run lint check:
```bash
python3 -m ruff check quanta/ --exclude "quanta/examples/*"
```

2. Run full test suite with coverage:
```bash
python3 -m pytest tests/ -q --no-header --tb=short --cov=quanta --cov-fail-under=80 2>&1 | tail -5
```

3. Version consistency check (5 files must match):
```bash
V1=$(grep '^version' pyproject.toml | cut -d'"' -f2) && V2=$(grep '__version__' quanta/__init__.py | cut -d'"' -f2) && V3=$(grep 'USER_AGENT' quanta/backends/ibm_rest.py | grep -o '[0-9]\.[0-9]\.[0-9]') && V4=$(grep '"version":' quanta/mcp_server.py | grep -o '[0-9]\.[0-9]\.[0-9]') && V5=$(grep 'version-' README.md | head -1 | grep -o '[0-9]\.[0-9]\.[0-9]') && if [ "$V1" = "$V2" ] && [ "$V2" = "$V3" ] && [ "$V3" = "$V4" ] && [ "$V4" = "$V5" ]; then echo "✅ Version consistent: $V1 (5 files)"; else echo "❌ Mismatch: pyproject=$V1 init=$V2 ibm=$V3 mcp=$V4 readme=$V5"; fi
```

4. MCP server load check:
```bash
python3 -c "from quanta.mcp_server import mcp; print('✅ MCP server loads OK')"
```

5. Documentation consistency — check for stale numbers:
```bash
echo "Stale '14 tools' refs:" && grep -rn "14 MCP\|14 tool" quanta/ docs/ README.md --include="*.md" --include="*.py" 2>/dev/null | grep -v __pycache__ | wc -l | xargs -I{} sh -c '[ {} -eq 0 ] && echo "✅ None found" || echo "❌ Found {} stale refs"'
```

6. Architecture alignment check:
```bash
python3 -c "
from quanta.qec.decoder import DecoderBase, MWPMDecoder, UnionFindDecoder
from quanta.result import Result
from quanta.core.circuit import CircuitDefinition
print(f'DecoderBase ABC: ✅')
print(f'MWPM inherits: {issubclass(MWPMDecoder, DecoderBase)}')
print(f'UF inherits: {issubclass(UnionFindDecoder, DecoderBase)}')
print(f'Result._repr_html_: {hasattr(Result, \"_repr_html_\")}')
print(f'Circuit._repr_html_: {hasattr(CircuitDefinition, \"_repr_html_\")}')
"
```

## Expected Results

- Lint: All checks passed
- Tests: 580+ passed, coverage ≥ 87%
- Version: Consistent across 5 files
- MCP: Loads without error
- Docs: No stale tool/test count references
- Architecture: DecoderBase ABC, Jupyter _repr_html_ both present
