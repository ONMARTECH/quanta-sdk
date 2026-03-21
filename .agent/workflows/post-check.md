---
description: Post-check — Run after every session to validate SDK health
---
// turbo-all

# Post-Check Workflow

Run this after every coding session to validate the SDK is healthy.

## 1. Lint

```bash
cd "/Users/aes/Antigravity Projects/Alfa/quanta"
python3 -m ruff check quanta/ --exclude "quanta/examples/*"
```

Expected: `All checks passed!`

## 2. Tests

```bash
cd "/Users/aes/Antigravity Projects/Alfa/quanta"
python3 -m pytest tests/ -q --no-header --tb=short --cov=quanta --cov-fail-under=80
```

Expected: All tests pass, coverage ≥ 80%.

## 3. Version Consistency

```bash
cd "/Users/aes/Antigravity Projects/Alfa/quanta"
echo "Checking version consistency..."
V1=$(grep '^version' pyproject.toml | cut -d'"' -f2)
V2=$(grep '__version__' quanta/__init__.py | cut -d'"' -f2)
V3=$(grep 'USER_AGENT' quanta/backends/ibm_rest.py | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+')
V4=$(grep '"version":' quanta/mcp_server.py | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+')
V5=$(grep 'version-' README.md | head -1 | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+')

if [ "$V1" = "$V2" ] && [ "$V2" = "$V3" ] && [ "$V3" = "$V4" ] && [ "$V4" = "$V5" ]; then
    echo "✅ Version consistent: $V1 across all 5 files"
else
    echo "❌ Version mismatch:"
    echo "  pyproject.toml:  $V1"
    echo "  __init__.py:     $V2"
    echo "  ibm_rest.py:     $V3"
    echo "  mcp_server.py:   $V4"
    echo "  README.md:       $V5"
fi
```

## 4. Outdated References Scan

```bash
cd "/Users/aes/Antigravity Projects/Alfa/quanta"
echo "Scanning for stale references..."
STALE=$(grep -rn "25 gate\|25 kapi\|25 quantum gate\|14 MCP\|14 tool\|586 pass" \
    README.md docs/ quanta/__init__.py pyproject.toml --include="*.py" --include="*.md" --include="*.toml" 2>/dev/null \
    | grep -v __pycache__ | grep -v ".pyc" || true)

if [ -z "$STALE" ]; then
    echo "✅ No stale references found"
else
    echo "⚠️ Stale references:"
    echo "$STALE"
fi
```

## 5. Structure Validation

```bash
cd "/Users/aes/Antigravity Projects/Alfa/quanta"
echo "Validating project structure..."

# Count source files
SRC=$(find quanta -name "*.py" -not -path "*__pycache__*" | wc -l | tr -d ' ')
TEST=$(find tests -name "test_*.py" | wc -l | tr -d ' ')
DOCS=$(find docs -name "*.md" | wc -l | tr -d ' ')

echo "  Source files: $SRC"
echo "  Test files:   $TEST"
echo "  Doc files:    $DOCS"

# Check critical files exist
for f in pyproject.toml README.md LICENSE .gitignore .env; do
    if [ -f "$f" ]; then
        echo "  ✅ $f exists"
    else
        echo "  ❌ $f MISSING"
    fi
done

# Check .env is gitignored
if git check-ignore -q .env 2>/dev/null; then
    echo "  ✅ .env is gitignored (credentials safe)"
else
    echo "  ⚠️ .env may not be gitignored!"
fi
```

## 6. Dependency & Impact Analysis

```bash
cd "/Users/aes/Antigravity Projects/Alfa/quanta"
echo "=== Dependency Analysis ==="

# Count MCP tools
MCP_TOOLS=$(grep -c '@mcp.tool' quanta/mcp_server.py)
echo "MCP tools: $MCP_TOOLS"

# Count gates
GATES=$(python3 -c "from quanta.core.gates import GATE_REGISTRY; print(len(GATE_REGISTRY))" 2>/dev/null || echo "?")
echo "Gates in registry: $GATES"

# Count algorithms
ALGOS=$(find quanta/layer3 -name "*.py" -not -name "__init__.py" -not -path "*__pycache__*" | wc -l | tr -d ' ')
echo "Layer3 algorithms: $ALGOS"

# Module docstring check
MISSING_DOCS=$(for f in $(find quanta -name "*.py" -not -path "*__pycache__*" -not -name "__init__.py"); do head -3 "$f" | grep -q '"""' || echo "  ⚠️ $f"; done)
if [ -z "$MISSING_DOCS" ]; then
    echo "✅ All modules have docstrings"
else
    echo "⚠️ Missing module docstrings:"
    echo "$MISSING_DOCS"
fi
```

## 7. Git Status

```bash
cd "/Users/aes/Antigravity Projects/Alfa/quanta"
echo "=== Git Status ==="
git status --short
echo ""
echo "Unpushed commits:"
git log origin/main..HEAD --oneline 2>/dev/null || echo "  (none)"
```
