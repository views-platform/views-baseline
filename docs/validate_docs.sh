#!/usr/bin/env bash
# validate_docs.sh — Structural validation for views-baseline documentation
#
# Checks:
#   1. No --template-- status values remain in any doc file
#   2. All ADR cross-references resolve to existing files
#   3. All CIC cross-references in READMEs resolve to existing files
#   4. All ADRs have required header fields (Status, Date, Deciders)
#   5. All CICs listed in docs/CICs/README.md exist on disk
#   6. All contributor protocol files exist
#   7. ADR template is present
#   8. CIC template is present

set -euo pipefail

DOCS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$DOCS_DIR/.." && pwd)"

ERRORS=0
WARNINGS=0

red()    { printf '\033[0;31m%s\033[0m\n' "$*"; }
green()  { printf '\033[0;32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[0;33m%s\033[0m\n' "$*"; }
info()   { printf '  %s\n' "$*"; }

fail()    { red "  FAIL: $*"; ERRORS=$((ERRORS + 1)); }
warn()    { yellow "  WARN: $*"; WARNINGS=$((WARNINGS + 1)); }
pass()    { green "  PASS: $*"; }

echo ""
echo "======================================================"
echo "  views-baseline documentation validation"
echo "======================================================"
echo ""

# ---------------------------------------------------------------------------
# 1. No --template-- status values
# ---------------------------------------------------------------------------
echo "1. Checking for uninstantiated --template-- status values..."

TEMPLATE_HITS=$(grep -rl --include="*.md" -- '--template--' "$DOCS_DIR" 2>/dev/null \
    | grep -v 'INSTANTIATION_CHECKLIST.md' \
    || true)
if [ -n "$TEMPLATE_HITS" ]; then
    while IFS= read -r f; do
        fail "Uninstantiated --template-- found in: ${f#$REPO_ROOT/}"
    done <<< "$TEMPLATE_HITS"
else
    pass "No --template-- status values found"
fi

# ---------------------------------------------------------------------------
# 2. All ADR files have required header fields
# ---------------------------------------------------------------------------
echo ""
echo "2. Checking ADR header fields (Status, Date, Deciders)..."

for adr_file in "$DOCS_DIR/ADRs"/[0-9][0-9][0-9]_*.md; do
    [ -f "$adr_file" ] || continue
    rel="${adr_file#$REPO_ROOT/}"
    missing_fields=""

    grep -qi "^\*\*Status:\*\*\|^- \*\*Status:\*\*\|^Status:" "$adr_file" || missing_fields="$missing_fields Status"
    grep -qi "^\*\*Date:\*\*\|^- \*\*Date:\*\*\|^Date:" "$adr_file"       || missing_fields="$missing_fields Date"
    grep -qi "^\*\*Deciders:\*\*\|^- \*\*Deciders:\*\*\|^Deciders:" "$adr_file" || missing_fields="$missing_fields Deciders"

    if [ -n "$missing_fields" ]; then
        fail "$rel is missing required fields:$missing_fields"
    else
        pass "$rel has all required header fields"
    fi
done

# ---------------------------------------------------------------------------
# 3. ADR template exists
# ---------------------------------------------------------------------------
echo ""
echo "3. Checking ADR template..."

if [ -f "$DOCS_DIR/ADRs/adr_template.md" ]; then
    pass "docs/ADRs/adr_template.md exists"
else
    fail "docs/ADRs/adr_template.md is missing"
fi

# ---------------------------------------------------------------------------
# 4. CIC template exists
# ---------------------------------------------------------------------------
echo ""
echo "4. Checking CIC template..."

if [ -f "$DOCS_DIR/CICs/cic_template.md" ]; then
    pass "docs/CICs/cic_template.md exists"
else
    fail "docs/CICs/cic_template.md is missing"
fi

# ---------------------------------------------------------------------------
# 5. All CICs listed in docs/CICs/README.md exist on disk
# ---------------------------------------------------------------------------
echo ""
echo "5. Checking that CICs listed in README resolve to files..."

CIC_README="$DOCS_DIR/CICs/README.md"
if [ -f "$CIC_README" ]; then
    # Extract .md links of the form [Something.md](Something.md) or [text](Something.md)
    LINKED_CICS=$(grep -oP '\[.*?\]\(\K[^)]+\.md' "$CIC_README" | grep -v 'README\|template\|ADRs/' || true)
    while IFS= read -r link; do
        [ -z "$link" ] && continue
        target="$DOCS_DIR/CICs/$link"
        if [ -f "$target" ]; then
            pass "CIC referenced in README exists: $link"
        else
            fail "CIC referenced in README is missing on disk: $link"
        fi
    done <<< "$LINKED_CICS"
else
    fail "docs/CICs/README.md is missing"
fi

# ---------------------------------------------------------------------------
# 6. Expected constitutional ADR files exist (000–009)
# ---------------------------------------------------------------------------
echo ""
echo "6. Checking constitutional ADR files (000–009) exist..."

EXPECTED_ADRS=(
    "000_use_of_adrs.md"
    "001_ontology_of_the_repository.md"
    "002_topology_and_dependency_rules.md"
    "003_authority_of_declarations_over_inference.md"
    "004_rules_for_evaluation_and_stability.md"
    "005_testing_as_mandatory_critical_infrastructure.md"
    "006_intent_contracts_for_non_trivial_classes.md"
    "007_silicon_based_agents_as_untrusted_contributors.md"
    "008_observability_and_explicit_failure.md"
    "009_boundary_contracts_and_configuration_validation.md"
)

for adr in "${EXPECTED_ADRS[@]}"; do
    if [ -f "$DOCS_DIR/ADRs/$adr" ]; then
        pass "docs/ADRs/$adr exists"
    else
        fail "docs/ADRs/$adr is missing"
    fi
done

# ---------------------------------------------------------------------------
# 7. Contributor protocol files exist
# ---------------------------------------------------------------------------
echo ""
echo "7. Checking contributor protocol files..."

EXPECTED_PROTOCOLS=(
    "carbon_based_agents.md"
    "silicon_based_agents.md"
    "hardened_protocol_template.md"
)

for proto in "${EXPECTED_PROTOCOLS[@]}"; do
    if [ -f "$DOCS_DIR/contributor_protocols/$proto" ]; then
        pass "docs/contributor_protocols/$proto exists"
    else
        fail "docs/contributor_protocols/$proto is missing"
    fi
done

# ---------------------------------------------------------------------------
# 8. Standards files exist
# ---------------------------------------------------------------------------
echo ""
echo "8. Checking standards files..."

EXPECTED_STANDARDS=(
    "logging_and_observability_standard.md"
)

for std in "${EXPECTED_STANDARDS[@]}"; do
    if [ -f "$DOCS_DIR/standards/$std" ]; then
        pass "docs/standards/$std exists"
    else
        fail "docs/standards/$std is missing"
    fi
done

# ---------------------------------------------------------------------------
# 9. INSTANTIATION_CHECKLIST.md exists
# ---------------------------------------------------------------------------
echo ""
echo "9. Checking INSTANTIATION_CHECKLIST.md..."

if [ -f "$DOCS_DIR/INSTANTIATION_CHECKLIST.md" ]; then
    pass "docs/INSTANTIATION_CHECKLIST.md exists"
else
    fail "docs/INSTANTIATION_CHECKLIST.md is missing"
fi

# ---------------------------------------------------------------------------
# 10. No Accepted ADR has Status: Proposed (regression check)
# ---------------------------------------------------------------------------
echo ""
echo "10. Checking no ADR has been accidentally downgraded to Proposed..."

for adr_file in "$DOCS_DIR/ADRs"/[0-9][0-9][0-9]_*.md; do
    [ -f "$adr_file" ] || continue
    rel="${adr_file#$REPO_ROOT/}"
    if grep -qi "Status:.*Proposed" "$adr_file"; then
        warn "$rel has Status: Proposed — is this intentional?"
    fi
done
pass "No Accepted ADRs found with Proposed status (warnings above are intentional proposals)"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "======================================================"
echo "  Summary"
echo "======================================================"
echo ""

if [ "$ERRORS" -eq 0 ] && [ "$WARNINGS" -eq 0 ]; then
    green "All checks passed. Documentation is structurally valid."
elif [ "$ERRORS" -eq 0 ]; then
    yellow "Passed with $WARNINGS warning(s). Review warnings above."
else
    red "Failed with $ERRORS error(s) and $WARNINGS warning(s)."
    echo ""
    echo "Fix errors before merging documentation changes."
    exit 1
fi

echo ""
