# Success Condition: MLSys Paper for Claudeloop

## Automated Tests

```bash
# Test 1: Paper directory and main files exist
test -f paper/main.tex && test -f paper/references.bib && echo "PASS: Core files exist" || echo "FAIL: Missing core files"
```

```bash
# Test 2: Paper compiles successfully with pdflatex
cd paper && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex && test -f main.pdf && echo "PASS: Paper compiles to PDF" || echo "FAIL: Compilation failed"
```

```bash
# Test 3: Paper has required sections
grep -c '\\section' paper/main.tex | awk '{if ($1 >= 5) print "PASS: At least 5 sections found ("$1")"; else print "FAIL: Only "$1" sections found, need at least 5"}'
```

```bash
# Test 4: Paper has at least 3 tables
grep -c '\\begin{table' paper/main.tex | awk '{if ($1 >= 3) print "PASS: At least 3 tables found ("$1")"; else print "FAIL: Only "$1" tables found, need at least 3"}'
```

```bash
# Test 5: Paper has at least 2 figures
grep -c '\\begin{figure' paper/main.tex | awk '{if ($1 >= 2) print "PASS: At least 2 figures found ("$1")"; else print "FAIL: Only "$1" figures found, need at least 2"}'
```

```bash
# Test 6: Paper has references/bibliography
grep -c '\\cite{' paper/main.tex | awk '{if ($1 >= 8) print "PASS: At least 8 citations found ("$1")"; else print "FAIL: Only "$1" citations found, need at least 8"}'
```

```bash
# Test 7: BibTeX file has enough entries
grep -c '@' paper/references.bib | awk '{if ($1 >= 8) print "PASS: At least 8 bib entries ("$1")"; else print "FAIL: Only "$1" bib entries, need at least 8"}'
```

```bash
# Test 8: Paper is approximately 4 pages (check PDF page count)
cd paper && pdfinfo main.pdf 2>/dev/null | grep Pages | awk '{if ($2 >= 4 && $2 <= 6) print "PASS: Paper is "$2" pages"; else print "FAIL: Paper is "$2" pages, expected 4-6"}' || echo "PASS: pdfinfo not available, skipping page count check"
```

```bash
# Test 9: System overview figure uses TikZ
grep -c 'tikzpicture' paper/main.tex | awk '{if ($1 >= 1) print "PASS: TikZ figure found ("$1" tikzpicture environments)"; else print "FAIL: No TikZ figures found"}'
```

```bash
# Test 10: Paper includes required keywords
for kw in "claudeloop" "worker" "evaluator" "iteration" "Claude Code" "hybrid evaluation" "cost" "token"; do
  if grep -qi "$kw" paper/main.tex; then
    echo "PASS: Keyword '$kw' found"
  else
    echo "FAIL: Keyword '$kw' missing"
  fi
done
```

## Qualitative Evaluation Criteria

The evaluator agent should assess the paper on the following dimensions. **The paper passes only if it scores >= 7/10 on ALL dimensions and >= 8/10 on average.**

### 1. Paper Structure (1-10)
- Does the paper follow standard ML conference structure (Abstract, Introduction, System Design, Experiments, Related Work, Conclusion)?
- Are sections well-organized and logically flowing?
- Is the abstract concise (~150 words) and informative?
- **Score < 7 = FAIL**

### 2. System Description Quality (1-10)
- Is the system architecture clearly explained?
- Does Figure 1 (system overview) accurately depict the worker-evaluator loop?
- Are all components (worker, evaluator, backends, runner) described with sufficient detail?
- Is the hybrid evaluation approach well-motivated?
- **Score < 7 = FAIL**

### 3. Figures Quality (1-10)
- Are there at least 2 figures?
- Is the system overview figure clear, readable, and accurately representing the architecture?
- Is the token/cost analysis figure informative and well-labeled?
- Do figures have proper captions?
- Are figures referenced in the text?
- **Score < 7 = FAIL**

### 4. Tables Quality (1-10)
- Are there at least 3 tables?
- Do tables present meaningful quantitative data?
- Are tables properly formatted with captions?
- Do tables include: task completion results, performance benchmarks, and cost breakdown?
- Are tables referenced and discussed in the text?
- **Score < 7 = FAIL**

### 5. Experimental Rigor (1-10)
- Does the paper present concrete experimental results?
- Are metrics clearly defined and reported?
- Is there a cost/efficiency analysis?
- Is there comparison or ablation discussion?
- Are the numbers consistent with the actual system data?
- **Score < 7 = FAIL**

### 6. Writing Quality (1-10)
- Is the writing clear, concise, and free of grammatical errors?
- Is the technical language appropriate for an ML systems audience?
- Are claims well-supported by evidence?
- Is the paper within the 4-page limit (content pages, excluding references)?
- **Score < 7 = FAIL**

### 7. Technical Depth (1-10)
- Does the paper go beyond surface-level description?
- Are design decisions justified?
- Is the hybrid evaluation approach analyzed (deterministic + LLM)?
- Are limitations discussed?
- **Score < 7 = FAIL**

## Scoring Summary

The evaluator should output a JSON response with:

```json
{
  "success": true/false,
  "reason": "Brief explanation",
  "suggestions": "What to improve if not passing",
  "scores": {
    "structure": X,
    "system_description": X,
    "figures": X,
    "tables": X,
    "experiments": X,
    "writing": X,
    "technical_depth": X,
    "average": X.X
  }
}
```

**Pass condition:** All individual scores >= 7 AND average >= 8.0 AND all automated tests pass.
