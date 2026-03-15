# Phase 2: Core Pipeline & Plugin Architecture

**Status**: Ready to start
**Modules**: pipeline, resolvers, cache, degradation
**Estimated**: 50-60 test files
**Goal**: Validate that context compilation and the plugin system actually work as documented

---

## Quick Start

```bash
ssh divante@192.168.50.34
cd ~/git/sr2
pip install --break-system-packages -e ".[dev]"

# Run Phase 2 tests
python3 -m pytest tests/test_pipeline/ tests/test_resolvers/ tests/test_cache/ tests/test_degradation/ -v
```

---

## Modules to Audit

### 1. pipeline (7 test files)

**Files**:
- `tests/test_pipeline/test_engine.py`
- `tests/test_pipeline/test_router.py`
- `tests/test_pipeline/test_conversation.py`
- `tests/test_pipeline/test_compiled_context.py`
- `tests/test_pipeline/test_layer_manager.py`
- (others as found)

**What it should do**:
- Compile contexts from multiple layers
- Enforce token budgets (hard limit, soft warnings)
- Manage conversation state (system, memory, conversation zones)
- Support multiple interfaces with different pipeline configs
- Validate cache consistency

**Questions to answer**:
- [ ] Does engine actually compile contexts? Or just load configs?
- [ ] Are token budgets enforced? What happens when you exceed?
- [ ] Are the three conversation zones properly separated?
- [ ] Does interface routing select the right pipeline config?
- [ ] Can pipeline configs be swapped per request?
- [ ] Are cache policies applied during compilation?

**Investigation steps**:
1. Run: `python3 -m pytest tests/test_pipeline/ -v`
2. Check for failures
3. Read engine tests - look for realistic data (not mocks)
4. Check if token counting is validated (token_budget field actually used)
5. Look for test data that spans multiple layers

---

### 2. resolvers (9 test files)

**Files**:
- `tests/test_resolvers/test_registry.py`
- `tests/test_resolvers/test_content_resolver.py`
- `tests/test_resolvers/test_input_resolver.py`
- `tests/test_resolvers/test_memory_resolver.py`
- `tests/test_resolvers/test_retrieval_resolver.py`
- `tests/test_resolvers/test_session_resolver.py`
- `tests/test_resolvers/test_tools_resolver.py`
- (others as found)

**What it should do**:
- Registry: register/deregister resolvers dynamically
- Each resolver plugs in a content source (input, memory, session, tools, etc.)
- Resolvers can be enabled/disabled per layer
- Resolver chain is composable

**Questions to answer**:
- [ ] Can resolvers be added/removed at runtime?
- [ ] Does registry validation work (e.g., no duplicate names)?
- [ ] Does each resolver return realistic content?
- [ ] Are resolver errors handled gracefully?
- [ ] Can you override a resolver per interface?
- [ ] Is the resolver contract (interface) clearly defined and tested?

**Investigation steps**:
1. Run: `python3 -m pytest tests/test_resolvers/ -v`
2. Check test data - are they mocking LLM calls or using realistic data?
3. Look for "add_resolver" / "remove_resolver" tests
4. Check if retrieval resolver actually retrieves or just returns []
5. Verify memory resolver works with real memory objects

---

### 3. cache (1 test file)

**Files**:
- `tests/test_cache/test_policies.py`
- (potentially more)

**What it should do**:
- Define cache policies (immutable, append_only, no_cache)
- Apply policies to layers
- Validate prefix reuse constraints
- Track cache hits and misses

**Questions to answer**:
- [ ] Are all three cache policies tested?
- [ ] Does immutable prevent changes?
- [ ] Does append_only allow mutations while preserving prefix?
- [ ] Are cache constraints validated?
- [ ] Can you query cache hit rate?

**Investigation steps**:
1. Run: `python3 -m pytest tests/test_cache/ -v`
2. Check how many test files exist (plan says ~1)
3. Look for policy application tests (not just definition)
4. Verify prefix tracking works

---

### 4. degradation (2 test files)

**Files**:
- `tests/test_degradation/test_circuit_breaker.py`
- `tests/test_degradation/test_degradation_ladder.py`
- (potentially more)

**What it should do**:
- Circuit breaker: open after N failures, retry after cooldown
- Degradation ladder: fallback order when a layer fails
- State transitions (closed → open → half_open → closed)

**Questions to answer**:
- [ ] Does circuit breaker open after threshold failures?
- [ ] Does it allow half-open state (trial request)?
- [ ] Are retries scheduled correctly?
- [ ] Does degradation ladder execute fallbacks in order?
- [ ] Are fallback results validated?

**Investigation steps**:
1. Run: `python3 -m pytest tests/test_degradation/ -v`
2. Check for state transition tests
3. Verify timeout/cooldown is actually validated
4. Look for fallback chain tests

---

## Common Issues to Look For

### ❌ Stubs (Core Feature is Missing)
**Symptom**: Test exists and passes, but feature isn't implemented
**Example**: 
```python
def test_interface_routing():
    pipeline = get_pipeline("telegram")
    assert pipeline is not None  # Just checks it loads, not that routing works
```
**Better**:
```python
def test_interface_routing_telegram_vs_http():
    telegram_config = get_pipeline("telegram")
    http_config = get_pipeline("http")
    assert telegram_config.token_budget != http_config.token_budget  # Different configs
```

### ❌ Over-Mocked Tests
**Symptom**: Every dependency is mocked, integration isn't tested
**Example**:
```python
def test_resolver_chain():
    registry = MockRegistry()  # Mocked
    resolver = MockResolver()  # Mocked
    assert registry.add(resolver) == True  # Trivial
```
**Better**:
```python
def test_resolver_chain():
    registry = ResolverRegistry()  # Real registry
    resolver = InputResolver(...)  # Real resolver
    registry.add(resolver)
    assert "input" in registry.names()  # Validates registration
```

### ⚠️ Missing Edge Cases
**Symptom**: Only happy path, no error handling
**Example**: Cache tests that don't test constraint violations
**Action**: Look for tests covering:
- Layer without resolver
- Conflicting cache policies
- Circuit breaker threshold exactly at N (not N-1, N+1)

### ⚠️ Weak Token Budget Tests
**Symptom**: Token budget field exists but not enforced
**Example**:
```python
def test_token_budget():
    config = load_config()
    assert config.token_budget == 32000  # Just checks the value
```
**Better**:
```python
def test_token_budget_enforced():
    # Add enough content to exceed budget
    context = compile_context(layers=[large_layer])
    assert context.total_tokens <= config.token_budget
```

---

## Audit Template (per module)

For each module, fill this out:

```markdown
## Module: {name}

**Files**: {list}
**Test Count**: {X} tests
**Compilation**: ✅/❌ {any import errors?}
**Execution**: ✅/❌ {do tests pass?}
**Overall Status**: ✅ Solid | ⚠️ Weak | ❌ Broken

### Assessment

**Feature Coverage**:
- [ ] Main feature tested (describe what it is)
- [ ] Error cases tested
- [ ] Edge cases tested
- [ ] Integration points tested (not over-mocked)

**Test Quality**:
- [ ] Assertions are specific (not just "not None")
- [ ] Fixtures are clean and reusable
- [ ] Mocks are at system boundaries (not internal APIs)
- [ ] Tests are independent (no ordering)

### Issues Found

1. Issue 1 (severity)
   - Location: file.py:line
   - Why: explanation
   - Impact: what breaks

2. Issue 2...

### Recommendations

- Recommendation 1
- Recommendation 2

### Notes

Any other observations
```

---

## How to Document Findings

As you audit each module, update: `testing/PHASE_2_FINDINGS.md`

Example entry:

```markdown
### pipeline ✅

**Status**: Solid
**Files Reviewed**: 7
**Tests**: 180

**Findings**:
- ✅ Engine tests compile contexts with real data
- ✅ Token budgets are enforced (test verifies limit)
- ✅ Interface routing selects correct config per interface
- ⚠️ Cache policy tests could cover more constraints
- ⚠️ Missing test: What happens when resolver fails?

**Recommendations**:
- Add test for resolver error handling
- Add test for cache constraint violation

---

### resolvers ⚠️

**Status**: Weak - Some resolvers are stubs
**Files Reviewed**: 9
**Tests**: 200

**Findings**:
- ✅ Registry add/remove/list work
- ✅ Input resolver returns user input correctly
- ❌ Retrieval resolver returns [] (stub - doesn't actually retrieve)
  - File: tests/test_resolvers/test_retrieval_resolver.py
  - Indicator: No test data provided to mock memory backend
- ❌ Memory resolver doesn't call extraction (stub)
  - File: tests/test_resolvers/test_memory_resolver.py
  - Indicator: Always returns empty dict

**Recommendations**:
- Implement retrieval resolver (requires memory backend mock)
- Implement memory resolver extraction call
```

---

## Output Files

### During Audit

Create: `testing/PHASE_2_FINDINGS.md`
- Document findings as you go (don't wait until end)
- Include all issues, even small ones
- Include your assessment of test quality

### After Audit

Create: `testing/phase-2-REPORT.md`
- Formalized version of findings
- Summary table of all modules
- Priority checklist of issues to fix
- Next steps before Phase 3

Use `testing/PHASE_REPORT_TEMPLATE.md` as the structure.

---

## When to Stop Audit, Start Fixing

If you find a **blocking issue** (test file doesn't run):
1. Document it in findings
2. Fix it immediately
3. Verify tests now run
4. Continue audit

If you find a **quality issue** (test runs but is weak):
1. Document it in findings
2. Continue audit to get full picture
3. Fix all issues after audit complete

If you find a **stub implementation** (feature documented but code doesn't do it):
1. Document it in findings
2. Continue audit to identify all stubs
3. Prioritize stub fixes in next phase

---

## Questions to Ask Yourself

As you read each test file:

1. **Would this test catch a real bug?**
   - If feature breaks, does test fail?
   - Or does it pass anyway?

2. **Could this test be understood in 30 seconds?**
   - Clear test name?
   - Obvious what's being tested?
   - Assertions are specific?

3. **Is the test environment realistic?**
   - Uses real objects or mocks?
   - Mocks only at system boundaries?
   - Doesn't depend on filesystem/network?

4. **Are all important paths tested?**
   - Happy path? ✅
   - Error cases? (resolver fails, budget exceeded)
   - Edge cases? (empty content, single item, multiple items)

5. **Is this feature actually implemented?**
   - Or is the code a stub?
   - How can you tell? (Look for actual logic, not just placeholders)

---

## Success Criteria for Phase 2

- [ ] All test files in these modules compile without import errors
- [ ] All tests pass
- [ ] Each module has documented assessment (solid/weak/broken)
- [ ] Issues are logged with severity
- [ ] All stubs identified and documented
- [ ] No blocking issues remain for Phase 3

---

## Next Phase

Once Phase 2 is complete:
1. Create `testing/phase-2-REPORT.md`
2. Review findings
3. Fix any blocking issues
4. Start Phase 3 (tests/test_memory/, tests/test_compaction/, tests/test_summarization/, tests/test_tools/, tests/test_a2a/)

**Don't start Phase 3 until Phase 2 report is documented.**
