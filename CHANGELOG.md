## v0.3.0 (2026-06-01)

### Feat

- move cli.py → cli/main.py, drop api_service dep, lazy-import uvicorn
- add initial source codebase (visionflow v0.1.0)

### Fix

- **ci**: fix YAML syntax error in bootstrap step of semantic-versioning workflow
- **ci**: bootstrap initial semver tag and fix cz dry-run exit-code handling
- **ci**: fix GitHub Pages permissions and semantic-versioning branch target
- **ci**: remove uv sync from release build, use uv tool for twine
- **ci**: wire release and PyPI publish as explicit downstream jobs
- import main function not module in __main__.py
- suppress bandit B324 by marking MD5 as non-security hash
- resolve CI failures in integration tests, docs build, and security checks
- update remaining template category labels to capture in cli.py
- update AttributeError message to ai_vision_tool in __getattr__

## v0.2.0 (2026-06-01)
