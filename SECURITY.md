# Security Guidelines

This document outlines security practices and guidelines for the FineTune project to protect sensitive data, API keys, and credentials.

## 🚨 Immediate Actions Required

### If You Received a Security Alert

1. **Do NOT panic** - Security alerts are often false positives for legitimate placeholder data
2. **Review the alert** - Check if the flagged content is real credentials or placeholder/example data
3. **Follow the response process** below based on your assessment

### False Positive Response Process

If the alert is for placeholder/example data:
1. **Verify it's actually a false positive** - Ensure the flagged content is not real credentials
2. **Report to InfoSec** - Email `infosec@redhat.com` with subject "Leak Pattern Update Request - False Positive"
3. **Include details**: Link to detection, explanation that it's template/placeholder data
4. **Update allowlist** - Add patterns to `.gitleaks.toml` (see section below)

### Real Security Incident Response

If real credentials were exposed:
1. **Contact InfoSec immediately** - Email `infosec@redhat.com`
2. **Rotate credentials** - Invalidate and regenerate the exposed secrets
3. **Remove from git history** - Follow git history cleaning instructions from InfoSec
4. **Update security practices** - Review and improve credential management

## 🛡️ Prevention: Credential Management Best Practices

### Never Commit Real Credentials

✅ **DO:**
- Use environment variables for all secrets
- Store credentials in secure credential management systems
- Use `.env` files for local development (excluded by `.gitignore`)
- Generate API keys dynamically in CI/CD pipelines
- Use placeholder values in example files

❌ **DON'T:**
- Hardcode API keys, passwords, or tokens in source code
- Commit `.env` files containing real credentials
- Use real credentials in test files
- Store credentials in configuration files committed to git

### Environment Variable Usage

For local development, create a `.env` file (not committed):
```bash
# .env (local only - never commit this file)
HF_TOKEN=your_real_huggingface_token_here
OPENAI_API_KEY=your_real_openai_key_here
WANDB_API_KEY=your_real_wandb_key_here
```

In your code, access these safely:
```python
import os
from pathlib import Path

# Load from environment
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    raise ValueError("HF_TOKEN environment variable is required")

# Alternative: load from .env file
from dotenv import load_dotenv
load_dotenv()  # Load .env file if it exists
```

### Creating Example/Template Files

When creating example configuration files:

✅ **GOOD Example:**
```yaml
# config.example.yml
huggingface:
  token: "hf_EXAMPLE_TOKEN_REPLACE_WITH_REAL_VALUE"  # notsecret

api_keys:
  openai: "sk-EXAMPLE_OPENAI_KEY_HERE"  # notsecret
  wandb: "EXAMPLE_WANDB_API_KEY"  # notsecret
```

❌ **BAD Example:**
```yaml
# config.yml (avoid realistic-looking fake keys)
huggingface:
  token: "hf_a1b2c3d4e5f6789abcdef123456789abcdef"

api_keys:
  openai: "sk-1nv4l1dK3yTh4tL00k5R34l"
```

### Best Practices for Placeholder Data

1. **Use obvious placeholders:**
   - `EXAMPLE_TOKEN_HERE`
   - `your_api_key_here`
   - `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

2. **Add security comments:**
   ```python
   API_KEY = "example_key_replace_me"  # notsecret
   ```

3. **Use the word "EXAMPLE" in placeholder values:**
   ```yaml
   token: "hf_EXAMPLE_TOKEN_REPLACE_WITH_REAL"
   ```

4. **Keep placeholders shorter than real secrets when possible**

5. **Use file naming conventions:**
   - `config.example.yml` (not `config.yml`)
   - `passwords.template.yml`
   - `secrets.sample.env`

## 🔧 Security Tools and Configuration

### GitLeaks Configuration

This repository includes a `.gitleaks.toml` configuration file that defines allowlist rules for false positive prevention. The configuration automatically excludes:

- Example and template files (`*.example`, `*.template`, `*.sample`)
- Test data directories (`test/`, `tests/`, `fixtures/`)
- Placeholder tokens with repeated characters (e.g., `xxxxxxxxxx`)
- Common placeholder patterns (containing "EXAMPLE", "placeholder", etc.)
- Lines with `# notsecret` comments

### Pre-commit Hooks (Red Hat Employees)

**Installation:**
1. Follow the [rh-pre-commit quickstart guide](https://source.redhat.com/departments/engineering/products/artificial_intelligence_enablement/teams/infosec_ai_security/documentation/leak_pattern_distribution_server)
2. Subscribe to updates for pattern improvements
3. Tokens are valid for 2 years but may need rotation for security incidents

**Usage:**
- Hooks automatically run before each commit
- If a potential leak is detected, the commit will be blocked
- Review the detection and either fix the issue or add to allowlist

### Testing Security Configuration

Use the `rh-gitleaks` tool to test your security configuration:

```bash
# Test current repository
rh-gitleaks --path=. -v

# Test with allowlist configuration
rh-gitleaks --path=. -v --additional-config=.gitleaks.toml

# Test specific branch
rh-gitleaks --path=. -v --branch=your-branch-name

# Test without git (treat as file scan)
rh-gitleaks --path=. -v --no-git
```

## 📋 Security Checklist for Developers

### Before Committing Code

- [ ] No hardcoded API keys, passwords, or tokens
- [ ] All credentials loaded from environment variables
- [ ] `.env` files added to `.gitignore` (already done for this project)
- [ ] Example files use obvious placeholder patterns
- [ ] Security-sensitive placeholder lines have `# notsecret` comments

### Creating New Configuration Files

- [ ] Use `.example`, `.template`, or `.sample` extensions for templates
- [ ] Include clear instructions for copying and customizing
- [ ] Use obvious placeholder values with "EXAMPLE" or similar markers
- [ ] Add security warnings in comments

### When Setting Up Development Environment

- [ ] Copy example configuration files to actual config files
- [ ] Replace all placeholder values with real credentials
- [ ] Verify real config files are excluded by `.gitignore`
- [ ] Test that application loads credentials correctly

### Before Creating Pull Requests

- [ ] Run security scans locally if available
- [ ] Review diff for any accidentally included credentials
- [ ] Verify all placeholder data is obviously fake
- [ ] Check that new files follow naming conventions

## 🚀 CI/CD Security Practices

### Environment Variables in CI

Store all secrets as encrypted environment variables in your CI/CD system:
- GitHub Actions: Repository/Organization secrets
- GitLab CI: Protected variables
- Jenkins: Credential stores

Example GitHub Actions usage:
```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    env:
      HF_TOKEN: ${{ secrets.HF_TOKEN }}
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: pytest tests/
```

### Credential Rotation

Establish a regular credential rotation schedule:
- **API Keys**: Rotate every 6-12 months
- **Service Account Keys**: Rotate annually
- **Passwords**: Follow organizational password policy
- **Certificates**: Monitor expiration dates

## 🆘 Emergency Response

### If Credentials Are Compromised

1. **Immediate action** - Rotate/revoke compromised credentials
2. **Assess scope** - Determine what data/systems may be affected
3. **Contact security team** - Follow organizational incident response
4. **Clean git history** - Remove credentials from repository history
5. **Update security practices** - Prevent similar incidents

### Contact Information

- **Red Hat InfoSec**: infosec@redhat.com
- **Security Alerts**: Include "Leak Pattern Update Request" in subject
- **Emergency Security Issues**: Follow organizational escalation procedures

## 📚 Additional Resources

- [Red Hat InfoSec Leak Pattern Distribution Server](https://source.redhat.com/departments/engineering/products/artificial_intelligence_enablement/teams/infosec_ai_security/documentation/leak_pattern_distribution_server)
- [rh-pre-commit Hook Documentation](https://source.redhat.com/departments/engineering/products/artificial_intelligence_enablement/teams/infosec_ai_security/documentation/leak_pattern_distribution_server)
- [GitLeaks Configuration Reference](https://github.com/zricethezav/gitleaks#configuration)
- [OWASP Secrets Management Guidelines](https://owasp.org/www-community/vulnerabilities/Use_of_hard-coded_credentials)

---

## 📝 Document Maintenance

This document should be updated whenever:
- New security tools are adopted
- Security incidents provide lessons learned
- Red Hat InfoSec guidelines are updated
- New credential types are introduced to the project

**Last Updated:** $(date)
**Document Owner:** Development Team
**Review Cycle:** Quarterly