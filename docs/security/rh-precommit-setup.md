# Red Hat Pre-commit Hook Setup Guide

This guide provides step-by-step instructions for Red Hat employees to install and configure the `rh-pre-commit` hook for automated security scanning.

## 🎯 Overview

The `rh-pre-commit` hook prevents committing secrets and sensitive data by scanning your code before each commit. It's specifically designed for Red Hat's security patterns and integrates with the company's leak detection system.

## 🔧 Prerequisites

- Red Hat employee with valid corporate credentials
- Access to Red Hat's internal tools and repositories
- Python 3.7+ installed on your development machine
- Git configured with your Red Hat email

## 📦 Installation Steps

### Step 1: Install rh-pre-commit

Follow the official Red Hat InfoSec quickstart guide:

1. **Access the installation guide**:
   - Visit: [rh-pre-commit quickstart](https://source.redhat.com/departments/engineering/products/artificial_intelligence_enablement/teams/infosec_ai_security/documentation/leak_pattern_distribution_server)
   - Authenticate with your Red Hat credentials

2. **Generate API token**:
   ```bash
   # Follow the token generation process in the quickstart guide
   # Tokens are valid for 2 years
   ```

3. **Install the hook**:
   ```bash
   # Follow the specific installation commands from the quickstart
   # This will install rh-pre-commit globally for all your repos
   ```

### Step 2: Verify Installation

Test that the hook is working correctly:

```bash
# Navigate to this repository
cd /path/to/finetune

# Test the rh-gitleaks command
rh-gitleaks --path=. -v

# Should show scanning results without errors
```

### Step 3: Configure for This Repository

The hook should automatically detect and use our `.gitleaks.toml` configuration file. To verify:

```bash
# Test with our allowlist configuration
rh-gitleaks --path=. -v --additional-config=.gitleaks.toml

# Should respect our allowlist rules and not flag placeholder tokens
```

## 🔍 Testing the Setup

### Test 1: Verify Hook is Active

Try to commit a test file with a fake secret:

```bash
# Create a test file with a fake secret
echo 'API_KEY="hf_realLookingButFakeToken123456789abcdef"' > test_secret.py

# Try to commit (should be blocked)
git add test_secret.py
git commit -m "Test commit with fake secret"

# Should be blocked by pre-commit hook
# Clean up
rm test_secret.py
```

### Test 2: Verify Allowlist Works

Test that our allowlist properly excludes legitimate placeholders:

```bash
# Create a test file with placeholder that should be allowed
echo 'API_KEY="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # notsecret' > test_placeholder.py

# Try to commit (should be allowed)
git add test_placeholder.py
git commit -m "Test commit with placeholder"

# Should be allowed due to allowlist rules
# Clean up
git reset HEAD~1
rm test_placeholder.py
```

## 🛠️ Configuration Details

### How the Hook Works

1. **Pre-commit execution**: Runs before each `git commit`
2. **Pattern matching**: Uses Red Hat's curated security patterns
3. **Allowlist processing**: Respects our `.gitleaks.toml` configuration
4. **Blocking commits**: Prevents commits with potential secrets

### Integration with .gitleaks.toml

Our repository includes a `.gitleaks.toml` file that:
- Excludes example/template files
- Allows placeholder tokens with obvious patterns
- Respects `# notsecret` comments
- Ignores test data directories

The hook automatically uses this configuration when scanning.

### Token Management

API tokens for the hook:
- **Validity**: 2 years from generation
- **Rotation**: May be required for security incidents
- **Storage**: Stored securely by the rh-pre-commit tool
- **Renewal**: Follow the quickstart guide to regenerate

## 🚨 Troubleshooting

### Common Issues

**Issue**: Hook not running on commits
```bash
# Solution: Verify installation
which rh-pre-commit
rh-pre-commit --version

# Re-install if needed following quickstart guide
```

**Issue**: False positives not being filtered
```bash
# Solution: Test allowlist configuration
rh-gitleaks --path=. -v --additional-config=.gitleaks.toml

# Verify .gitleaks.toml is in repository root
ls -la .gitleaks.toml
```

**Issue**: Token expired or authentication errors
```bash
# Solution: Regenerate API token
# Follow token generation steps in quickstart guide
# May need to uninstall and reinstall rh-pre-commit
```

**Issue**: Hook running slowly
```bash
# Check if there are large files being scanned
# Consider adding paths to .gitleaks.toml allowlist
# Contact InfoSec if patterns need optimization
```

### Getting Help

1. **Check documentation**: Review the [official rh-pre-commit docs](https://source.redhat.com/departments/engineering/products/artificial_intelligence_enablement/teams/infosec_ai_security/documentation/leak_pattern_distribution_server)

2. **Test locally**: Use `rh-gitleaks` to debug issues
   ```bash
   # Debug specific files
   rh-gitleaks --path=specific_file.py -v

   # Debug specific branch
   rh-gitleaks --path=. --branch=feature-branch -v

   # Skip git and scan files directly
   rh-gitleaks --path=. --no-git -v
   ```

3. **Contact InfoSec**: Email `infosec@redhat.com` for:
   - Installation issues
   - Pattern update requests
   - False positive reports
   - Token renewal problems

## 📋 Best Practices

### For Development Workflow

1. **Commit frequently**: Hook scans incremental changes more efficiently
2. **Use descriptive commits**: Helps with debugging if issues arise
3. **Test locally**: Use `rh-gitleaks` to check before committing
4. **Keep allowlist current**: Update `.gitleaks.toml` as needed

### For Team Collaboration

1. **Share this guide**: Ensure all team members follow setup
2. **Document exceptions**: Add comments explaining legitimate exceptions
3. **Regular reviews**: Periodically review and update security patterns
4. **Stay informed**: Subscribe to InfoSec updates for pattern changes

## 🔄 Maintenance

### Regular Tasks

- **Monthly**: Check for rh-pre-commit updates
- **Quarterly**: Review and update `.gitleaks.toml` allowlist
- **Annually**: Verify token validity and renew if needed
- **As needed**: Update patterns based on false positive reports

### Updates and Notifications

Subscribe to the InfoSec documentation to receive:
- New pattern releases
- Security tool updates
- Best practice changes
- Critical security notifications

## 🤝 Support

For issues specific to this repository:
- **Repository security**: Check `SECURITY.md` in repository root
- **False positives**: Update `.gitleaks.toml` or contact InfoSec
- **Setup issues**: Follow this guide or contact team leads

For Red Hat InfoSec support:
- **Email**: infosec@redhat.com
- **Subject format**: "rh-pre-commit: [brief description]"
- **Include**: Repository URL, error messages, steps to reproduce

---

**Last Updated**: $(date)
**Owner**: FineTune Development Team
**Review**: Required when InfoSec updates tools or patterns