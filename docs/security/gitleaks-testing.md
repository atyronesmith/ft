# GitLeaks Configuration Testing Guide

This document provides instructions for testing the `.gitleaks.toml` configuration to ensure it properly handles false positives while still detecting real secrets.

## 🧪 Testing Overview

Our `.gitleaks.toml` configuration includes allowlist rules designed to prevent false positives for:
- Example and template files
- Placeholder tokens with obvious patterns
- Test data directories
- Lines with `# notsecret` comments

## 📋 Manual Validation Checklist

Since `rh-gitleaks` is not currently installed, you can manually validate the configuration:

### ✅ Allowlist Rules Verification

Our configuration should allow these patterns:

**1. Placeholder Tokens with Repeated Characters:**
```
hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
api_key_************************
```

**2. Tokens with "EXAMPLE" or Similar Keywords:**
```
hf_EXAMPLE_TOKEN_REPLACE_WITH_REAL
API_KEY="PLACEHOLDER_VALUE_HERE"
secret: "your_secret_here"
```

**3. Lines with Security Comments:**
```python
API_KEY = "fake_key_123"  # notsecret
TOKEN = "test_value"  # not secret
```

**4. Files in Excluded Paths:**
```
./test/credentials.py
./examples/config.yml
./docs/api-keys.md
./fixtures/sample_data.json
```

### ❌ Should Still Detect Real Patterns

The configuration should NOT allow realistic-looking secrets:
```
hf_a1b2c3d4e5f6789012345678901234567890abcd  # Too realistic
sk-a1b2c3d4e5f6789012345678901234567890abcd  # Too realistic
real_api_key_without_obvious_placeholder     # No clear placeholder markers
```

## 🔧 Testing with rh-gitleaks (When Available)

Once you have `rh-gitleaks` installed, test with these commands:

### Basic Configuration Test
```bash
# Test current repository with our configuration
rh-gitleaks --path=. -v --additional-config=.gitleaks.toml
```

### Test Historical False Positive
```bash
# Test specific commit that triggered false positive
rh-gitleaks --path=. --commit-from=cafea21 --commit-to=cafea21 -v --additional-config=.gitleaks.toml
```

### Test Specific Files
```bash
# Test a specific file pattern
rh-gitleaks --path=./test_file.py -v --additional-config=.gitleaks.toml
```

### Test Without Git Context
```bash
# Scan files directly (useful for testing new patterns)
rh-gitleaks --path=. --no-git -v --additional-config=.gitleaks.toml
```

## 🧪 Create Test Cases

You can create temporary test files to validate the configuration:

### Test File 1: Should Be Allowed (Placeholders)
```bash
cat > test_placeholders.py << 'EOF'
# Test file for placeholder patterns
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # notsecret
OPENAI_KEY = "sk-EXAMPLE_KEY_REPLACE_WITH_REAL"  # notsecret
API_SECRET = "your_api_secret_here"  # notsecret
PLACEHOLDER_TOKEN = "PLACEHOLDER_VALUE"  # notsecret
EOF
```

### Test File 2: Should Be Blocked (Realistic)
```bash
cat > test_realistic.py << 'EOF'
# Test file for realistic patterns (should be detected)
HF_TOKEN = "hf_a1b2c3d4e5f6789012345678901234567890"
OPENAI_KEY = "sk-proj1234567890abcdef1234567890abcdef"
API_SECRET = "live_1234567890abcdef1234567890abcdef"
EOF
```

### Test and Clean Up
```bash
# Test both files
rh-gitleaks --path=test_placeholders.py -v --additional-config=.gitleaks.toml
rh-gitleaks --path=test_realistic.py -v --additional-config=.gitleaks.toml

# Clean up test files
rm test_placeholders.py test_realistic.py
```

## 📊 Expected Results

When testing with `rh-gitleaks`:

### ✅ Should PASS (No alerts):
- Files with `.example`, `.template`, `.sample` extensions
- Tokens with 8+ repeated characters (`xxxxxxxx`, `********`)
- Lines containing "EXAMPLE", "placeholder", "your_key_here"
- Lines with `# notsecret` comments
- Files in `test/`, `tests/`, `examples/`, `docs/` directories

### ⚠️ Should ALERT (Detected as potential secrets):
- Realistic-looking API keys without obvious placeholder patterns
- Valid-format tokens without clear "fake" indicators
- Secrets in files outside excluded directories
- Lines without `# notsecret` comments that look like real credentials

## 🔍 Configuration Validation

### Syntax Check
Validate the TOML syntax of our configuration:
```bash
# If you have a TOML validator installed
python -c "import toml; toml.load('.gitleaks.toml'); print('Valid TOML syntax')"
```

### Regex Pattern Testing
Test individual regex patterns from the configuration:
```python
import re

# Test patterns from our .gitleaks.toml
patterns = [
    r'[a-zA-Z0-9_]*[xX]{8,}[a-zA-Z0-9_]*',  # Tokens with 8+ x characters
    r'.*EXAMPLE.*',                           # Lines with EXAMPLE
    r'hf_[xX]{30,}',                         # HF tokens with x placeholders
]

test_strings = [
    "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",  # Should match
    "hf_EXAMPLE_TOKEN_HERE",                   # Should match
    "hf_a1b2c3d4e5f67890abcdef",              # Should NOT match
]

for pattern in patterns:
    print(f"Testing pattern: {pattern}")
    for test_str in test_strings:
        if re.search(pattern, test_str):
            print(f"  ✅ MATCH: {test_str}")
        else:
            print(f"  ❌ NO MATCH: {test_str}")
    print()
```

## 🚨 Troubleshooting

### Common Issues

**Configuration not being loaded:**
- Ensure `.gitleaks.toml` is in repository root
- Check TOML syntax is valid
- Verify you're using `--additional-config=.gitleaks.toml` flag

**False positives still occurring:**
- Check if the pattern matches your allowlist rules
- Add more specific regex patterns
- Consider adding file path exclusions

**Real secrets not being detected:**
- Test without allowlist to ensure base patterns work
- Verify patterns are not too broad
- Check that test secrets look realistic enough

### Getting Help

1. **Validate configuration**: Use TOML syntax checkers
2. **Test patterns**: Use regex testing tools online
3. **Check documentation**: Review [GitLeaks configuration docs](https://github.com/zricethezav/gitleaks#configuration)
4. **Contact InfoSec**: Email infosec@redhat.com for pattern improvement suggestions

## 📝 Maintenance

### Regular Testing

Perform these tests:
- **Weekly**: Quick test with known false positives
- **Monthly**: Full repository scan with rh-gitleaks
- **After pattern updates**: Test new patterns against existing allowlist
- **Before releases**: Comprehensive security scan

### Configuration Updates

Update `.gitleaks.toml` when:
- New false positive patterns are discovered
- New file types or directories need exclusion
- Red Hat InfoSec provides pattern updates
- Team reporting indicates allowlist gaps

---

**Last Updated**: $(date)
**Next Review**: When rh-gitleaks is installed and available
**Owner**: FineTune Security Team