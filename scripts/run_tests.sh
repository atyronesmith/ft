#!/bin/bash

# Test runner script for FineTune
# Usage: ./scripts/run_tests.sh [options]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
VERBOSE=false
COVERAGE=false
SPECIFIC_TEST=""
TEST_TYPE="all"
MARKERS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -t|--test)
            SPECIFIC_TEST="$2"
            shift 2
            ;;
        --unit)
            TEST_TYPE="unit"
            shift
            ;;
        --integration)
            TEST_TYPE="integration"
            shift
            ;;
        --slow)
            MARKERS="${MARKERS} --slow"
            shift
            ;;
        --no-mlx)
            MARKERS="${MARKERS} -m 'not requires_mlx'"
            shift
            ;;
        --no-torch)
            MARKERS="${MARKERS} -m 'not requires_torch'"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose       Show verbose test output"
            echo "  -c, --coverage      Generate coverage report"
            echo "  -t, --test FILE     Run specific test file"
            echo "  --unit             Run only unit tests"
            echo "  --integration      Run only integration tests"
            echo "  --slow             Include slow tests"
            echo "  --no-mlx           Skip tests requiring MLX"
            echo "  --no-torch         Skip tests requiring PyTorch"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h for help"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}FineTune Test Runner${NC}"
echo "===================="

# Check if we're in a Poetry environment
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo -e "${YELLOW}Not in virtual environment. Activating Poetry shell...${NC}"
    eval "$(poetry env info --path)/bin/activate"
fi

# Build pytest command
PYTEST_CMD="poetry run pytest"

# Add test directory based on type
if [[ "$TEST_TYPE" == "unit" ]]; then
    PYTEST_CMD="${PYTEST_CMD} tests/unit"
    echo -e "${BLUE}Running unit tests...${NC}"
elif [[ "$TEST_TYPE" == "integration" ]]; then
    PYTEST_CMD="${PYTEST_CMD} tests/integration"
    echo -e "${BLUE}Running integration tests...${NC}"
elif [[ -n "$SPECIFIC_TEST" ]]; then
    PYTEST_CMD="${PYTEST_CMD} ${SPECIFIC_TEST}"
    echo -e "${BLUE}Running specific test: ${SPECIFIC_TEST}${NC}"
else
    PYTEST_CMD="${PYTEST_CMD} tests/"
    echo -e "${BLUE}Running all tests...${NC}"
fi

# Add verbose flag
if [[ "$VERBOSE" == true ]]; then
    PYTEST_CMD="${PYTEST_CMD} -v"
fi

# Add coverage
if [[ "$COVERAGE" == true ]]; then
    PYTEST_CMD="${PYTEST_CMD} --cov=finetune --cov-report=term-missing --cov-report=html"
    echo -e "${BLUE}Coverage report will be generated...${NC}"
fi

# Add markers
if [[ -n "$MARKERS" ]]; then
    PYTEST_CMD="${PYTEST_CMD} ${MARKERS}"
fi

# Add color output
PYTEST_CMD="${PYTEST_CMD} --color=yes"

# Show test configuration
echo ""
echo "Test Configuration:"
echo "-------------------"
echo "Command: ${PYTEST_CMD}"

# Check for required dependencies
echo ""
echo "Checking dependencies..."

# Check MLX availability
if poetry run python -c "import mlx" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} MLX is available"
else
    echo -e "${YELLOW}⚠${NC} MLX is not available (some tests will be skipped)"
fi

# Check PyTorch availability
if poetry run python -c "import torch" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} PyTorch is available"
else
    echo -e "${YELLOW}⚠${NC} PyTorch is not available (some tests will be skipped)"
fi

# Check Transformers availability
if poetry run python -c "import transformers" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Transformers is available"
else
    echo -e "${YELLOW}⚠${NC} Transformers is not available (some tests will be skipped)"
fi

echo ""
echo "Running tests..."
echo "================"

# Run the tests
if $PYTEST_CMD; then
    echo ""
    echo -e "${GREEN}✓ All tests passed!${NC}"

    # Show coverage report location if generated
    if [[ "$COVERAGE" == true ]]; then
        echo ""
        echo -e "${BLUE}Coverage report generated:${NC}"
        echo "  HTML: htmlcov/index.html"
        echo "  Run 'open htmlcov/index.html' to view in browser"
    fi

    exit 0
else
    echo ""
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
