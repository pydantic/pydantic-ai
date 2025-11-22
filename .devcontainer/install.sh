#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Pydantic AI DevContainer Setup ===${NC}"
echo ""

# Check if INSTALL_MODE is set via environment variable
if [ -n "$INSTALL_MODE" ]; then
    echo -e "${YELLOW}INSTALL_MODE environment variable detected: $INSTALL_MODE${NC}"
    MODE="$INSTALL_MODE"
else
    # Detect if running interactively
    if [ -t 0 ] && [ -t 1 ]; then
        # Interactive mode - prompt user
        echo "Choose installation mode:"
        echo ""
        echo -e "${GREEN}1) Standard${NC} (Recommended)"
        echo "   - Cloud API providers + development tools"
        echo "   - Excludes: Heavy ML frameworks (PyTorch, transformers, vLLM)"
        echo "   - Use case: PR testing, bug fixes, feature development (95% of users)"
        echo ""
        echo -e "${BLUE}2) Full${NC} (ML Development)"
        echo "   - Everything including ML frameworks for local model inference"
        echo "   - Use case: Working on outlines integration, local model features"
        echo ""

        while true; do
            read -p "Enter your choice (1 or 2) [default: 1]: " choice
            choice=${choice:-1}

            case $choice in
                1)
                    MODE="standard"
                    break
                    ;;
                2)
                    MODE="full"
                    break
                    ;;
                *)
                    echo "Invalid choice. Please enter 1 or 2."
                    ;;
            esac
        done
    else
        # Non-interactive mode (agent/CI) - default to standard
        echo -e "${YELLOW}Non-interactive mode detected (agent/CI)${NC}"
        echo "Defaulting to STANDARD installation (excludes heavy ML frameworks)."
        echo "To override, set INSTALL_MODE=full environment variable."
        MODE="standard"
    fi
fi

echo ""
echo -e "${BLUE}Installing in ${MODE^^} mode...${NC}"
echo ""

# Run installation based on mode
if [ "$MODE" = "standard" ]; then
    echo "Installing standard mode (cloud APIs + dev tools, excluding ML frameworks)..."
    uv sync --frozen --group lint --group docs
elif [ "$MODE" = "full" ]; then
    echo "Installing full mode (everything including ML frameworks)..."
    uv sync --frozen --all-extras --all-packages --group lint --group docs
else
    echo -e "${YELLOW}Warning: Unknown mode '$MODE', defaulting to standard${NC}"
    uv sync --frozen --group lint --group docs
fi

# Install pre-commit hooks
echo ""
echo "Installing pre-commit hooks..."
pre-commit install --install-hooks

echo ""
echo -e "${GREEN}✓ Installation complete!${NC}"
echo ""

if [ "$MODE" = "standard" ]; then
    echo "You're using STANDARD mode (ML frameworks excluded)."
    echo "To install ML frameworks later if needed:"
    echo "  make install                    # Install everything"
    echo "  uv sync --frozen --all-extras   # Install all extras for pydantic-ai-slim"
fi

echo ""
echo "Ready to start developing! Try:"
echo "  make test          # Run tests"
echo "  make               # Run all checks"
echo "  make docs-serve    # Serve documentation"
echo ""
