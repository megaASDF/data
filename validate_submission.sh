#!/bin/bash

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$SCRIPT_DIR/example/input"
OUTPUT_DIR="$SCRIPT_DIR/example/output"
OUTPUT_FILE="$OUTPUT_DIR/submission.csv"
TEAM="your_team"
DOMAIN="222.255.250.24:8001"
IMAGE_NAME="$DOMAIN/$TEAM/submission"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "Step 1: Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo -e "${RED}ERROR: Docker is not installed or not in PATH${NC}"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi
echo -e "${GREEN}✓ Docker is installed${NC}"
echo ""

echo "Step 2: Checking input..."
[ ! -d "$INPUT_DIR" ] && echo -e "${RED}Error: Input directory not found${NC}" && exit 1
[ ! -f "$INPUT_DIR/test.csv" ] && echo -e "${RED}Error: test.csv not found${NC}" && exit 1
echo -e "${GREEN}✓ Input data found${NC}"

mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_FILE"

echo "Step 3: Building Docker image..."
docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"
echo -e "${GREEN}✓ Build successful${NC}"

echo "Step 4: Running inference..."
docker run --rm \
    -v "$INPUT_DIR:/data/input" \
    -v "$OUTPUT_DIR:/data/output" \
    -e INPUT_PATH=/data/input \
    -e OUTPUT_PATH=/data/output \
    "$IMAGE_NAME" > /dev/null
echo -e "${GREEN}✓ Inference complete${NC}"

echo "Step 5: Validating output..."
[ ! -f "$OUTPUT_FILE" ] && echo -e "${RED}Error: Output file not found${NC}" && exit 1

# Check file format
HEADER=$(head -n 1 "$OUTPUT_FILE" | tr -d '\r\n')
if [ "$HEADER" != "Customer_number,Avg_Trans_Amount" ]; then
    echo -e "${RED}Error: Invalid header. Expected 'Customer_number,Avg_Trans_Amount', got '$HEADER'${NC}"
    exit 1
fi

LINE_COUNT=$(wc -l < "$OUTPUT_FILE")
if [ "$LINE_COUNT" -lt 2 ]; then
    echo -e "${RED}Error: Output file is empty or has only header${NC}"
    exit 1
fi

# Validate that all lines have exactly 2 columns
echo "  - Checking column structure..."
INVALID_LINES=$(awk -F',' 'NR>1 && NF!=2 {print NR}' "$OUTPUT_FILE")
if [ ! -z "$INVALID_LINES" ]; then
    echo -e "${RED}Error: Some lines don't have exactly 2 columns: $INVALID_LINES${NC}"
    exit 1
fi

# Validate that Customer_number column contains integers
echo "  - Checking Customer_number format..."
INVALID_CUSTOMER=$(awk -F',' 'NR>1 && $1 !~ /^[0-9]+$/ {print NR": "$1}' "$OUTPUT_FILE")
if [ ! -z "$INVALID_CUSTOMER" ]; then
    echo -e "${RED}Error: Invalid Customer_number values found:${NC}"
    echo "$INVALID_CUSTOMER"
    exit 1
fi

# Validate that Avg_Trans_Amount column contains numbers
echo "  - Checking Avg_Trans_Amount format..."
INVALID_AMOUNT=$(awk -F',' 'NR>1 && $2 !~ /^-?[0-9]+\.?[0-9]*$/ {print NR": "$2}' "$OUTPUT_FILE")
if [ ! -z "$INVALID_AMOUNT" ]; then
    echo -e "${RED}Error: Invalid Avg_Trans_Amount values found:${NC}"
    echo "$INVALID_AMOUNT"
    exit 1
fi

PREDICTION_COUNT=$((LINE_COUNT - 1))
echo -e "${GREEN}✓ Generated $PREDICTION_COUNT predictions${NC}"
echo -e "${GREEN}✓ All validations passed${NC}"

echo -e "${GREEN}---------------------------------------------${NC}"
echo -e "${GREEN}You can submit your Docker image now! ${NC}"
echo -e "${GREEN}---------------------------------------------${NC}"
echo -e "${GREEN}To submit your Docker image, run:${NC}"
echo -e "${GREEN}   docker push $DOMAIN/$TEAM/submission${NC}"
echo -e "${GREEN}---------------------------------------------${NC}"