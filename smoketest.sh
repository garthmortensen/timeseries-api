#!/bin/bash
# smoketest.sh - API Smoke Test Script for Time Series Pipeline API

# Text formatting
# \033 is the escape character, the [ is the start of the color code, 0 is the style, 32 is the color code for green.
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

API_PORT=8001
BASE_URL="http://localhost:${API_PORT}"

echo -e "${YELLOW}=== Time Series Pipeline API Smoke Test ===${NC}"
echo -e "${YELLOW}Running tests against: ${BASE_URL}${NC}\n"

# Function to run a test and check for response code
run_test() {
  local endpoint=$1
  local description=$2
  local payload=$3
  
  echo -e "${YELLOW}Testing ${endpoint} - ${description}...${NC}"
  
  # Run the curl command and capture both status code and response
  response=$(curl -s -w "\n%{http_code}" -X POST "${BASE_URL}${endpoint}" \
    -H "Content-Type: application/json" \
    -d "${payload}")
  
  # Extract the status code from the last line of the response
  status_code=$(echo "$response" | tail -n1)
  # Extract the response body (everything except the last line)
  response_body=$(echo "$response" | sed '$d')
  
  # Check if the status code indicates success (200-299)
  if [[ $status_code -ge 200 && $status_code -lt 300 ]]; then
    echo -e "${GREEN}✓ Success: Status code ${status_code}${NC}"
    # Truncate the response if it's too long
    if [[ ${#response_body} -gt 500 ]]; then
      echo -e "Response (truncated): ${response_body:0:500}...\n"
    else
      echo -e "Response: ${response_body}\n"
    fi
  else
    echo -e "${RED}✗ Failed: Status code ${status_code}${NC}"
    echo -e "Error response: ${response_body}\n"
    return 1
  fi
  
  return 0
}

# Test 1: Generate Data
echo -e "${YELLOW}Test 1: Generate Data${NC}"
test_1_payload='{
  "start_date": "2023-01-01",
  "end_date": "2023-01-10",
  "anchor_prices": {
    "GME": 150.0,
    "BYND": 200.0,
    "BP": 15.0
  }
}'
run_test "/api/v1/generate_data" "Generate synthetic time series data" "$test_1_payload"
test_1_result=$?

# Test 2: Scale Data
echo -e "${YELLOW}Test 2: Scale Data${NC}"
test_2_payload='{
  "method": "standardize",
  "data": [
    {"date": "2023-01-01", "price": 100},
    {"date": "2023-01-02", "price": 101},
    {"date": "2023-01-03", "price": 102},
    {"date": "2023-01-04", "price": 103},
    {"date": "2023-01-05", "price": 104}
  ]
}'
run_test "/api/v1/scale_data" "Scale time series data" "$test_2_payload"
test_2_result=$?

# Test 3: Test Stationarity
echo -e "${YELLOW}Test 3: Test Stationarity${NC}"
test_3_payload='{
  "data": [
    {"date": "2023-01-01", "price": 100},
    {"date": "2023-01-02", "price": 101},
    {"date": "2023-01-03", "price": 102},
    {"date": "2023-01-04", "price": 103},
    {"date": "2023-01-05", "price": 104}
  ]
}'
run_test "/api/v1/test_stationarity" "Test for stationarity" "$test_3_payload"
test_3_result=$?

# Test 4: Run ARIMA
echo -e "${YELLOW}Test 4: Run ARIMA Model${NC}"
test_4_payload='{
  "p": 1,
  "d": 1,
  "q": 1,
  "data": [
    {"date": "2023-01-01", "price": 100},
    {"date": "2023-01-02", "price": 101},
    {"date": "2023-01-03", "price": 102},
    {"date": "2023-01-04", "price": 103},
    {"date": "2023-01-05", "price": 104},
    {"date": "2023-01-06", "price": 105},
    {"date": "2023-01-07", "price": 106},
    {"date": "2023-01-08", "price": 107},
    {"date": "2023-01-09", "price": 108},
    {"date": "2023-01-10", "price": 109}
  ]
}'
run_test "/api/v1/run_arima" "Run ARIMA model on time series" "$test_4_payload"
test_4_result=$?

# Test 5: Run GARCH
echo -e "${YELLOW}Test 5: Run GARCH Model${NC}"
test_5_payload='{
  "p": 1,
  "q": 1,
  "data": [
    {"date": "2023-01-01", "price": 100},
    {"date": "2023-01-02", "price": 101},
    {"date": "2023-01-03", "price": 102},
    {"date": "2023-01-04", "price": 103},
    {"date": "2023-01-05", "price": 104},
    {"date": "2023-01-06", "price": 105},
    {"date": "2023-01-07", "price": 106},
    {"date": "2023-01-08", "price": 107},
    {"date": "2023-01-09", "price": 108},
    {"date": "2023-01-10", "price": 109}
  ]
}'
run_test "/api/v1/run_garch" "Run GARCH model on time series" "$test_5_payload"
test_5_result=$?

# Test 6: Run Complete Pipeline
echo -e "${YELLOW}Test 6: Run Complete Pipeline${NC}"
test_6_payload='{
  "start_date": "2023-01-01",
  "end_date": "2023-01-10",
  "anchor_prices": {
    "GME": 150.0,
    "BYND": 200.0,
    "BP": 15.0
  },
  "scaling_method": "standardize",
  "arima_params": {
    "p": 1,
    "d": 1,
    "q": 1
  },
  "garch_params": {
    "p": 1,
    "q": 1,
    "dist": "t"
  }
}'
run_test "/api/v1/run_pipeline" "Execute the entire pipeline" "$test_6_payload"
test_6_result=$?

# Summary
echo -e "${YELLOW}=== Test Summary ===${NC}"
total_tests=6
passed_tests=0

for test_result in $test_1_result $test_2_result $test_3_result $test_4_result $test_5_result $test_6_result; do
  if [[ $test_result -eq 0 ]]; then
    passed_tests=$((passed_tests+1))
  fi
done

echo -e "Passed: ${GREEN}$passed_tests${NC} / Total: $total_tests"

if [[ $passed_tests -eq $total_tests ]]; then
  echo -e "${GREEN}All tests passed!${NC}"
  exit 0
else
  echo -e "${RED}Some tests failed. Check the logs above for details.${NC}"
  exit 1
fi