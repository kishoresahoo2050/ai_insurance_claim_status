#!/usr/bin/env bash
# Run the InsureAI Streamlit app

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "🛡️  Starting InsureAI – Insurance Claim Assistant"
echo "=================================================="

# Generate synthetic data if not present
if [ ! -f "data/synthetic/claims.json" ]; then
  echo "📊 Generating synthetic claim data..."
  python data/synthetic/generate_claims.py
fi

# Create processed dir
mkdir -p data/processed

# Launch Streamlit
echo "🚀 Launching Streamlit UI..."
echo "   Open: http://localhost:8501"
echo ""
echo "   Set your API key first:"
echo "   export GOOGLE_API_KEY=your_key_here"
echo ""

streamlit run src/ui/app.py \
  --server.port 8501 \
  --server.headless true \
  --browser.gatherUsageStats false
