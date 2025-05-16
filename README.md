# # DEPI Diabetes Detection 🩺

A web-based application for diabetes risk assessment and data analysis using machine learning.

## System Requirements

- Python (3.7 - 3.10 recommended)
- Modern web browser (Chrome, Firefox, Edge)
- Standard computer hardware (no special requirements)

## Installation & Setup (Running Locally)

1. **Clone the Repository** 
   ```bash
   git clone https://github.com/amr21ss/CLS_AI_PROJECT.git
   cd CLS_AI_PROJECT
   cd Phase2
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   # For Windows
   python -m venv venv
   venv\Scripts\activate

   # For macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. **File Paths**:
   - Ensure data files are in the `data/` directory
   - Ensure model files are in the `models/` directory


2. **Streamlit Multi-Page Setup**:
   - Place `input_form.py` and `analysis.py` inside a `pages/` directory in the project root
   - Keep `home.py` in the root directory as the main entry point

## Execution Guide

### Option 1: Access the Deployed Version (Recommended)

Simply visit: 
[https://depiproject-a7txbipsjwvaawrmsftjef.streamlit.app/]

### Option 2: Run Locally

1. Navigate to the project directory
2. Run the application:
   ```bash
   python -m streamlit run home.py   ```
3. Access in browser at `http://localhost:8501`

## Key Files

- `home.py` - Main landing page
- `pages/1_input_form.py` - Diabetes risk assessment tool
- `pages/2_analysis.py` - Data analysis dashboard
- `data/diabetes.csv` - Dataset
- `models/depi_xgb.pkl` - Trained XGBoost model

## Disclaimer

For informational and educational purposes only. NOT a substitute for professional medical advice, diagnosis, or treatment.
