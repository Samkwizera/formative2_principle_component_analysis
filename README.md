# Principal Component Analysis (PCA) - Formative Assignment

## African Economic Crisis Dataset Analysis

This project implements Principal Component Analysis (PCA) from scratch using NumPy to analyze the African Economic, Banking, and Systemic Crisis dataset. The implementation demonstrates dimensionality reduction while preserving variance in economic indicators across 13 African countries from 1860 to 2014.

## Dataset

**African Economic, Banking and Systemic Crisis Data**
- **Source**: [Kaggle - African Crises Dataset](https://www.kaggle.com/datasets/chirin/africa-economic-banking-and-systemic-crisis-data)
- **Countries**: 13 African nations (Algeria, Angola, Central African Republic, Egypt, Ivory Coast, Kenya, Mauritius, Morocco, Nigeria, South Africa, Tunisia, Zambia, Zimbabwe)
- **Time Period**: 1860 - 2014
- **Total Records**: 1,059 observations
- **Features**: 14 columns including:
  - Country identifiers (cc3, country)
  - Temporal data (year)
  - Crisis indicators (systemic_crisis, banking_crisis, currency_crises, inflation_crises)
  - Economic metrics (exch_usd, inflation_annual_cpi, gdp_weighted_default)
  - Debt indicators (domestic_debt_in_default, sovereign_external_debt_default)
  - Political factors (independence)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook or VS Code with Jupyter extension

###  Clone the Repository
```bash
git clone https://github.com/Samkwizera/formative2_principle_component_analysis.git
cd formative2_principle_component_analysis
```

###  Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

###  Install Required Libraries
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

###  Download the Dataset
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/chirin/africa-economic-banking-and-systemic-crisis-data)
2. Extract `african_crises.csv` to the `archive (1)` folder in the project directory
3. Ensure the file path is: `archive (1)/african_crises.csv`

---

## Required Libraries

Create a `requirements.txt` file with:
```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
jupyter>=1.0.0
```

## Usage

### Running the Notebook

1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open the notebook**:
   - Navigate to `Template_PCA_Formative_1[Peer_Pair_Number] (1).ipynb`

3. **Run cells sequentially**:
   - Click "Run All" or execute cells one by one using `Shift + Enter`


## Citations

- **Dataset**: Reinhart, C. and Rogoff, K. (2010). "This Time is Different: Eight Centuries of Financial Folly"
- **Source**: Kaggle - African Economic Crisis Dataset
- **Methodology**: Principal Component Analysis (Pearson, 1901; Hotelling, 1933)
