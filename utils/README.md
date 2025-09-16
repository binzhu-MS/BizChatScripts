# Utils Directory - Utility Functions

This directory contains core utility functions and modules for the BizChatScripts project. These utilities provide common functionality that can be imported and used across different parts of the project.

## 📁 **Utility Modules**

### 📊 **`file_reader.py`**
**Universal file reader for multiple document formats**

A comprehensive utility that can read content from PDF, Word, Excel, PowerPoint, and text files with robust fallback mechanisms.

**Supported Formats:**
- **PDF** (.pdf) - PyPDF2, PyMuPDF fallback
- **Word** (.docx, .doc) - python-docx, docx2txt, COM automation
- **Excel** (.xlsx, .xls, .xlsb) - Multiple engines with extensive fallback
- **PowerPoint** (.pptx, .ppt) - python-pptx, COM automation  
- **Text** (.txt, .md, .rtf) - Encoding detection

**Usage:**
```python
from utils.file_reader import UniversalFileReader

reader = UniversalFileReader()
result = reader.read_file("document.pdf")
content = result['content']  # Ready for LLM processing
```

**Key Features:**
- Multiple fallback methods for maximum file compatibility
- Structured output with metadata
- Fire CLI integration for command-line usage
- Perfect for LLM content processing pipelines

---

### 📝 **`json_utils.py`**
**JSON file and data handling utilities**

Provides robust functions for reading and writing JSON files with proper error handling.

**Key Functions:**
- `read_json_file(file_path)` - Read JSON files with error handling
- `write_json_file(data, file_path)` - Write data to JSON files
- `validate_json_data()` - Validate JSON structure
- `merge_json_objects()` - Merge multiple JSON objects

**Usage:**
```python
from utils.json_utils import read_json_file, write_json_file

# Read JSON data
data = read_json_file("data.json")

# Write JSON data  
write_json_file(processed_data, "results.json")
```

**Features:**
- UTF-8 encoding support
- Comprehensive error handling
- Type hints for better development experience
- Validation and merging capabilities

---

### 📊 **`statistics_utils.py`**
**Statistical analysis and metrics calculation**

Provides statistical functions for analyzing experimental data, including t-tests and proportion tests.

**Key Functions:**
- `tsingle()` - One-sample t-test
- `tpaired()` - Paired t-test
- `tindep()` - Independent t-test
- `tdiff()` - Calculate statistical differences
- `prop_test()` - Proportion tests

**Usage:**
```python
from utils.statistics_utils import tdiff, tsingle

# Calculate statistical difference
result = tdiff(control_values, experiment_values)

# Perform one-sample t-test
test_result = tsingle(values, mean=0)
```

**Features:**
- Built on scipy.stats and statsmodels
- Comprehensive statistical testing
- Proper p-value calculations
- Support for different test alternatives

---

### 📄 **`markdown_reports.py`**
**Markdown and HTML report generation**

Utilities for generating formatted reports and tables in Markdown and HTML formats.

**Key Functions:**
- `format_table()` - Generate HTML tables from data
- `format_facet_grouped_table()` - Group data by facets in tables
- `create_measurement_reports()` - Generate measurement reports
- `format_metrics_display()` - Format metrics for display

**Usage:**
```python
from utils.markdown_reports import format_table

# Generate HTML table
html_table = format_table(measurements, headers=['Metric', 'Value'])
```

**Features:**
- HTML table generation with proper escaping
- Facet-based grouping for complex data
- Customizable formatting options
- Integration with measurement systems

---

## 🚀 **Usage Patterns**

### **Importing Utilities**
All utilities follow the same import pattern used throughout the project:

```python
# Import from utils package (recommended)
from utils.file_reader import UniversalFileReader
from utils.json_utils import read_json_file
from utils.statistics_utils import tdiff

# Alternative: Import specific functions
from utils.file_reader import UniversalFileReader
```

### **Example Integration**
Here's how utilities are typically used together:

```python
from utils.file_reader import UniversalFileReader
from utils.json_utils import write_json_file
from utils.statistics_utils import tdiff

# Read document content
reader = UniversalFileReader()
result = reader.read_file("data.xlsx")

# Process and analyze
if result['status'] == 'success':
    # Save results
    write_json_file(result, "processed_data.json")
    
    # Perform statistical analysis if applicable
    # analysis = tdiff(control_data, experiment_data)
```

## 📦 **Dependencies**

### **Core Dependencies** (always required)
- `fire` - Command-line interface generation
- `pandas` - Data manipulation (for file_reader)
- `pathlib` - Path handling

### **Optional Dependencies** (for specific utilities)
- **file_reader.py**: `PyPDF2`, `python-docx`, `docx2txt`, `python-pptx`, `chardet`, `pywin32`
- **statistics_utils.py**: `scipy`, `statsmodels`
- **json_utils.py**: No additional dependencies

### **Installation**
All dependencies are included in the project's `requirements.txt`:
```bash
pip install -r requirements.txt
```

## 🔧 **Development Guidelines**

### **Adding New Utilities**
When adding new utility modules:

1. **Follow naming convention**: Use descriptive names ending in `_utils.py`
2. **Include docstrings**: Document all functions with clear parameters and return types
3. **Add type hints**: Use proper type annotations for better IDE support
4. **Handle errors**: Include comprehensive error handling
5. **Update this README**: Add documentation for new utilities

### **Import Patterns**
Use the established import pattern from other tools:
```python
# At the top of files that use utilities
from utils.module_name import function_name
```

### **Testing**
Test utilities using the test files in the `tests/` directory:
```bash
python tests/test_file_reader.py
```

## 🎯 **Integration with Project Structure**

The utils directory integrates with the overall project structure:

```
BizChatScripts/
├── utils/                    # ← Core utilities (this directory)
│   ├── file_reader.py       # File reading utility
│   ├── json_utils.py        # JSON handling
│   ├── statistics_utils.py  # Statistical functions
│   └── markdown_reports.py  # Report generation
├── examples/                 # Usage examples
├── tests/                    # Test files
├── tools/                    # Project-specific tools
└── llms/                     # LLM framework
```

**Usage from different directories:**
- **From tools/**: `from utils.file_reader import UniversalFileReader`
- **From examples/**: `from utils.json_utils import read_json_file`
- **From project root**: `from utils.statistics_utils import tdiff`

This ensures consistent, reusable utility functions across the entire project! 🚀
