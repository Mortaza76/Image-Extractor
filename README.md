# Image Extractor - Dual-Model OCR & Document Parsing Pipeline

A sophisticated, production-ready document processing system that leverages cutting-edge AI models to extract and structure text from document images with exceptional accuracy.

## 🎯 Overview

This project implements a dual-model OCR architecture that combines **DeepSeek-OCR** and **Qwen3-VL** for text extraction, then intelligently merges and parses the results into structured JSON. The system supports multiple document types including driver licenses, insurance cards, and generic documents.

### Key Features

- ✅ **Dual-Model OCR**: Combines DeepSeek-OCR and Qwen3-VL for superior accuracy
- ✅ **Intelligent Merging**: Smart conflict resolution between model outputs
- ✅ **Structured Parsing**: Converts extracted text to structured JSON
- ✅ **Quality Assessment**: Comprehensive quality metrics and confidence scoring
- ✅ **Multi-Language Support**: Handles 50+ languages
- ✅ **Vision-Enhanced Parsing**: Optional vision model for error correction
- ✅ **Error Handling**: Graceful degradation and robust error management

## 📋 Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Demo Results](#demo-results)
- [Document Types](#document-types)
- [Performance](#performance)
- [Technical Details](#technical-details)

## 🏗️ Architecture

The system uses a multi-stage processing pipeline:

1. **Image Preprocessing**: Automatic resizing and format optimization
2. **Dual-Model OCR Extraction**: Parallel execution of two OCR engines
3. **Intelligent Comparison**: Advanced similarity analysis and conflict detection
4. **Smart Merging**: Rule-based conflict resolution with priority algorithms
5. **Quality Assessment**: Multi-dimensional quality scoring system
6. **Structured Parsing**: AI-powered conversion to structured JSON format
7. **Post-Processing**: Error correction and data normalization

### Two Implementation Variants

#### 1. `extract_dual_model.py` (Text-Only Parsing)
- **Parser**: Mistral 7B (Text-only LLM)
- **Speed**: ⭐⭐⭐⭐⭐ Faster (~5-6 minutes)
- **Use Case**: High-volume batch processing, excellent OCR quality

#### 2. `extract_dual_model_qwen3.py` ⭐ **RECOMMENDED**
- **Parser**: Qwen3-VL 2B (Vision-Language Model)
- **Accuracy**: ⭐⭐⭐⭐⭐ Superior (can verify OCR against image)
- **Use Case**: Production systems, complex layouts, highest accuracy requirements

## 🚀 Installation

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running locally
- Required Ollama models:
  ```bash
  ollama pull deepseek-ocr
  ollama pull qwen3-vl:8b
  ollama pull qwen3-vl:2b  # For vision-enhanced parsing
  ollama pull mistral:7b    # For text-only parsing
  ```

### Install Dependencies

```bash
pip install requests pillow
```

## 💻 Usage

### Basic Usage

```bash
# Using vision-enhanced parser (recommended)
python extract_dual_model_qwen3.py "driver license 3.png"

# Using text-only parser (faster)
python extract_dual_model.py "driver license 3.png"
```

### Command Line Arguments

```bash
python extract_dual_model_qwen3.py <image_path>
```

If no image path is provided, it defaults to `"driver license 3.png"`.

### Output

The script generates:
- Console output with detailed extraction metrics
- `dual_model_extraction.json` - Complete extraction results with structured data

## 📊 Demo Results

### Example 1: Driver License Extraction

**Input Image:**
![Driver License 3](driver%20license%203.png)

*Note: The image above shows the actual driver license that was processed to generate the extraction results below.*

**Extraction Results:**

#### DeepSeek-OCR Output:
```
New York State  
USA  
Driver's License  

ID 123 456 789  
Class D  

MOTORIST  
MICHELLE, MARIE  
2345 ANYWHERE STREET  
ALBANY, NY 12222  

DCE 10/31/1990  
Issued 03/07/2022  
Expires 10/31/2029  

E NONE  
R NONE  

Sex F  Height 5'-08"  Eyes BRO  

Organ  
Donor
```

#### Qwen3-VL Output:
```
New York State USA  
DRIVER LICENSE  
123 456 789  
Class D  
MOTORIST  
MICHELLE, MARIE  
2345 ANYWHERE STREET ALBANY, NY 12222  
DOB 10/31/1990  
issued 03/07/2022  
Expires 10/31/2029  
R NONE  
R NONE  
Sex F Height 5'0" Evs BRO  
Michelle M. Michael  
OCT 90  
Organ Donor  
NYS
```

#### Quality Metrics:
- **Similarity Score**: 76.51%
- **Agreement Percentage**: 27.78%
- **Confidence Score**: 85.4%
- **Completeness Score**: 135.29%

#### Merged Result:
The system intelligently combines both extractions, resolving conflicts and preserving unique information from each model.

#### Structured JSON Output:
```json
{
  "document_type": "driver_license",
  "license_info": {
    "license_number": "123 456 789",
    "class": "D",
    "issue_date": "03/07/2022",
    "expiration_date": "10/31/2029"
  },
  "personal_info": {
    "last_name": "MICHELLE",
    "first_name": "MARIE",
    "date_of_birth": "10/31/1990",
    "sex": "F",
    "height": "5'-08\"",
    "eye_color": "BRO",
    "address": "2345 ANYWHERE STREET",
    "city": "ALBANY",
    "state": "NY",
    "zip_code": "12222"
  }
}
```

### Example 2: Insurance Card Extraction

**Input Image:**
![Insurance Card 1](insurance%20card%201.png)

*Note: The image above shows a sample insurance card that was processed by the system.*

**Extraction Results:**

The system extracts:
- Insurance provider information
- Member ID and group number
- Prescription information (RX BIN, RX PCN)
- Copay information
- Deductibles (Individual and Family, In-Network and Out-of-Network)
- Out-of-pocket maximums

See `insurance_card_extraction.json` for complete structured output.

## 📄 Document Types

### Supported Documents

1. **Driver Licenses**
   - License number, class, dates
   - Personal information (name, DOB, address)
   - Physical characteristics (height, weight, eye color)
   - Restrictions and endorsements
   - Organ donor information

2. **Insurance Cards**
   - Member information
   - Plan details
   - Prescription information
   - Copays and deductibles
   - Out-of-pocket maximums

3. **Generic Documents**
   - Flexible JSON structure
   - Automatic field organization
   - Extensible schema

## ⚡ Performance

### Processing Times

- **DeepSeek-OCR**: ~21 seconds average
- **Qwen3-VL OCR**: ~142 seconds average
- **Total OCR Time**: ~163 seconds for dual extraction
- **Text-Only Parsing**: ~3 minutes total
- **Vision-Enhanced Parsing**: ~15-20 minutes for complex documents

### Accuracy Metrics

- **Similarity Score**: Typically 70-90% between models
- **Agreement Percentage**: Varies by document quality (typically 30-80%)
- **Confidence Score**: Typically 80-95% for good quality documents
- **Completeness**: Often exceeds 100% due to complementary model strengths

## 🔧 Technical Details

### Models Used

- **DeepSeek-OCR**: Specialized OCR model for text extraction
- **Qwen3-VL 8B**: Vision-language model for OCR and parsing
- **Qwen3-VL 2B**: Faster vision model for parsing (recommended)
- **Mistral 7B**: Text-only LLM for parsing (alternative)

### Key Algorithms

1. **Text Comparison**: SequenceMatcher for fuzzy text matching
2. **Conflict Resolution**: Multi-rule priority system
3. **Quality Assessment**: Weighted scoring (40% completeness, 30% agreement, 30% similarity)
4. **Document Type Detection**: Keyword-based classification

### Error Handling

- Graceful degradation if one model fails
- Configurable timeouts for each processing stage
- Comprehensive error catching and reporting
- Fallback mechanisms for single-model operation

## 📁 Project Structure

```
text-extraction-pipeline/
├── extract_dual_model.py          # Text-only parsing implementation
├── extract_dual_model_qwen3.py     # Vision-enhanced parsing (recommended)
├── dual_model_extraction.json      # Sample extraction results
├── insurance_card_extraction.json  # Insurance card results
├── PROJECT_REPORT.md               # Comprehensive project documentation
├── README.md                       # This file
├── driver license 3.png          # Sample driver license (used in demo)
├── insurance card 1.png           # Sample insurance card
└── ...                            # Additional sample images
```

## 🌍 Multi-Language Support

The system supports **50+ languages** including:
- English
- Chinese (Simplified and Traditional)
- Japanese
- Korean
- Arabic
- And many more...

Full Unicode support with UTF-8 encoding throughout the pipeline.

## 📈 Use Cases

- **Financial Services**: KYC document processing
- **Healthcare**: Insurance card and medical record digitization
- **Government**: ID verification and document processing
- **Legal**: Contract and document analysis
- **Real Estate**: Property document processing
- **HR**: Employee document management

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available for use.

## 🙏 Acknowledgments

- **DeepSeek** for the OCR model
- **Qwen** for the vision-language models
- **Mistral AI** for the text parsing model
- **Ollama** for local LLM serving infrastructure

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Built with ❤️ using state-of-the-art AI models**
