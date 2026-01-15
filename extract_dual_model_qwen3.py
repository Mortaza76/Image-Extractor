"""
Dual-model text extraction using DeepSeek-OCR and Qwen3-VL.
Compares and merges results for improved accuracy.
Then uses Qwen3-VL vision model to parse extracted text into structured JSON
by analyzing both the image and the OCR output.
"""
import requests
import base64
import io
from PIL import Image
import json
from datetime import datetime
import sys
import time
import re
from difflib import SequenceMatcher
import concurrent.futures

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def load_and_prepare_image(image_path: str, max_width: int = 1200):
    """Load and prepare image for OCR."""
    image = Image.open(image_path)
    print(f"Image size: {image.size}")
    
    # Resize if too large
    if image.width > max_width:
        ratio = max_width / image.width
        new_height = int(image.height * ratio)
        print(f"Resizing image to ({max_width}, {new_height}) for faster processing")
        image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
    
    # Convert to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    return image, img_base64

def extract_with_deepseek_ocr(img_base64: str):
    """Extract text from image using DeepSeek-OCR via Ollama."""
    print("\n[DeepSeek-OCR] Starting extraction...")
    start_time = time.time()
    
    prompt = "Perform OCR on this image. Extract all visible text."
    
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "deepseek-ocr",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [img_base64]
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.0
                }
            },
            timeout=120
        )
        
        extraction_time = time.time() - start_time
        
        if response.status_code != 200:
            print(f"[DeepSeek-OCR] Error: API returned status {response.status_code}")
            return None, {"error": f"API error: {response.status_code}", "extraction_time": extraction_time}
        
        result = response.json()
        extracted_text = result.get("message", {}).get("content", "").strip()
        
        if not extracted_text:
            extracted_text = result.get("response", "").strip()
        
        metadata = {
            "model": "deepseek-ocr",
            "extraction_time": round(extraction_time, 2),
            "character_count": len(extracted_text),
            "line_count": len(extracted_text.split('\n')),
            "status": "success"
        }
        
        print(f"[DeepSeek-OCR] Extraction complete in {extraction_time:.2f}s ({len(extracted_text)} chars)")
        return extracted_text, metadata
        
    except Exception as e:
        extraction_time = time.time() - start_time
        print(f"[DeepSeek-OCR] Error: {e}")
        return None, {"error": str(e), "extraction_time": extraction_time, "status": "failed"}

def extract_with_qwen3_vl(img_base64: str):
    """Extract text from image using Qwen3-VL via Ollama."""
    print("\n[Qwen3-VL] Starting extraction...")
    start_time = time.time()
    
    prompt = """Extract all text from this document image. 
Return the text exactly as it appears, preserving:
- Line breaks
- Spacing
- All numbers, letters, and special characters
- Field labels and their corresponding values

Do not add any interpretation or context. Extract only what is visible in the image."""
    
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "qwen3-vl:8b",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [img_base64]
                    }
                ],
                "stream": False
            },
            timeout=180
        )
        
        extraction_time = time.time() - start_time
        
        if response.status_code != 200:
            print(f"[Qwen3-VL] Error: API returned status {response.status_code}")
            return None, {"error": f"API error: {response.status_code}", "extraction_time": extraction_time}
        
        result = response.json()
        extracted_text = result.get("message", {}).get("content", "").strip()
        
        if not extracted_text:
            extracted_text = result.get("response", "").strip()
        
        metadata = {
            "model": "qwen3-vl:8b",
            "extraction_time": round(extraction_time, 2),
            "character_count": len(extracted_text),
            "line_count": len(extracted_text.split('\n')),
            "status": "success"
        }
        
        print(f"[Qwen3-VL] Extraction complete in {extraction_time:.2f}s ({len(extracted_text)} chars)")
        return extracted_text, metadata
        
    except Exception as e:
        extraction_time = time.time() - start_time
        print(f"[Qwen3-VL] Error: {e}")
        return None, {"error": str(e), "extraction_time": extraction_time, "status": "failed"}

def normalize_text(text: str):
    """Normalize text for comparison."""
    if not text:
        return ""
    # Remove extra whitespace but preserve line breaks
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join([line for line in lines if line])

def calculate_similarity(text1: str, text2: str):
    """Calculate similarity between two texts."""
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1, text2).ratio()

def compare_extractions(deepseek_text: str, qwen_text: str):
    """Compare two extraction results."""
    print("\n[COMPARISON] Analyzing results...")
    
    # Normalize texts
    ds_normalized = normalize_text(deepseek_text)
    qw_normalized = normalize_text(qwen_text)
    
    # Split into lines
    ds_lines = [line for line in ds_normalized.split('\n') if line.strip()]
    qw_lines = [line for line in qw_normalized.split('\n') if line.strip()]
    
    # Calculate similarity
    similarity = calculate_similarity(ds_normalized, qw_normalized)
    
    # Find common lines (exact matches)
    common_lines = []
    unique_to_deepseek = []
    unique_to_qwen = []
    
    ds_set = set(ds_lines)
    qw_set = set(qw_lines)
    
    common_lines = list(ds_set & qw_set)
    unique_to_deepseek = list(ds_set - qw_set)
    unique_to_qwen = list(qw_set - ds_set)
    
    # Find similar lines (fuzzy matches)
    similar_lines = []
    for ds_line in unique_to_deepseek[:]:
        for qw_line in unique_to_qwen[:]:
            if calculate_similarity(ds_line, qw_line) > 0.8:
                similar_lines.append({
                    "deepseek": ds_line,
                    "qwen": qw_line,
                    "similarity": calculate_similarity(ds_line, qw_line)
                })
                if ds_line in unique_to_deepseek:
                    unique_to_deepseek.remove(ds_line)
                if qw_line in unique_to_qwen:
                    unique_to_qwen.remove(qw_line)
    
    # Calculate agreement percentage
    total_unique_lines = len(common_lines) + len(unique_to_deepseek) + len(unique_to_qwen)
    agreement = (len(common_lines) / total_unique_lines * 100) if total_unique_lines > 0 else 0
    
    comparison = {
        "similarity_score": round(similarity * 100, 2),
        "agreement_percentage": round(agreement, 2),
        "common_lines": common_lines,
        "unique_to_deepseek": unique_to_deepseek,
        "unique_to_qwen": unique_to_qwen,
        "similar_lines": similar_lines,
        "total_lines_deepseek": len(ds_lines),
        "total_lines_qwen": len(qw_lines),
        "common_count": len(common_lines),
        "unique_deepseek_count": len(unique_to_deepseek),
        "unique_qwen_count": len(unique_to_qwen)
    }
    
    print(f"[COMPARISON] Similarity: {comparison['similarity_score']}%")
    print(f"[COMPARISON] Agreement: {comparison['agreement_percentage']}%")
    print(f"[COMPARISON] Common lines: {len(common_lines)}")
    print(f"[COMPARISON] Unique to DeepSeek-OCR: {len(unique_to_deepseek)}")
    print(f"[COMPARISON] Unique to Qwen3-VL: {len(unique_to_qwen)}")
    
    return comparison

def resolve_conflict(deepseek_value: str, qwen_value: str):
    """Resolve conflict between two values using priority rules."""
    # Rule 1: Length-based (prefer longer, more complete)
    if len(qwen_value) > len(deepseek_value) * 1.2:
        return qwen_value, "qwen3-vl (longer/more complete)"
    elif len(deepseek_value) > len(qwen_value) * 1.2:
        return deepseek_value, "deepseek-ocr (longer/more complete)"
    
    # Rule 2: Format validation (dates, IDs, etc.)
    # Check for date format
    date_pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'
    if re.search(date_pattern, qwen_value) and not re.search(date_pattern, deepseek_value):
        return qwen_value, "qwen3-vl (better date format)"
    elif re.search(date_pattern, deepseek_value) and not re.search(date_pattern, qwen_value):
        return deepseek_value, "deepseek-ocr (better date format)"
    
    # Check for ID/Number format (with dashes or letters)
    id_pattern = r'[\dA-Z-]+'
    if re.match(id_pattern, qwen_value) and len(qwen_value) > len(deepseek_value):
        return qwen_value, "qwen3-vl (more complete ID)"
    elif re.match(id_pattern, deepseek_value) and len(deepseek_value) > len(qwen_value):
        return deepseek_value, "deepseek-ocr (more complete ID)"
    
    # Rule 3: Default - prefer Qwen3-VL for structure, DeepSeek-OCR for exact text
    # For now, prefer the one with more context (usually Qwen3-VL)
    if len(qwen_value) >= len(deepseek_value):
        return qwen_value, "qwen3-vl (default - better structure)"
    else:
        return deepseek_value, "deepseek-ocr (default - more precise)"

def merge_extractions(deepseek_text: str, qwen_text: str, comparison: dict):
    """Intelligently merge two extraction results."""
    print("\n[MERGING] Combining results...")
    
    merged_lines = []
    conflicts_resolved = []
    
    # Step 1: Add common elements (high confidence) - these are most reliable
    merged_lines.extend(comparison['common_lines'])
    
    # Step 2: Resolve similar lines (conflicts) - use priority rules
    for similar in comparison['similar_lines']:
        resolved, reason = resolve_conflict(similar['deepseek'], similar['qwen'])
        merged_lines.append(resolved)
        conflicts_resolved.append({
            "deepseek": similar['deepseek'],
            "qwen": similar['qwen'],
            "resolved": resolved,
            "reason": reason
        })
    
    # Step 3: Add unique elements from both models (one model may catch what other misses)
    # Filter out lines that are too similar to already merged lines
    all_merged = set(merged_lines)
    for line in comparison['unique_to_deepseek']:
        # Check if this line is significantly different from already merged lines
        is_unique = True
        for merged_line in all_merged:
            if calculate_similarity(line, merged_line) > 0.85:
                is_unique = False
                break
        if is_unique:
            merged_lines.append(line)
            all_merged.add(line)
    
    for line in comparison['unique_to_qwen']:
        # Check if this line is significantly different from already merged lines
        is_unique = True
        for merged_line in all_merged:
            if calculate_similarity(line, merged_line) > 0.85:
                is_unique = False
                break
        if is_unique:
            merged_lines.append(line)
            all_merged.add(line)
    
    # Remove exact duplicates while preserving order
    seen = set()
    unique_merged = []
    for line in merged_lines:
        line_lower = line.lower().strip()
        if line_lower not in seen and line.strip():
            seen.add(line_lower)
            unique_merged.append(line)
    
    merged_text = '\n'.join(unique_merged)
    
    print(f"[MERGING] Merged {len(unique_merged)} unique lines")
    print(f"[MERGING] Resolved {len(conflicts_resolved)} conflicts")
    
    return merged_text, conflicts_resolved

def assess_quality(merged_text: str, comparison: dict):
    """Assess the quality of merged extraction."""
    print("\n[QUALITY] Assessing extraction quality...")
    
    # Completeness: based on line count (relative to both models)
    max_lines = max(comparison['total_lines_deepseek'], comparison['total_lines_qwen'])
    merged_lines = len([line for line in merged_text.split('\n') if line.strip()])
    completeness = (merged_lines / max_lines * 100) if max_lines > 0 else 0
    
    # Agreement score
    agreement = comparison['agreement_percentage']
    
    # Similarity score
    similarity = comparison['similarity_score']
    
    # Confidence score (weighted combination)
    confidence = (completeness * 0.4) + (agreement * 0.3) + (similarity * 0.3)
    
    # Quality flags
    quality_flags = []
    if agreement < 70:
        quality_flags.append("Low agreement between models (<70%)")
    if completeness < 80:
        quality_flags.append("Low completeness (<80%)")
    if confidence < 80:
        quality_flags.append("Low confidence score (<80%)")
    
    quality = {
        "completeness_score": round(completeness, 2),
        "agreement_score": round(agreement, 2),
        "similarity_score": round(similarity, 2),
        "confidence_score": round(confidence, 2),
        "quality_flags": quality_flags
    }
    
    print(f"[QUALITY] Completeness: {quality['completeness_score']}%")
    print(f"[QUALITY] Agreement: {quality['agreement_score']}%")
    print(f"[QUALITY] Confidence: {quality['confidence_score']}%")
    if quality_flags:
        print(f"[QUALITY] Flags: {', '.join(quality_flags)}")
    
    return quality

def detect_document_type(text: str):
    """Detect document type from extracted text."""
    text_lower = text.lower()
    
    # Check for driver license indicators
    if any(keyword in text_lower for keyword in ['driver license', 'drivers license', 'dl ', 'dob', 'exp', 'hgt', 'wgt', 'sex']):
        return "driver_license"
    
    # Check for insurance card indicators
    if any(keyword in text_lower for keyword in ['insurance', 'member id', 'group number', 'rxbin', 'rxpcn', 'deductible', 'copay']):
        return "insurance_card"
    
    # Default to generic document
    return "document"

def post_process_structured_data(structured_data: dict):
    """Post-process structured data to fix common OCR errors."""
    if not structured_data or not isinstance(structured_data, dict):
        return structured_data
    
    # Fix weight field: "Ib" → "lb" (common OCR error)
    if "personal_info" in structured_data:
        personal_info = structured_data["personal_info"]
        if "weight" in personal_info and personal_info["weight"]:
            weight = str(personal_info["weight"])
            # Fix "Ib" → "lb" when it appears after numbers
            if " Ib" in weight or weight.endswith("Ib"):
                weight = weight.replace(" Ib", " lb").replace("Ib", "lb")
                personal_info["weight"] = weight
    
    # Fix last_name: If it looks like it came from a signature, try to find the official name
    # This is a fallback - the prompt should handle it, but this adds extra safety
    if "personal_info" in structured_data:
        personal_info = structured_data["personal_info"]
        # If last_name contains common signature patterns, we might want to flag it
        # But we'll let the vision model handle it primarily through the improved prompt
    
    return structured_data

def parse_with_qwen3_vl(img_base64: str, merged_text: str, deepseek_text: str, qwen_text: str, document_type: str = None):
    """Parse merged text into structured JSON using Qwen3-VL 2B vision model.
    
    Qwen3-VL 2B will:
    1. Look at the original image
    2. Review the extracted text from both OCR models
    3. Detect document type from the image
    4. Generate structured JSON output
    
    Note: Using 2B model for faster processing. Trade-off: slightly lower accuracy than 8B model.
    """
    print("\n[QWEN3-VL PARSING] Parsing text into structured JSON using Qwen3-VL 2B vision model...")
    start_time = time.time()
    
    # Auto-detect document type if not provided
    if not document_type:
        document_type = detect_document_type(merged_text)
    
    print(f"[QWEN3-VL PARSING] Detected document type: {document_type}")
    
    # Create prompt based on document type
    if document_type == "driver_license":
        prompt = f"""You are analyzing a DRIVER LICENSE document. Look at the image and the extracted OCR text below.

This image is a driver license. Two OCR models (DeepSeek-OCR and Qwen3-VL) have extracted text from this image.

EXTRACTED TEXT FROM BOTH MODELS:
DeepSeek-OCR extracted:
{deepseek_text}

Qwen3-VL extracted:
{qwen_text}

MERGED EXTRACTED TEXT:
{merged_text}

TASK: 
1. Look at the IMAGE to understand the document structure and layout
2. Review the EXTRACTED TEXT from both models
3. Parse the text into structured JSON format
4. Use the image to verify and correct any OCR errors
5. Understand the document layout to properly map fields

IMPORTANT PARSING RULES:
1. NAME FIELDS: Look for official name fields with labels/codes like:
   - "1 SAMPLE" or "* SAMPLE" = Last name (OFFICIAL)
   - "2 SUZY A" or "* SUZY A" = First name (OFFICIAL)
   - ALWAYS prioritize these labeled fields over signature lines
   - Signature lines (like "Suzy A. Sampson") are for signature field only, NOT for name extraction
   - If you see both "1 SAMPLE" and a signature with "Sampson", use "SAMPLE" as last_name
   - Use the IMAGE to verify which is the official name field

2. ADDRESS PARSING: Split address lines properly:
   - "8123 STREET ADDRESS" = address field
   - "YOUR CITY WA 99999-1234" = Split into: city="YOUR CITY", state="WA", zip_code="99999-1234"
   - If you see a line like "YOUR CITY WA 99999-1234", extract:
     * city = "YOUR CITY" (everything before the state code)
     * state = "WA" (the 2-letter state code)
     * zip_code = "99999-1234" (the numbers after state)
   - The city is the text between the address and the state code
   - Use the IMAGE to verify the address structure

3. EYE COLOR: Look for field codes like "18 EYES BLU" and extract "BLU" as eye_color
   - Use the IMAGE to verify eye color if visible

4. SIGNATURE: Extract handwritten signatures separately from name fields
   - Use the IMAGE to identify the signature area

5. FIELD CODES: Pay attention to field codes (numbers/letters before values):
   - "3 DOB" = Date of Birth
   - "4a ISS" = Issue Date
   - "4b EXP" = Expiration Date
   - "15 SEX" = Sex
   - "16 HGT" = Height
   - "17 WGT" = Weight
   - "18 EYES" = Eye Color
   - "12 RESTRICTIONS" = Restrictions
   - "9a END" = Endorsements
   - Use the IMAGE to verify field positions and values

IMPORTANT PARSING RULES:
1. NAME FIELDS: Look for official name fields with labels/codes like:
   - "1 SAMPLE" or "* SAMPLE" = Last name (OFFICIAL)
   - "2 SUZY A" or "* SUZY A" = First name (OFFICIAL)
   - ALWAYS prioritize these labeled fields over signature lines
   - Signature lines (like "Suzy A. Sampson") are for signature field only, NOT for name extraction
   - If you see both "1 SAMPLE" and a signature with "Sampson", use "SAMPLE" as last_name

2. ADDRESS PARSING: Split address lines properly:
   - "8123 STREET ADDRESS" = address field
   - "YOUR CITY WA 99999-1234" = Split into: city="YOUR CITY", state="WA", zip_code="99999-1234"
   - If you see a line like "YOUR CITY WA 99999-1234", extract:
     * city = "YOUR CITY" (everything before the state code)
     * state = "WA" (the 2-letter state code)
     * zip_code = "99999-1234" (the numbers after state)
   - The city is the text between the address and the state code

3. EYE COLOR: Look for field codes like "18 EYES BLU" and extract "BLU" as eye_color

4. SIGNATURE: Extract handwritten signatures separately from name fields

5. FIELD CODES: Pay attention to field codes (numbers/letters before values):
   - "3 DOB" = Date of Birth
   - "4a ISS" = Issue Date
   - "4b EXP" = Expiration Date
   - "15 SEX" = Sex
   - "16 HGT" = Height
   - "17 WGT" = Weight
   - "18 EYES" = Eye Color
   - "12 RESTRICTIONS" = Restrictions
   - "9a END" = Endorsements

Return a JSON object with this exact structure (use null for missing fields):
{{
  "document_type": "driver_license",
  "license_info": {{
    "license_number": "...",
    "class": "...",
    "issue_date": "...",
    "expiration_date": "...",
    "restrictions": "...",
    "endorsements": "..."
  }},
  "personal_info": {{
    "last_name": "...",
    "first_name": "...",
    "middle_name": "...",
    "date_of_birth": "...",
    "sex": "...",
    "height": "...",
    "weight": "...",
    "eye_color": "...",
    "address": "...",
    "city": "...",
    "state": "...",
    "zip_code": "..."
  }},
  "organ_donor": {{
    "donor_id": "...",
    "donor_status": "..."
  }},
  "state": "...",
  "signature": "..."
}}

Extract all visible information. Preserve exact values as they appear in the text. Return ONLY valid JSON, no additional text."""
    
    elif document_type == "insurance_card":
        prompt = f"""You are analyzing an INSURANCE CARD document. Look at the image and the extracted OCR text below.

This image is an insurance card. Two OCR models (DeepSeek-OCR and Qwen3-VL) have extracted text from this image.

EXTRACTED TEXT FROM BOTH MODELS:
DeepSeek-OCR extracted:
{deepseek_text}

Qwen3-VL extracted:
{qwen_text}

MERGED EXTRACTED TEXT:
{merged_text}

TASK:
1. Look at the IMAGE to understand the document structure and layout
2. Review the EXTRACTED TEXT from both models
3. Parse the text into structured JSON format
4. Use the image to verify and correct any OCR errors
5. Understand the table structure (In-Network vs Out-of-Network columns)
6. Map values correctly to their table positions

Return a JSON object with this exact structure (use null for missing fields):
{{
  "document_type": "insurance_card",
  "header": {{
    "insurance_provider": "...",
    "company": "..."
  }},
  "member_info": {{
    "name": "...",
    "member_id": "..."
  }},
  "plan_info": {{
    "group_number": "...",
    "service_type": "...",
    "care_type": "..."
  }},
  "prescription_info": {{
    "rx_bin": "...",
    "rx_pcn": "...",
    "rx_group": "..."
  }},
  "copays": {{
    "office_visit": "...",
    "emergency_room": "...",
    "urgent_care": "..."
  }},
  "deductibles": {{
    "individual": {{
      "in_network": "...",
      "out_network": "..."
    }},
    "family": {{
      "in_network": "...",
      "out_network": "..."
    }}
  }},
  "out_of_pocket_maximums": {{
    "individual": {{
      "in_network": "...",
      "out_network": "..."
    }},
    "family": {{
      "in_network": "...",
      "out_network": "..."
    }}
  }}
}}

Pay attention to table structure. The card may have a table with 'In Ntwk' and 'Out Ntwk' columns. Map values correctly to their columns and rows (Individual vs Family). Extract all visible information. Return ONLY valid JSON, no additional text."""
    
    else:
        prompt = f"""You are analyzing a document. Look at the image and the extracted OCR text below.

Two OCR models (DeepSeek-OCR and Qwen3-VL) have extracted text from this image.

EXTRACTED TEXT FROM BOTH MODELS:
DeepSeek-OCR extracted:
{deepseek_text}

Qwen3-VL extracted:
{qwen_text}

MERGED EXTRACTED TEXT:
{merged_text}

TASK:
1. Look at the IMAGE to understand the document type and structure
2. Review the EXTRACTED TEXT from both models
3. Parse the text into structured JSON format
4. Use the image to verify and correct any OCR errors
5. Organize information into appropriate fields based on document type

Return a JSON object with a logical structure based on the document content. Extract all visible information and organize it into appropriate fields. Return ONLY valid JSON, no additional text."""
    
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "qwen3-vl:2b",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [img_base64]  # Pass the image so Qwen3-VL can see it
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.1  # Low temperature for consistent parsing
                }
            },
            timeout=300  # Reduced timeout for faster 2B model (5 minutes should be sufficient)
        )
        
        parsing_time = time.time() - start_time
        
        if response.status_code != 200:
            print(f"[QWEN3-VL PARSING] Error: API returned status {response.status_code}")
            return None, {"error": f"API error: {response.status_code}", "parsing_time": parsing_time}
        
        result = response.json()
        parsed_text = result.get("message", {}).get("content", "").strip()
        
        if not parsed_text:
            parsed_text = result.get("response", "").strip()
        
        # Try to extract JSON from the response (might have markdown code blocks)
        # Remove markdown code blocks if present
        if "```json" in parsed_text:
            parsed_text = parsed_text.split("```json")[1].split("```")[0].strip()
        elif "```" in parsed_text:
            parsed_text = parsed_text.split("```")[1].split("```")[0].strip()
        
        # Try to parse JSON
        try:
            structured_data = json.loads(parsed_text)
            
            # Post-process to fix common OCR errors
            structured_data = post_process_structured_data(structured_data)
            
            print(f"[QWEN3-VL PARSING] Successfully parsed into structured JSON ({parsing_time:.2f}s)")
            return structured_data, {
                "model": "qwen3-vl:2b",
                "parsing_time": round(parsing_time, 2),
                "status": "success",
                "document_type": document_type
            }
        except json.JSONDecodeError as e:
            print(f"[QWEN3-VL PARSING] Warning: Failed to parse JSON response: {e}")
            print(f"[QWEN3-VL PARSING] Raw response: {parsed_text[:500]}...")
            # Return the raw text as fallback
            return {"raw_text": parsed_text, "parse_error": str(e)}, {
                "model": "qwen3-vl:2b",
                "parsing_time": round(parsing_time, 2),
                "status": "json_parse_error",
                "document_type": document_type
            }
        
    except requests.exceptions.Timeout:
        parsing_time = time.time() - start_time
        print(f"[QWEN3-VL PARSING] Parsing timed out after 300 seconds (5 minutes)")
        return None, {"model": "qwen3-vl:2b", "status": "failed", "error": "Timeout", "parsing_time": parsing_time}
    except Exception as e:
        parsing_time = time.time() - start_time
        print(f"[QWEN3-VL PARSING] Error during parsing: {e}")
        import traceback
        traceback.print_exc()
        return None, {"model": "qwen3-vl:2b", "status": "failed", "error": str(e), "parsing_time": parsing_time}

def extract_with_dual_models(image_path: str):
    """Extract text using both DeepSeek-OCR and Qwen3-VL, then merge results."""
    print("="*80)
    print("DUAL-MODEL TEXT EXTRACTION")
    print("Using DeepSeek-OCR + Qwen3-VL")
    print("="*80)
    
    # Load and prepare image
    image, img_base64 = load_and_prepare_image(image_path)
    
    # Extract with both models (can run in parallel)
    print("\n[EXTRACTION] Running both models...")
    start_time = time.time()
    
    # Run both extractions
    deepseek_text, deepseek_meta = extract_with_deepseek_ocr(img_base64)
    qwen_text, qwen_meta = extract_with_qwen3_vl(img_base64)
    
    total_time = time.time() - start_time
    
    # Handle failures
    if not deepseek_text and not qwen_text:
        print("\n[ERROR] Both models failed!")
        return None
    
    if not deepseek_text:
        print("\n[WARNING] DeepSeek-OCR failed, using Qwen3-VL result only")
        # Parse with Qwen3-VL vision model even with single model result
        structured_data, parsing_meta = parse_with_qwen3_vl(img_base64, qwen_text, "", qwen_text)
        return {
            "extraction_timestamp": datetime.now().isoformat(),
            "models_used": {
                "qwen3_vl": qwen_meta,
                "vision_parser": {"model": "qwen3-vl:2b"}
            },
            "extractions": {"qwen3_vl": {"raw_text": qwen_text, **qwen_meta}},
            "merged_result": {"raw_text": qwen_text, "source": "qwen3-vl-only"},
            "structured_data": structured_data if structured_data else None,
            "parsing_metadata": parsing_meta,
            "metadata": {"total_extraction_time": round(total_time, 2), "image_size": f"{image.size[0]}x{image.size[1]}", "processing_mode": "single_model_with_vision_parsing"}
        }, image
    
    if not qwen_text:
        print("\n[WARNING] Qwen3-VL failed, using DeepSeek-OCR result only")
        # Parse with Qwen3-VL vision model even with single model result
        structured_data, parsing_meta = parse_with_qwen3_vl(img_base64, deepseek_text, deepseek_text, "")
        return {
            "extraction_timestamp": datetime.now().isoformat(),
            "models_used": {
                "deepseek_ocr": deepseek_meta,
                "vision_parser": {"model": "qwen3-vl:2b"}
            },
            "extractions": {"deepseek_ocr": {"raw_text": deepseek_text, **deepseek_meta}},
            "merged_result": {"raw_text": deepseek_text, "source": "deepseek-ocr-only"},
            "structured_data": structured_data if structured_data else None,
            "parsing_metadata": parsing_meta,
            "metadata": {"total_extraction_time": round(total_time, 2), "image_size": f"{image.size[0]}x{image.size[1]}", "processing_mode": "single_model_with_vision_parsing"}
        }, image
    
    # Compare results
    comparison = compare_extractions(deepseek_text, qwen_text)
    
    # Merge results
    merged_text, conflicts = merge_extractions(deepseek_text, qwen_text, comparison)
    
    # Assess quality
    quality = assess_quality(merged_text, comparison)
    
    # Parse merged text into structured JSON using Qwen3-VL vision model
    # Qwen3-VL will look at the image AND the extracted text to generate structured JSON
    structured_data, parsing_meta = parse_with_qwen3_vl(img_base64, merged_text, deepseek_text, qwen_text)
    
    # Build output structure
    output = {
        "extraction_timestamp": datetime.now().isoformat(),
        "models_used": {
            "deepseek_ocr": {"model": "deepseek-ocr:latest"},
            "qwen3_vl": {"model": "qwen3-vl:8b"},
            "vision_parser": {"model": "qwen3-vl:2b"}
        },
        "extractions": {
            "deepseek_ocr": {
                "raw_text": deepseek_text,
                **deepseek_meta
            },
            "qwen3_vl": {
                "raw_text": qwen_text,
                **qwen_meta
            }
        },
        "comparison": comparison,
        "merged_result": {
            "raw_text": merged_text,
            "confidence_score": quality["confidence_score"],
            "completeness_score": quality["completeness_score"],
            "source": "merged",
            "conflicts_resolved": conflicts
        },
        "structured_data": structured_data if structured_data else None,
        "parsing_metadata": parsing_meta,
        "quality": quality,
        "metadata": {
            "total_extraction_time": round(total_time, 2),
            "image_size": f"{image.size[0]}x{image.size[1]}",
            "processing_mode": "dual_model_with_vision_parsing"
        }
    }
    
    return output, image

def visualize_results(image, output_data):
    """Visualize the dual-model extraction results."""
    print("\n" + "="*80)
    print("DUAL-MODEL EXTRACTION RESULTS")
    print("="*80)
    
    print(f"\n[IMAGE] Image Information:")
    print(f"   Size: {image.size[0]} x {image.size[1]} pixels")
    
    print(f"\n[EXTRACTIONS] Individual Model Results:")
    print("-" * 80)
    
    if "deepseek_ocr" in output_data.get("extractions", {}):
        ds = output_data["extractions"]["deepseek_ocr"]
        print(f"\n[DeepSeek-OCR]")
        print(f"   Status: {ds.get('status', 'unknown')}")
        print(f"   Time: {ds.get('extraction_time', 0)}s")
        print(f"   Characters: {ds.get('character_count', 0)}")
        print(f"   Lines: {ds.get('line_count', 0)}")
    
    if "qwen3_vl" in output_data.get("extractions", {}):
        qw = output_data["extractions"]["qwen3_vl"]
        print(f"\n[Qwen3-VL]")
        print(f"   Status: {qw.get('status', 'unknown')}")
        print(f"   Time: {qw.get('extraction_time', 0)}s")
        print(f"   Characters: {qw.get('character_count', 0)}")
        print(f"   Lines: {qw.get('line_count', 0)}")
    
    print(f"\n[COMPARISON] Model Comparison:")
    print("-" * 80)
    comp = output_data.get("comparison", {})
    print(f"   Similarity: {comp.get('similarity_score', 0)}%")
    print(f"   Agreement: {comp.get('agreement_percentage', 0)}%")
    print(f"   Common lines: {comp.get('common_count', 0)}")
    print(f"   Unique to DeepSeek-OCR: {comp.get('unique_deepseek_count', 0)}")
    print(f"   Unique to Qwen3-VL: {comp.get('unique_qwen_count', 0)}")
    
    print(f"\n[MERGED RESULT] Final Combined Text:")
    print("=" * 80)
    merged = output_data.get("merged_result", {})
    print(merged.get("raw_text", ""))
    print("=" * 80)
    
    print(f"\n[STRUCTURED DATA] Parsed JSON:")
    print("=" * 80)
    structured = output_data.get("structured_data")
    if structured:
        print(json.dumps(structured, indent=2, ensure_ascii=False))
    else:
        print("No structured data available (parsing may have failed)")
    print("=" * 80)
    
    parsing_meta = output_data.get("parsing_metadata", {})
    if parsing_meta:
        print(f"\n[PARSING] Qwen3-VL Vision Parsing Metadata:")
        print(f"   Model: {parsing_meta.get('model', 'unknown')}")
        print(f"   Status: {parsing_meta.get('status', 'unknown')}")
        print(f"   Time: {parsing_meta.get('parsing_time', 0)}s")
        print(f"   Document Type: {parsing_meta.get('document_type', 'unknown')}")
        if parsing_meta.get('error'):
            print(f"   Error: {parsing_meta.get('error')}")
    
    print(f"\n[QUALITY] Quality Assessment:")
    print("-" * 80)
    quality = output_data.get("quality", {})
    print(f"   Completeness: {quality.get('completeness_score', 0)}%")
    print(f"   Agreement: {quality.get('agreement_score', 0)}%")
    print(f"   Confidence: {quality.get('confidence_score', 0)}%")
    
    if quality.get("quality_flags"):
        print(f"   Flags: {', '.join(quality['quality_flags'])}")
    
    if merged.get("conflicts_resolved"):
        print(f"\n[CONFLICTS] Resolved {len(merged['conflicts_resolved'])} conflicts")
        for i, conflict in enumerate(merged['conflicts_resolved'][:3], 1):
            print(f"   {i}. {conflict['reason']}")
            print(f"      DeepSeek: {conflict['deepseek'][:50]}...")
            print(f"      Qwen3-VL: {conflict['qwen'][:50]}...")
            print(f"      Resolved: {conflict['resolved'][:50]}...")
    
    print(f"\n[JSON] Full JSON Output:")
    print("-" * 80)
    print(json.dumps(output_data, indent=2, ensure_ascii=False)[:500] + "...")
    print("-" * 80)
    
    # Save results
    output_file = "dual_model_extraction.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n[SUCCESS] Results saved to: {output_file}")

def main():
    import sys
    
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "Driver License 2.png"  # Default to driver license
    
    print(f"\n[INPUT] Processing image: {image_path}")
    
    # Extract with dual models
    result = extract_with_dual_models(image_path)
    
    if result is None:
        print("\n[ERROR] Extraction failed!")
        return
    
    output_data, image = result
    
    # Visualize results
    visualize_results(image, output_data)
    
    print("\n" + "="*80)
    print("[SUCCESS] Dual-model extraction with Qwen3-VL vision parsing complete!")
    print("="*80)

if __name__ == "__main__":
    main()

