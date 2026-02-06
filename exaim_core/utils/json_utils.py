"""JSON utility functions for extracting and parsing JSON from LLM outputs."""

import json
import re
from typing import Optional


def extract_json_from_text(text: str) -> Optional[dict]:
    """Extract JSON object from text output.
    
    Tries multiple strategies:
    1. Clean special tokens and thought markers
    2. Find JSON between ```json markers
    3. Find first complete JSON object in text
    4. Regex fallback to extract JSON block
    5. Extract from markdown code blocks
    
    Args:
        text: Text content that may contain JSON
        
    Returns:
        Parsed JSON dictionary if found, None otherwise
    """
    # Pre-processing: Remove special tokens and thought markers that may appear before JSON
    # Common patterns: <unused94>, <thought>, <identity>, etc.
    text = re.sub(r'<unused\d+>', '', text)
    text = re.sub(r'</?thought>', '', text)
    text = re.sub(r'</?identity>', '', text)
    text = re.sub(r'</?reasoning>', '', text)
    
    # Strategy 1: Look for ```json ... ``` blocks
    json_match = re.search(r'```(?:json)?\s*\n?(\{.*?\})\s*\n?```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Strategy 2: Find first complete JSON object using brace matching
    start_idx = text.find('{')
    if start_idx != -1:
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i in range(start_idx, len(text)):
            char = text[i]
            
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"':
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = text[start_idx:i+1]
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            pass
    
    # Strategy 3: Regex fallback - extract largest JSON-like block
    # This catches cases where brace matching fails due to nested structures
    regex_match = re.search(r'\{.*\}', text, re.DOTALL)
    if regex_match:
        try:
            return json.loads(regex_match.group())
        except json.JSONDecodeError:
            pass
    
    # Strategy 4: Last resort - find first { and last } to construct a block
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace:last_brace+1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    
    return None
