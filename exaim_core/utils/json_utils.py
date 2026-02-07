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


def extract_json_with_cot_fallback(text: str) -> Optional[dict]:
    """Robust JSON extractor that strips Chain-of-Thought traces.

    This function extends `extract_json_from_text` behavior by specifically
    removing CoT traces that commonly appear with MedGemma/vLLM outputs
    (for example text prefixed with `<unused94>` or the literal word
    "thought"), handling Markdown code blocks, and then locating the
    first valid JSON object using brace-matching as a fallback.

    Rules implemented:
    - Remove any explicit tag-wrapped content like `<unusedNN>...` that
      either closes (`</unusedNN>`) or continues up to the first JSON
      object.
    - Remove any leading or in-line CoT sections that start with the word
      "thought" (case-insensitive) up to the first JSON brace.
    - Prefer JSON inside Markdown fences (```json ... ```), falling back
      to the first balanced JSON object using brace matching.
    """
    if text is None:
        return None

    # Work on a copy
    cleaned = str(text)
    original_text = cleaned  # Keep for fallback debugging

    # 1) Remove tag-wrapped reasoning blocks: <unused123>...</unused123>
    cleaned = re.sub(r'<unused\d+>.*?</unused\d+>', '', cleaned, flags=re.DOTALL)

    # 2) If there's an opening tag with no explicit close, drop everything
    # from the tag up to the first JSON brace (to remove "<unused94>thought ... { ...")
    cleaned = re.sub(r'<unused\d+>.*?(?=\{)', '', cleaned, flags=re.DOTALL)

    # 3) Remove CoT traces that begin with the word 'thought' (case-insensitive)
    #    up to the first JSON brace. This covers patterns like "thought: ... {"
    cleaned = re.sub(r'(?i)\bthought\b[:\s-]*.*?(?=\{)', '', cleaned, flags=re.DOTALL)

    # 4) Remove any stray single-line markers starting with 'thought' that
    #    weren't caught above (defensive)
    cleaned = re.sub(r'(?im)^\s*thought[:\s-]?.*$', '', cleaned)

    # 5) Try to extract from ```json code fences first
    json_match = re.search(r'```(?:json)?\s*\n?(\{.*?\})\s*\n?```', cleaned, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 6) Brace-matching: find the first balanced JSON object starting at the
    #    first '{' after cleaning. This is robust to nested braces and strings.
    start_idx = cleaned.find('{')
    if start_idx != -1:
        brace_count = 0
        in_string = False
        escape_next = False

        for i in range(start_idx, len(cleaned)):
            char = cleaned[i]

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
                        candidate = cleaned[start_idx:i+1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break

    # 7) Regex fallback: first {.*} match (less precise)
    regex_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if regex_match:
        try:
            return json.loads(regex_match.group())
        except json.JSONDecodeError:
            pass

    # 8) ULTRA AGGRESSIVE: Try to find JSON in the ORIGINAL text by looking for
    #    patterns like "rationale", "stream_state", "is_complete" etc. which
    #    are BufferAnalysis field names
    if '"rationale"' in original_text or '"stream_state"' in original_text:
        # Find the first { after any of these field names
        for field in ['"rationale"', '"stream_state"', '"is_complete"', '"is_relevant"', '"is_novel"']:
            field_pos = original_text.find(field)
            if field_pos != -1:
                # Search backward to find the opening {
                brace_pos = original_text.rfind('{', 0, field_pos)
                if brace_pos != -1:
                    # Use brace matching from this position
                    brace_count = 0
                    in_string = False
                    escape_next = False
                    for i in range(brace_pos, len(original_text)):
                        char = original_text[i]
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
                                    candidate = original_text[brace_pos:i+1]
                                    try:
                                        return json.loads(candidate)
                                    except json.JSONDecodeError:
                                        break
                break

    # 9) Last resort: delegate to older extractor which has additional heuristics
    try:
        return extract_json_from_text(text)
    except Exception:
        return None
