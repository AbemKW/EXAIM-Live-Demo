export type DiffType = 'added' | 'removed' | 'equal';

export interface DiffResult {
  type: DiffType;
  value: string;
}

/**
 * A simple character-based diffing algorithm.
 * For production use, a library like 'diff' would be better.
 * This is a basic implementation for the purpose of the demo.
 */
export function diffChars(oldStr: string, newStr: string): DiffResult[] {
  const results: DiffResult[] = [];
  
  // Basic optimization: if strings are equal, return a single 'equal' result
  if (oldStr === newStr) {
    return [{ type: 'equal', value: newStr }];
  }

  // Simple Myers-ish diff or LCS could be complex to implement from scratch perfectly.
  // We'll use a simpler approach for character-by-character diffing:
  // We'll find common prefixes and suffixes and then mark the middle as replaced.
  // Actually, for a better "glow" effect, we want to see what exactly changed.
  
  let prefixLen = 0;
  while (prefixLen < oldStr.length && prefixLen < newStr.length && oldStr[prefixLen] === newStr[prefixLen]) {
    prefixLen++;
  }
  
  if (prefixLen > 0) {
    results.push({ type: 'equal', value: oldStr.substring(0, prefixLen) });
  }
  
  let suffixLen = 0;
  while (suffixLen < (oldStr.length - prefixLen) && 
         suffixLen < (newStr.length - prefixLen) && 
         oldStr[oldStr.length - 1 - suffixLen] === newStr[newStr.length - 1 - suffixLen]) {
    suffixLen++;
  }
  
  const removed = oldStr.substring(prefixLen, oldStr.length - suffixLen);
  const added = newStr.substring(prefixLen, newStr.length - suffixLen);
  
  if (removed.length > 0) {
    results.push({ type: 'removed', value: removed });
  }
  
  if (added.length > 0) {
    results.push({ type: 'added', value: added });
  }
  
  if (suffixLen > 0) {
    results.push({ type: 'equal', value: oldStr.substring(oldStr.length - suffixLen) });
  }
  
  return results;
}
