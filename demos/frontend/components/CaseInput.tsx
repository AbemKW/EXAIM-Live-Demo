'use client';

import { useState, FormEvent, useRef, useEffect } from 'react';
import { useIsProcessing } from '@/store/cdssStore';
import type { CaseRequest, DemoMode, TraceFile } from '@/lib/types';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';

interface CaseInputProps {
  onError?: (message: string) => void;
}

export default function CaseInput({ onError }: CaseInputProps) {
  const [caseText, setCaseText] = useState('');
  const [mode, setMode] = useState<DemoMode>('live_demo');
  const [traces, setTraces] = useState<TraceFile[]>([]);
  const [selectedTrace, setSelectedTrace] = useState<string>('');
  const [loadingTraces, setLoadingTraces] = useState(false);
  const isProcessing = useIsProcessing();
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const formRef = useRef<HTMLFormElement>(null);

  // Fetch available traces when mode changes to trace_replay
  useEffect(() => {
    if (mode === 'trace_replay') {
      fetchTraces();
    }
  }, [mode]);

  const fetchTraces = async () => {
    setLoadingTraces(true);
    try {
      // Auto-detect API URL: use relative path in production, localhost in dev
      const apiUrl = typeof window !== 'undefined' && window.location.hostname !== 'localhost'
        ? '' // Relative URL for production (nginx will route)
        : 'http://localhost:8001';
      
      const response = await fetch(`${apiUrl}/api/traces`);
      if (!response.ok) {
        throw new Error('Failed to fetch traces');
      }
      const data = await response.json();
      
      // Case descriptions mapping
      const caseDescriptions: Record<string, string> = {
        'case-33651373.trace.jsonl.gz': 'Hereditary Spinocerebellar Ataxia presenting with progressive cerebellar ataxia and dysarthria.',
        'case-34895021.trace.jsonl.gz': 'Stage II lung adenocarcinoma in a 40-year-old woman with a significant family history of cancer.',
        'case-34922935.trace.jsonl.gz': 'Duodenal cholesterolosis characterized by multiple yellowish elevated lesions found on endoscopy.',
        'case-34989141.trace.jsonl.gz': 'Achondroplasia defined by disproportionate short stature and bilateral short femurs on ultrasound.',
        'case-35478097.trace.jsonl.gz': 'Bilateral conjunctival papilloma with pinkish nodular lesions unresponsive to anti-inflammatory treatment.',
        'case-35602476.trace.jsonl.gz': 'Autosomal Dominant Hyper-IgE Syndrome with high serum IgE, recurrent infections, and a STAT3 mutation.',
        'case-35795791.trace.jsonl.gz': 'Lennox-Gastaut Syndrome involving multiple seizure types, cognitive impairment, and slow spike-wave EEG patterns.',
        'case-37308247.trace.jsonl.gz': 'Vestibular dysfunction and continuous dizziness likely secondary to a neurodegenerative disorder.',
        'case-37337880.trace.jsonl.gz': 'Apert syndrome featuring craniosynostosis, syndactyly, and distinct facial anomalies.',
        'case-pmc9949831.trace.jsonl.gz': 'Newborn craniofacial anomalies including hypotelorism and a single blind-ended nostril.'
      };
      
      // Add descriptions to traces
      const tracesWithDescriptions = (data.traces || []).map((trace: TraceFile) => ({
        ...trace,
        description: caseDescriptions[trace.file_path.split('/').pop() || ''] || trace.file_path
      }));
      
      setTraces(tracesWithDescriptions);
      // Auto-select first trace if available
      if (tracesWithDescriptions.length > 0 && !selectedTrace) {
        setSelectedTrace(tracesWithDescriptions[0].file_path);
      }
    } catch (error) {
      console.error('Error fetching traces:', error);
    } finally {
      setLoadingTraces(false);
    }
  };

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const newHeight = Math.min(textareaRef.current.scrollHeight, 200);
      textareaRef.current.style.height = `${newHeight}px`;
    }
  }, [caseText]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();

    // Validate based on mode
    if (mode === 'live_demo') {
      const trimmedCase = caseText.trim();
      if (!trimmedCase) {
        alert('Please enter a patient case');
        return;
      }
    } else if (mode === 'trace_replay') {
      if (!selectedTrace) {
        alert('Please select a trace file');
        return;
      }
    }

    try {
      // In production (HF Spaces), use relative URL since nginx routes /api/* to backend
      // In development, use localhost:8000
      const apiUrl = typeof window !== 'undefined' && window.location.hostname !== 'localhost'
        ? '' // Relative URL for production (nginx will route)
        : 'http://localhost:8001';
      
      const requestBody: CaseRequest = {
        mode,
        ...(mode === 'live_demo' ? { case: caseText.trim() } : { trace_file: selectedTrace }),
      };

      console.log('Submitting case request:', { apiUrl, requestBody });

      const response = await fetch(`${apiUrl}/api/process-case`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      console.log('Response status:', response.status);

      if (!response.ok) {
        // Check for vLLM unavailability (503 status)
        if (response.status === 503) {
          const errorMessage = 'AI model service is currently unavailable. Please try again in a moment.';
          if (onError) {
            onError(errorMessage);
          } else {
            alert(errorMessage);
          }
          return;
        }
        
        const error = await response.json();
        throw new Error(error.detail || 'Failed to process case');
      }

      const result = await response.json();
      console.log('Case processed:', result);

      // Clear input after successful submission (only for live_demo)
      if (mode === 'live_demo') {
        setCaseText('');
      }
    } catch (error) {
      console.error('Error processing case:', error);
      alert(`Error: ${(error as Error).message}`);
    }
  };

  return (
    <div className="w-full flex flex-col items-center gap-6">
      {/* Title and description */}
      <div className="text-center">
        <h2 className="text-2xl font-semibold text-foreground mb-2">
          {mode === 'live_demo' ? 'Enter Patient Case' : 'Select Trace File'}
        </h2>
        <p className="text-muted-foreground text-base max-w-lg">
          {mode === 'live_demo' 
            ? "Describe the patient's symptoms, history, and relevant clinical information for AI-powered analysis."
            : "Select a frozen trace file to replay through the system."}
        </p>
      </div>

      {/* Mode selector */}
      <div className="w-full flex flex-col gap-3">
        <label className="text-sm font-medium text-foreground">Mode</label>
        <div className="flex gap-2">
          <Button
            type="button"
            variant={mode === 'live_demo' ? 'default' : 'outline'}
            onClick={() => {
              setMode('live_demo');
              setSelectedTrace('');
            }}
            disabled={isProcessing}
            className="flex-1"
          >
            Live Demo
          </Button>
          <Button
            type="button"
            variant={mode === 'trace_replay' ? 'default' : 'outline'}
            onClick={() => setMode('trace_replay')}
            disabled={isProcessing}
            className="flex-1"
          >
            Trace Replay
          </Button>
        </div>
      </div>

      {/* Trace file selector (only shown in trace_replay mode) */}
      {mode === 'trace_replay' && (
        <div className="w-full flex flex-col gap-3">
          <label htmlFor="trace-select" className="text-sm font-medium text-foreground">
            Trace File
          </label>
          <select
            id="trace-select"
            value={selectedTrace}
            onChange={(e) => setSelectedTrace(e.target.value)}
            disabled={isProcessing || loadingTraces}
            className="w-full px-4 py-3 rounded-xl bg-muted/20 backdrop-blur-xl border border-white/10 focus:border-primary/50 focus:bg-muted/30 focus:ring-primary/50 focus:ring-[3px] transition-all shadow-lg text-foreground disabled:opacity-50 disabled:cursor-not-allowed outline-none custom-select"
          >
            <option value="">Select a case...</option>
            {traces.map((trace) => (
              <option key={trace.file_path} value={trace.file_path}>
                {trace.description || ''}
              </option>
            ))}
          </select>
          {loadingTraces && (
            <p className="text-sm text-muted-foreground">Loading traces...</p>
          )}
        </div>
      )}

      {/* Input form */}
      <form 
        ref={formRef} 
        onSubmit={handleSubmit} 
        className="w-full relative"
      >
        <div className="relative">
          {mode === 'live_demo' && (
            <Textarea
              ref={textareaRef}
              value={caseText}
              onChange={(e) => setCaseText(e.target.value)}
              placeholder="Enter patient case description..."
              rows={4}
              disabled={isProcessing}
              className="resize-none text-base pr-16 py-4 px-5 rounded-2xl bg-muted/20 backdrop-blur-xl border border-white/10 focus:border-primary/50 focus:bg-muted/30 transition-all shadow-lg case-input-textarea glass-morphism"
              style={{ minHeight: '120px', maxHeight: '200px' }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey && !isProcessing && caseText.trim()) {
                  e.preventDefault();
                  formRef.current?.requestSubmit();
                }
              }}
            />
          )}
          {mode === 'live_demo' && (
            <button
              type="submit"
              disabled={isProcessing || !caseText.trim()}
              className="absolute right-4 bottom-4 h-12 w-12 rounded-xl bg-primary/80 backdrop-blur-md hover:bg-primary/90 text-primary-foreground disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl flex items-center justify-center focus:outline-none focus:ring-2 focus:ring-primary/50 z-10 border border-white/20 liquid-button"
            >
              {isProcessing ? (
                <svg className="h-5 w-5 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              ) : (
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
                </svg>
              )}
            </button>
          )}
        </div>
      </form>

      {/* Submit button for trace replay mode */}
      {mode === 'trace_replay' && (
        <Button
          type="button"
          onClick={handleSubmit}
          disabled={isProcessing || !selectedTrace || loadingTraces}
          className="w-full"
        >
          {isProcessing ? (
            <>
              <svg className="h-5 w-5 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Processing...
            </>
          ) : (
            'Replay Trace'
          )}
        </Button>
      )}

      {/* Helper text (only for live_demo) */}
      {mode === 'live_demo' && (
        <p className="text-sm text-muted-foreground/70 text-center">
          Press <kbd className="px-1.5 py-0.5 rounded bg-muted/30 border border-white/10 text-xs font-mono">Enter</kbd> to submit or <kbd className="px-1.5 py-0.5 rounded bg-muted/30 border border-white/10 text-xs font-mono">Shift+Enter</kbd> for new line
        </p>
      )}
    </div>
  );
}

