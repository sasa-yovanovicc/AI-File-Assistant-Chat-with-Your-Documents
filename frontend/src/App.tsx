import React, { useState } from 'react';

interface ChatResponseSource { source: string; chunk_index: number; score?: number; preview?: string }
interface ChatResponse { question: string; answer: string; sources: ChatResponseSource[]; prompt_preview?: string; kw_coverage?: number; confidence?: string; reason?: string }

// v2 API types
interface V2ChatResponse { 
  query: { text: string; max_results: number };
  result: {
    answer: string;
    confidence: { level: string; description: string };
    sources: ChatResponseSource[];
    metadata: { architecture: string };
  };
}

const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export default function App() {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [resp, setResp] = useState<ChatResponse | null>(null);
  const [k, setK] = useState(3);
  const [minScore, setMinScore] = useState<string>('');
  const [useLLM, setUseLLM] = useState(true);
  const [useCleanArch, setUseCleanArch] = useState(true);
  const [showPrompt, setShowPrompt] = useState(false);

  async function submit(e?: React.FormEvent) {
    e?.preventDefault();
    const q = question.trim();
    if(!q) return;
    setLoading(true); setError(null); setResp(null);
    try {
  const params = new URLSearchParams();
  params.set('k', String(k));
  params.set('llm', String(useLLM));
  if(minScore.trim()) params.set('min_score', minScore.trim());
  
  // Choose endpoint based on architecture
  const endpoint = useCleanArch ? '/v2/chat' : '/chat';
  if(!useCleanArch) {
    params.set('use_clean_arch', 'false');
  }
  
  const url = useCleanArch 
    ? `${API_BASE}${endpoint}?k=${k}`
    : `${API_BASE}${endpoint}?${params.toString()}`;
  
  const r = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q })
      });
      if(!r.ok) {
        setError('Error '+r.status);
      } else {
        const data = await r.json();
        // Handle both legacy and v2 response formats
        if (useCleanArch && data.result) {
          // v2 API format
          const v2Data = data as V2ChatResponse;
          setResp({
            question: v2Data.query?.text || q,
            answer: v2Data.result.answer,
            sources: v2Data.result.sources || [],
            confidence: v2Data.result.confidence?.level
          });
        } else {
          // Legacy format
          setResp(data as ChatResponse);
        }
      }
    } catch(err: any) {
      setError(err.message || String(err));
    } finally { setLoading(false); }
  }

  return (
    <div className="app">
      <h1>AI File Assistant</h1>
      <form onSubmit={submit} className="chat-form" style={{gap:'0.75rem', display:'flex', flexDirection:'column'}}>
        <textarea value={question} onChange={e=>setQuestion(e.target.value)} placeholder="Type your question" />
        <div className="tuning" style={{display:'flex', flexWrap:'wrap', gap:'1rem', alignItems:'flex-end'}}>
          <label style={{display:'flex', flexDirection:'column', fontSize:12}}>k
            <input type="number" min={1} max={15} value={k} onChange={e=>setK(Number(e.target.value))} style={{width:70}} />
          </label>
          <label style={{display:'flex', flexDirection:'column', fontSize:12}}>min_score
            <input type="text" placeholder="auto" value={minScore} onChange={e=>setMinScore(e.target.value)} style={{width:80}} />
          </label>
          <label style={{display:'flex', alignItems:'center', gap:4, fontSize:12}} title="Use AI (OpenAI/Ollama) for generative answers vs simple text extraction">LLM
            <input type="checkbox" checked={useLLM} onChange={e=>setUseLLM(e.target.checked)} />
          </label>
          <label style={{display:'flex', alignItems:'center', gap:4, fontSize:12}} title="Use Clean Architecture (recommended)">Clean Arch
            <input type="checkbox" checked={useCleanArch} onChange={e=>setUseCleanArch(e.target.checked)} />
          </label>
          <label style={{display:'flex', alignItems:'center', gap:4, fontSize:12}}>Show prompt
            <input type="checkbox" checked={showPrompt} onChange={e=>setShowPrompt(e.target.checked)} />
          </label>
        </div>
        <div className="row">
          <button type="submit" disabled={loading}>{loading? 'Sending...' : 'Send'}</button>
          <button type="button" onClick={()=>{setQuestion(''); setResp(null);}}>Clear</button>
        </div>
      </form>
      {error && <div className="error">{error}</div>}
      {resp && (
        <div className="answer">
          <p><strong>Question:</strong> {resp.question}</p>
          <p><strong>Answer:</strong><br />{resp.answer}</p>
          <div style={{fontSize:12, opacity:0.7, display:'flex', gap:'1rem'}}>
            {typeof resp.kw_coverage === 'number' && <span>keyword coverage: {(resp.kw_coverage*100).toFixed(0)}%</span>}
            {resp.confidence && <span>confidence: {resp.confidence}</span>}
            {resp.reason && resp.reason !== 'ok' && <span>reason: {resp.reason}</span>}
          </div>
          {showPrompt && resp.prompt_preview && <details style={{margin:'0.5rem 0'}}><summary style={{cursor:'pointer'}}>Prompt preview</summary><pre style={{whiteSpace:'pre-wrap'}}>{resp.prompt_preview}</pre></details>}
          <div className="sources" style={{display:'flex', flexDirection:'column', gap:4}}>
            <strong>Sources:</strong>
            {resp.sources.map(s=> {
              const name = s.source.split('\\').slice(-1)[0];
              return (
                <div key={s.source+':'+s.chunk_index} style={{fontSize:12, background:'#f5f5f5', padding:'4px 6px', borderRadius:4}}>
                  <code>{name}#{s.chunk_index}</code>
                  {typeof s.score === 'number' && <span style={{marginLeft:6, opacity:0.7}}>score {s.score.toFixed(3)}</span>}
                  {s.preview && <div style={{marginTop:4}} dangerouslySetInnerHTML={{__html: s.preview.replace(/\*\*(.*?)\*\*/g,'<mark>$1</mark>')}} />}
                </div>
              );
            })}
          </div>
        </div>
      )}
      <footer>Backend: {API_BASE}</footer>
    </div>
  );
}
