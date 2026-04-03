import React, { useState, useEffect, useRef } from 'react';
import { docsAPI, analysisAPI } from '../services/api';
import { Upload, FileText, CheckCircle, AlertCircle, Loader2, Brain, ArrowUp, Plus, X, Paperclip } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const PIPELINE_STEPS = [
    { key: 'uploading', label: 'Uploading document...', icon: Upload },
    { key: 'processing', label: 'Extracting & segmenting clauses...', icon: FileText },
    { key: 'analyzing', label: 'Running AI classification & risk scoring...', icon: Brain },
    { key: 'complete', label: 'Analysis complete!', icon: CheckCircle },
];

const UploadForm = ({ onComplete, onAnalysisReady }) => {
    const [file, setFile] = useState(null);
    const [textContent, setTextContent] = useState('');
    const [uploading, setUploading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [pipelineStep, setPipelineStep] = useState(null);
    const [statusMessage, setStatusMessage] = useState('');
    const [error, setError] = useState('');
    const [menuOpen, setMenuOpen] = useState(false);
    const pollRef = useRef(null);
    const textareaRef = useRef(null);
    const menuRef = useRef(null);
    const plusBtnRef = useRef(null);

    useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

    // Auto-resize textarea
    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
            textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 160) + 'px';
        }
    }, [textContent]);

    // Close menu on outside click
    useEffect(() => {
        const handleClickOutside = (e) => {
            if (menuOpen && menuRef.current && !menuRef.current.contains(e.target) && !plusBtnRef.current.contains(e.target)) {
                setMenuOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, [menuOpen]);

    const getUploadFile = () => {
        if (file) return file;
        if (textContent.trim().length >= 10)
            return new File([textContent.trim()], 'pasted_policy.txt', { type: 'text/plain' });
        return null;
    };

    const canSubmit = !!file || textContent.trim().length >= 10;

    const runFullPipeline = async () => {
        const uploadFile = getUploadFile();
        if (!uploadFile) return;
        setUploading(true); setProgress(0); setError(''); setPipelineStep('uploading');
        setStatusMessage('Uploading your document...'); setMenuOpen(false);
        try {
            const uploadRes = await docsAPI.upload(uploadFile, (e) => setProgress(Math.round((e.loaded * 100) / e.total)));
            const docId = uploadRes.data.document_id;
            setProgress(100); setStatusMessage('Processing document...');
            setPipelineStep('processing');
            await waitForProcessing(docId);
            setPipelineStep('analyzing'); setStatusMessage('Running AI-powered analysis...');
            await analysisAPI.trigger(docId);
            const analysisData = await waitForAnalysis(docId);
            setPipelineStep('complete'); setStatusMessage('Analysis complete!');
            setUploading(false);
            if (onAnalysisReady) setTimeout(() => onAnalysisReady(docId, analysisData), 800);
            else if (onComplete) onComplete(docId);
        } catch (err) {
            setError(err.response?.data?.detail || err.message || 'Processing failed');
            setPipelineStep('failed'); setUploading(false);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey && canSubmit && !uploading) {
            e.preventDefault();
            runFullPipeline();
        }
    };

    const waitForProcessing = (docId) => new Promise((resolve, reject) => {
        let attempts = 0;
        const startTime = Date.now();
        const POLL_INTERVAL = 2000; // 2s instead of 5s for faster feedback
        const MAX_WAIT_MS = 4 * 60 * 1000; // 4 minute max
        pollRef.current = setInterval(async () => {
            attempts++;
            const elapsed = Math.round((Date.now() - startTime) / 1000);
            try {
                const res = await docsAPI.get(docId);
                const status = res.data?.status;
                if (['processed', 'extracted', 'classified'].includes(status)) {
                    clearInterval(pollRef.current); resolve();
                } else if (status === 'failed') {
                    clearInterval(pollRef.current);
                    reject(new Error(res.data?.error_message || 'Processing failed'));
                } else {
                    setStatusMessage(`Extracting text & segmenting clauses... (${elapsed}s)`);
                }
                if (Date.now() - startTime > MAX_WAIT_MS) {
                    clearInterval(pollRef.current); reject(new Error('Processing timed out after 4 minutes'));
                }
            } catch {
                if (Date.now() - startTime > MAX_WAIT_MS) {
                    clearInterval(pollRef.current); reject(new Error('Processing timed out'));
                }
            }
        }, POLL_INTERVAL);
    });

    const waitForAnalysis = (docId) => new Promise((resolve, reject) => {
        let attempts = 0;
        const startTime = Date.now();
        const POLL_INTERVAL = 2000; // 2s polling for faster feedback
        const MAX_WAIT_MS = 5 * 60 * 1000; // 5 minute max
        pollRef.current = setInterval(async () => {
            attempts++;
            const elapsed = Math.round((Date.now() - startTime) / 1000);
            try {
                const res = await analysisAPI.results(docId);
                if (res.data && res.data.clauses) {
                    clearInterval(pollRef.current); resolve(res.data);
                } else {
                    setStatusMessage(`AI classification & risk scoring in progress... (${elapsed}s)`);
                }
                if (Date.now() - startTime > MAX_WAIT_MS) {
                    clearInterval(pollRef.current); reject(new Error('Analysis timed out after 5 minutes'));
                }
            } catch (err) {
                // 404 means analysis not done yet — keep polling
                if (err.response?.status === 404) {
                    setStatusMessage(`AI models processing your document... (${elapsed}s)`);
                } else if (Date.now() - startTime > MAX_WAIT_MS) {
                    clearInterval(pollRef.current); reject(new Error('Analysis timed out'));
                }
            }
        }, POLL_INTERVAL);
    });

    const currentStepIndex = PIPELINE_STEPS.findIndex(s => s.key === pipelineStep);
    const isRunning = pipelineStep && pipelineStep !== 'complete' && pipelineStep !== 'failed';
    const resetForm = () => { setFile(null); setTextContent(''); setPipelineStep(null); setError(''); setProgress(0); setStatusMessage(''); };

    return (
        <div style={{ width: '100%', maxWidth: '800px', margin: '0 auto', display: 'flex', flexDirection: 'column', height: '100%', flex: 1 }}>
            
            {/* Header (centered) */}
            <div style={{ 
                flex: 1, 
                display: 'flex', 
                flexDirection: 'column', 
                justifyContent: 'center', 
                alignItems: 'center', 
                textAlign: 'center',
                opacity: (isRunning || pipelineStep === 'complete') ? 0 : 1,
                pointerEvents: (isRunning || pipelineStep === 'complete') ? 'none' : 'auto',
                transition: 'opacity 0.4s ease',
                paddingBottom: '40px'
            }}>
                <h2 style={{ fontSize: '36px', fontWeight: '700', color: 'var(--color-ink)', margin: 0, letterSpacing: '-0.02em' }}>
                    Upload & Analyze
                </h2>
                <p style={{ fontSize: '16px', color: 'var(--color-denim)', marginTop: '12px', maxWidth: '400px' }}>
                    Paste your policy text or attach a document to get started.
                </p>
            </div>

            {/* Bottom section (Pipeline & Input) */}
            <div style={{ width: '100%', paddingBottom: '10px' }}>

            <AnimatePresence mode="wait">
                {/* ── PIPELINE PROGRESS ── */}
                {(isRunning || pipelineStep === 'complete') && (
                    <motion.div key="pipeline" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                        style={{ background: 'var(--color-card)', borderRadius: '16px', border: '1px solid var(--color-card-border)', padding: '32px', boxShadow: '0 2px 16px rgba(13,27,42,0.06)', marginBottom: '16px' }}>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                            {PIPELINE_STEPS.map((step, idx) => {
                                const isActive = step.key === pipelineStep;
                                const isDone = idx < currentStepIndex || pipelineStep === 'complete';
                                return (
                                    <div key={step.key} style={{
                                        display: 'flex', alignItems: 'center', gap: '12px', padding: '14px 16px', borderRadius: '10px',
                                        border: `1px solid ${isActive ? 'var(--color-card-border)' : isDone ? 'var(--color-success)' : 'var(--color-card-border)'}`,
                                        background: isActive ? 'var(--color-card-border)' : isDone ? 'var(--color-success)' : 'var(--color-card-border)',
                                    }}>
                                        {isDone && !isActive ? <CheckCircle size={18} style={{ color: 'var(--color-success)', flexShrink: 0 }} />
                                            : isActive ? <Loader2 size={18} style={{ color: 'var(--color-dusk)', flexShrink: 0, animation: 'spin 1s linear infinite' }} />
                                                : <step.icon size={18} style={{ color: 'var(--color-denim)', flexShrink: 0 }} />}
                                        <span style={{ fontSize: '15px', fontWeight: '500', color: isActive ? 'var(--color-dusk)' : isDone ? 'var(--color-success)' : 'var(--color-denim)' }}>
                                            {step.label}
                                        </span>
                                    </div>
                                );
                            })}
                        </div>
                        {pipelineStep === 'uploading' && (
                            <div style={{ borderRadius: '8px', height: '6px', background: 'rgba(119,141,169,0.1)', overflow: 'hidden', marginTop: '16px' }}>
                                <div style={{ height: '100%', borderRadius: '8px', background: 'linear-gradient(90deg,#415a77,#778da9)', width: `${progress}%`, transition: 'width 0.3s' }} />
                            </div>
                        )}
                        <p style={{ fontSize: '13px', textAlign: 'center', color: 'var(--color-denim)', marginTop: '14px' }}>{statusMessage}</p>
                        {pipelineStep === 'complete' && (
                            <div style={{ marginTop: '20px', textAlign: 'center' }}>
                                <button onClick={resetForm} style={{ padding: '10px 28px', borderRadius: '10px', border: '1.5px solid rgba(119,141,169,0.2)', background: 'transparent', color: 'var(--color-dusk)', fontSize: '15px', fontWeight: '600', cursor: 'pointer' }}>
                                    + Analyze another document
                                </button>
                            </div>
                        )}
                    </motion.div>
                )}

                {/* ── UNIFIED INPUT BAR (Gemini-style) ── */}
                {!isRunning && pipelineStep !== 'complete' && (
                    <motion.div key="form" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                        style={{ display: 'flex', flexDirection: 'column', gap: '0' }}>

                        {/* Attached file chip — shown above the bar */}
                        <AnimatePresence>
                            {file && (
                                <motion.div
                                    initial={{ opacity: 0, y: 8, height: 0 }}
                                    animate={{ opacity: 1, y: 0, height: 'auto' }}
                                    exit={{ opacity: 0, y: -8, height: 0 }}
                                    style={{ marginBottom: '10px' }}
                                >
                                    <div style={{
                                        display: 'inline-flex', alignItems: 'center', gap: '10px',
                                        padding: '10px 16px', borderRadius: '12px',
                                        background: 'var(--color-card)',
                                        border: '1px solid rgba(16,185,129,0.25)',
                                        boxShadow: '0 2px 8px rgba(13,27,42,0.05)',
                                    }}>
                                        <FileText size={16} style={{ color: 'var(--color-success)', flexShrink: 0 }} />
                                        <div style={{ minWidth: 0 }}>
                                            <p style={{ fontSize: '14px', fontWeight: '600', color: 'var(--color-text-primary)', margin: 0, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '300px' }}>
                                                {file.name}
                                            </p>
                                            <p style={{ fontSize: '11px', color: 'var(--color-denim)', margin: 0 }}>
                                                {(file.size / 1024).toFixed(1)} KB
                                            </p>
                                        </div>
                                        <button
                                            onClick={() => setFile(null)}
                                            style={{
                                                width: '24px', height: '24px', borderRadius: '50%', border: 'none',
                                                background: 'var(--color-danger)', color: 'var(--color-danger)',
                                                cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
                                                flexShrink: 0, transition: 'background 0.2s',
                                            }}
                                            onMouseEnter={e => e.currentTarget.style.background = 'var(--color-danger)'}
                                            onMouseLeave={e => e.currentTarget.style.background = 'var(--color-danger)'}
                                        >
                                            <X size={14} />
                                        </button>
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>

                        {/* The unified input bar */}
                        <div style={{
                            background: 'var(--color-card)',
                            borderRadius: '24px',
                            border: '1.5px solid var(--color-card-border)',
                            boxShadow: '0 4px 24px rgba(13,27,42,0.08)',
                            padding: '6px 8px 6px 6px',
                            display: 'flex', alignItems: 'flex-end', gap: '4px',
                            position: 'relative',
                            transition: 'border-color 0.2s, box-shadow 0.2s',
                        }}>
                            {/* Plus button with dropdown */}
                            <div style={{ position: 'relative', flexShrink: 0, marginBottom: '6px' }}>
                                <button
                                    ref={plusBtnRef}
                                    onClick={() => setMenuOpen(!menuOpen)}
                                    style={{
                                        width: '38px', height: '38px', borderRadius: '50%', border: '1.5px solid var(--color-card-border)',
                                        background: menuOpen ? 'var(--color-card-border)' : 'transparent',
                                        color: 'var(--color-denim)', cursor: 'pointer',
                                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                                        transition: 'all 0.2s',
                                    }}
                                    onMouseEnter={e => { if (!menuOpen) e.currentTarget.style.background = 'var(--color-card-border)'; }}
                                    onMouseLeave={e => { if (!menuOpen) e.currentTarget.style.background = 'transparent'; }}
                                >
                                    <Plus size={20} style={{
                                        transition: 'transform 0.25s ease',
                                        transform: menuOpen ? 'rotate(45deg)' : 'rotate(0deg)',
                                    }} />
                                </button>

                                {/* Dropdown menu */}
                                <AnimatePresence>
                                    {menuOpen && (
                                        <motion.div
                                            ref={menuRef}
                                            initial={{ opacity: 0, y: 8, scale: 0.95 }}
                                            animate={{ opacity: 1, y: 0, scale: 1 }}
                                            exit={{ opacity: 0, y: 8, scale: 0.95 }}
                                            transition={{ duration: 0.15, ease: 'easeOut' }}
                                            style={{
                                                position: 'absolute', bottom: '100%', left: '0',
                                                marginBottom: '8px', minWidth: '220px',
                                                background: 'var(--color-card)',
                                                borderRadius: '16px',
                                                border: '1px solid var(--color-card-border)',
                                                boxShadow: '0 8px 32px rgba(13,27,42,0.12), 0 2px 8px rgba(13,27,42,0.06)',
                                                padding: '6px',
                                                zIndex: 50,
                                            }}
                                        >
                                            <input
                                                id="file-input"
                                                type="file"
                                                accept=".pdf,.docx,.doc,.txt,.html"
                                                onChange={e => {
                                                    if (e.target.files[0]) {
                                                        setFile(e.target.files[0]);
                                                        setError('');
                                                        setMenuOpen(false);
                                                    }
                                                }}
                                                style={{ display: 'none' }}
                                            />
                                            <button
                                                onClick={() => document.getElementById('file-input').click()}
                                                style={{
                                                    width: '100%', display: 'flex', alignItems: 'center', gap: '12px',
                                                    padding: '12px 14px', borderRadius: '12px', border: 'none',
                                                    background: 'transparent', color: 'var(--color-text-primary)',
                                                    fontSize: '14px', fontWeight: '500', cursor: 'pointer',
                                                    transition: 'background 0.15s',
                                                    textAlign: 'left',
                                                }}
                                                onMouseEnter={e => e.currentTarget.style.background = 'var(--color-card-border)'}
                                                onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                                            >
                                                <Paperclip size={18} style={{ color: 'var(--color-denim)' }} />
                                                <span>Upload document</span>
                                            </button>
                                            <div style={{ height: '1px', background: 'var(--color-card-border)', margin: '2px 10px' }} />
                                            <button
                                                onClick={() => {
                                                    setMenuOpen(false);
                                                    if (textareaRef.current) textareaRef.current.focus();
                                                }}
                                                style={{
                                                    width: '100%', display: 'flex', alignItems: 'center', gap: '12px',
                                                    padding: '12px 14px', borderRadius: '12px', border: 'none',
                                                    background: 'transparent', color: 'var(--color-text-primary)',
                                                    fontSize: '14px', fontWeight: '500', cursor: 'pointer',
                                                    transition: 'background 0.15s',
                                                    textAlign: 'left',
                                                }}
                                                onMouseEnter={e => e.currentTarget.style.background = 'var(--color-card-border)'}
                                                onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                                            >
                                                <FileText size={18} style={{ color: 'var(--color-denim)' }} />
                                                <span>Paste policy text</span>
                                            </button>
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </div>

                            {/* Textarea */}
                            <textarea
                                ref={textareaRef}
                                value={textContent}
                                onChange={e => { setTextContent(e.target.value); setError(''); }}
                                onKeyDown={handleKeyDown}
                                placeholder="Paste policy text here or upload a document..."
                                rows={1}
                                style={{
                                    flex: 1, border: 'none', outline: 'none', resize: 'none',
                                    fontSize: '15px', lineHeight: '1.6', color: 'var(--color-text-primary)',
                                    background: 'transparent', fontFamily: 'inherit',
                                    padding: '10px 0', maxHeight: '160px', overflowY: 'auto',
                                }}
                            />

                            {/* Send button */}
                            <button
                                onClick={runFullPipeline}
                                disabled={!canSubmit || uploading}
                                style={{
                                    width: '40px', height: '40px', borderRadius: '50%', border: 'none',
                                    background: canSubmit
                                        ? 'linear-gradient(135deg,#415a77,#778da9)'
                                        : 'var(--color-card-border)',
                                    color: canSubmit ? 'var(--color-card)' : 'var(--color-denim)',
                                    cursor: canSubmit ? 'pointer' : 'default',
                                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                                    flexShrink: 0, marginBottom: '6px',
                                    transition: 'all 0.25s',
                                    boxShadow: canSubmit ? '0 2px 10px rgba(65,90,119,0.3)' : 'none',
                                }}
                            >
                                <ArrowUp size={18} />
                            </button>
                        </div>

                        {/* Helper text */}
                        <AnimatePresence>
                            {(textContent || file) && (
                                <motion.p
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    exit={{ opacity: 0 }}
                                    style={{ fontSize: '12px', color: 'var(--color-denim)', marginTop: '10px', textAlign: 'center' }}
                                >
                                    Press Enter or click ↑ to analyze · Shift+Enter for new line
                                </motion.p>
                            )}
                        </AnimatePresence>

                        {/* Supported formats hint */}
                        {!textContent && !file && (
                            <p style={{ fontSize: '12px', color: 'var(--color-denim)', marginTop: '12px', textAlign: 'center' }}>
                                Supports PDF, DOCX, TXT, HTML up to 50 MB
                            </p>
                        )}

                        {/* Error */}
                        {error && (
                            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ marginTop: '16px', borderRadius: '12px', border: '1px solid #fecaca', background: 'var(--color-card)', padding: '14px 16px', display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
                                <AlertCircle size={18} style={{ color: 'var(--color-danger)', flexShrink: 0 }} />
                                <div style={{ flex: 1 }}>
                                    <p style={{ fontWeight: '600', fontSize: '14px', color: 'var(--color-danger)', margin: 0 }}>Processing failed</p>
                                    <p style={{ fontSize: '13px', color: 'var(--color-danger)', margin: '4px 0 0' }}>{error}</p>
                                </div>
                                <button onClick={resetForm} style={{ fontSize: '12px', color: 'var(--color-denim)', background: 'none', border: 'none', cursor: 'pointer' }}>Try again</button>
                            </motion.div>
                        )}
                    </motion.div>
                )}
            </AnimatePresence>
            </div>
        </div>
    );
};

export default UploadForm;
