import React, { useState, useCallback, useEffect, useRef } from 'react';
import { docsAPI, analysisAPI } from '../services/api';
import { Upload, FileText, CheckCircle, AlertCircle, Loader2, Brain, BarChart3, Sparkles } from 'lucide-react';
import { motion } from 'framer-motion';

const PIPELINE_STEPS = [
    { key: 'uploading', label: 'Uploading document...', icon: Upload },
    { key: 'processing', label: 'Extracting & processing text...', icon: FileText },
    { key: 'analyzing', label: 'Running AI analysis...', icon: Brain },
    { key: 'complete', label: 'Analysis complete!', icon: CheckCircle },
];

const UploadForm = ({ onComplete, onAnalysisReady }) => {
    const [file, setFile] = useState(null);
    const [uploading, setUploading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [pipelineStep, setPipelineStep] = useState(null);
    const [statusMessage, setStatusMessage] = useState('');
    const [error, setError] = useState('');
    const [dragActive, setDragActive] = useState(false);
    const pollRef = useRef(null);

    useEffect(() => {
        return () => { if (pollRef.current) clearInterval(pollRef.current); };
    }, []);

    const handleDrop = useCallback((e) => {
        e.preventDefault();
        setDragActive(false);
        if (e.dataTransfer.files?.[0]) { setFile(e.dataTransfer.files[0]); setError(''); setPipelineStep(null); }
    }, []);

    const runFullPipeline = async () => {
        if (!file) return;
        setUploading(true); setProgress(0); setError(''); setPipelineStep('uploading');
        setStatusMessage('Uploading your document...');

        try {
            const uploadRes = await docsAPI.upload(file, (e) => { setProgress(Math.round((e.loaded * 100) / e.total)); });
            const docId = uploadRes.data.document_id;
            setProgress(100); setStatusMessage('Upload complete! Processing document...');
            setPipelineStep('processing');
            await waitForProcessing(docId);
            setPipelineStep('analyzing'); setStatusMessage('Running AI-powered analysis...');
            await analysisAPI.trigger(docId);
            const analysisData = await waitForAnalysis(docId);
            setPipelineStep('complete'); setStatusMessage('Analysis complete! Showing results...');
            setUploading(false);
            if (onAnalysisReady) setTimeout(() => onAnalysisReady(docId, analysisData), 800);
            else if (onComplete) onComplete(docId);
        } catch (err) {
            setError(err.response?.data?.detail || err.message || 'Processing failed');
            setPipelineStep('failed'); setUploading(false);
        }
    };

    const waitForProcessing = (docId) => new Promise((resolve, reject) => {
        let attempts = 0;
        pollRef.current = setInterval(async () => {
            attempts++;
            try {
                const res = await docsAPI.get(docId);
                const status = res.data?.status;
                if (['processed', 'extracted', 'classified'].includes(status)) { clearInterval(pollRef.current); resolve(); }
                else if (status === 'failed') { clearInterval(pollRef.current); reject(new Error(res.data?.error_message || 'Processing failed')); }
                else setStatusMessage(`Processing document... (step ${Math.min(attempts, 6)}/6)`);
                if (attempts >= 120) { clearInterval(pollRef.current); reject(new Error('Timed out')); }
            } catch { if (attempts >= 120) { clearInterval(pollRef.current); reject(new Error('Timed out')); } }
        }, 5000);
    });

    const waitForAnalysis = (docId) => new Promise((resolve, reject) => {
        let attempts = 0;
        pollRef.current = setInterval(async () => {
            attempts++;
            try {
                const res = await analysisAPI.results(docId);
                if (res.data) { clearInterval(pollRef.current); resolve(res.data); }
                if (attempts >= 60) { clearInterval(pollRef.current); reject(new Error('Analysis timed out')); }
            } catch { if (attempts >= 60) { clearInterval(pollRef.current); reject(new Error('Analysis timed out')); } }
        }, 5000);
    });

    const currentStepIndex = PIPELINE_STEPS.findIndex(s => s.key === pipelineStep);
    const isRunning = pipelineStep && pipelineStep !== 'complete' && pipelineStep !== 'failed';

    return (
        <div className="rounded-2xl border p-8 max-w-2xl mx-auto"
            style={{ background: 'white', borderColor: 'rgba(119,141,169,0.12)' }}>
            <h2 className="text-xl font-display font-bold mb-6 flex items-center gap-2" style={{ color: '#0d1b2a' }}>
                <Sparkles className="w-5 h-5" style={{ color: '#415a77' }} />
                Upload & Analyze
            </h2>

            {/* Drop zone */}
            {!isRunning && pipelineStep !== 'complete' && (
                <div
                    className="relative border-2 border-dashed rounded-xl p-12 text-center transition-all cursor-pointer"
                    style={{ borderColor: dragActive ? '#415a77' : 'rgba(119,141,169,0.2)', background: dragActive ? 'rgba(65,90,119,0.04)' : 'transparent' }}
                    onDragOver={e => { e.preventDefault(); setDragActive(true); }}
                    onDragLeave={() => setDragActive(false)}
                    onDrop={handleDrop}
                    onClick={() => document.getElementById('file-input').click()}
                >
                    <input id="file-input" type="file" accept=".pdf,.docx,.doc,.txt,.html"
                        onChange={e => { setFile(e.target.files[0]); setError(''); setPipelineStep(null); }}
                        className="hidden" />
                    <FileText className="w-12 h-12 mx-auto mb-4" style={{ color: '#d1d5db' }} />
                    {file ? (
                        <div>
                            <p className="font-semibold text-sm" style={{ color: '#1b263b' }}>{file.name}</p>
                            <p className="text-xs mt-1" style={{ color: '#778da9' }}>{(file.size / 1024).toFixed(1)} KB</p>
                        </div>
                    ) : (
                        <div>
                            <p className="font-medium text-sm" style={{ color: '#415a77' }}>Drop your document here</p>
                            <p className="text-xs mt-1" style={{ color: '#778da9' }}>PDF, DOCX, TXT, HTML — up to 50MB</p>
                        </div>
                    )}
                </div>
            )}

            {/* Upload button */}
            {file && !isRunning && pipelineStep !== 'complete' && (
                <motion.button initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                    onClick={runFullPipeline} disabled={uploading}
                    className="mt-6 w-full py-3.5 rounded-xl font-semibold text-sm text-white transition-all hover:scale-[1.01] disabled:opacity-50 flex items-center justify-center gap-2"
                    style={{ background: 'linear-gradient(135deg, #415a77, #778da9)', boxShadow: '0 4px 20px rgba(65,90,119,0.3)' }}>
                    <Sparkles className="w-4 h-4" /> Upload & Analyze Automatically
                </motion.button>
            )}

            {/* Pipeline steps */}
            {(isRunning || pipelineStep === 'complete') && (
                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="mt-6 space-y-3">
                    {PIPELINE_STEPS.map((step, idx) => {
                        const isActive = step.key === pipelineStep;
                        const isDone = idx < currentStepIndex || pipelineStep === 'complete';
                        return (
                            <div key={step.key} className="flex items-center gap-3 p-3 rounded-lg border transition-all"
                                style={{
                                    background: isActive ? 'rgba(65,90,119,0.06)' : isDone ? 'rgba(16,185,129,0.04)' : 'rgba(119,141,169,0.03)',
                                    borderColor: isActive ? 'rgba(65,90,119,0.15)' : isDone ? 'rgba(16,185,129,0.1)' : 'rgba(119,141,169,0.06)',
                                }}>
                                {isDone && !isActive ? (
                                    <CheckCircle className="w-4 h-4" style={{ color: '#10b981' }} />
                                ) : isActive ? (
                                    <Loader2 className="w-4 h-4 animate-spin" style={{ color: '#415a77' }} />
                                ) : (
                                    <step.icon className="w-4 h-4" style={{ color: '#d1d5db' }} />
                                )}
                                <span className="text-xs font-medium" style={{ color: isActive ? '#415a77' : isDone ? '#10b981' : '#778da9' }}>
                                    {step.label}
                                </span>
                            </div>
                        );
                    })}
                    {pipelineStep === 'uploading' && (
                        <div className="rounded-full h-1.5 overflow-hidden" style={{ background: 'rgba(119,141,169,0.1)' }}>
                            <div className="h-full rounded-full transition-all duration-300" style={{ width: `${progress}%`, background: '#415a77' }} />
                        </div>
                    )}
                    <p className="text-xs text-center animate-pulse" style={{ color: '#778da9' }}>{statusMessage}</p>
                </motion.div>
            )}

            {/* Error */}
            {error && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                    className="mt-6 rounded-xl border p-4 flex items-center gap-3"
                    style={{ background: '#fef2f2', borderColor: '#fecaca' }}>
                    <AlertCircle className="w-5 h-5 flex-shrink-0" style={{ color: '#ef4444' }} />
                    <div>
                        <p className="text-sm font-medium" style={{ color: '#b91c1c' }}>Processing failed</p>
                        <p className="text-xs mt-0.5" style={{ color: '#ef4444' }}>{error}</p>
                    </div>
                </motion.div>
            )}
        </div>
    );
};

export default UploadForm;
