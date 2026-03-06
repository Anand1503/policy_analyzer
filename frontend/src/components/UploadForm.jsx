import React, { useState, useCallback } from 'react';
import { docsAPI } from '../services/api';
import { Upload, FileText, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';

const UploadForm = ({ onComplete }) => {
    const [file, setFile] = useState(null);
    const [uploading, setUploading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [result, setResult] = useState(null);
    const [error, setError] = useState('');
    const [dragActive, setDragActive] = useState(false);

    const ALLOWED = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain', 'text/html'];

    const handleDrop = useCallback((e) => {
        e.preventDefault();
        setDragActive(false);
        if (e.dataTransfer.files?.[0]) {
            setFile(e.dataTransfer.files[0]);
            setError('');
            setResult(null);
        }
    }, []);

    const handleUpload = async () => {
        if (!file) return;
        setUploading(true);
        setProgress(0);
        setError('');

        try {
            const res = await docsAPI.upload(file, (e) => {
                setProgress(Math.round((e.loaded * 100) / e.total));
            });
            setResult(res.data);
            if (onComplete) onComplete(res.data.document_id);
        } catch (err) {
            setError(err.response?.data?.detail || 'Upload failed');
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="bg-white rounded-2xl shadow-soft border border-slate-200/60 p-8 max-w-2xl mx-auto">
            <h2 className="text-xl font-display font-bold text-slate-800 mb-6 flex items-center gap-2">
                <Upload className="w-5 h-5 text-indigo-600" />
                Upload Document
            </h2>

            {/* Drag & Drop Zone */}
            <div
                className={`relative border-2 border-dashed rounded-xl p-12 text-center transition-all cursor-pointer ${dragActive ? 'border-blue-500 bg-blue-50' : 'border-slate-200 hover:border-blue-300 hover:bg-slate-50'
                    }`}
                onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
                onDragLeave={() => setDragActive(false)}
                onDrop={handleDrop}
                onClick={() => document.getElementById('file-input').click()}
            >
                <input
                    id="file-input"
                    type="file"
                    accept=".pdf,.docx,.doc,.txt,.html"
                    onChange={(e) => { setFile(e.target.files[0]); setError(''); setResult(null); }}
                    className="hidden"
                />
                <FileText className="w-12 h-12 text-slate-300 mx-auto mb-4" />
                {file ? (
                    <div>
                        <p className="font-semibold text-slate-700">{file.name}</p>
                        <p className="text-sm text-slate-400 mt-1">{(file.size / 1024).toFixed(1)} KB</p>
                    </div>
                ) : (
                    <div>
                        <p className="font-medium text-slate-600">Drop your document here</p>
                        <p className="text-sm text-slate-400 mt-1">PDF, DOCX, TXT, HTML — up to 50MB</p>
                    </div>
                )}
            </div>

            {/* Upload Button */}
            {file && !result && (
                <motion.button
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    onClick={handleUpload}
                    disabled={uploading}
                    className="mt-6 w-full bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-700 hover:to-indigo-800 text-white py-3.5 rounded-xl font-medium transition-all duration-300 disabled:opacity-50 flex items-center justify-center gap-2 shadow-[0_4px_14px_0_rgba(15,23,42,0.39)] hover:shadow-[0_6px_20px_rgba(79,70,229,0.23)] hover:-translate-y-0.5 disabled:hover:translate-y-0 disabled:hover:shadow-none"
                >
                    {uploading ? (
                        <>
                            <Loader2 className="w-5 h-5 animate-spin" />
                            Uploading... {progress}%
                        </>
                    ) : (
                        <>
                            <Upload className="w-5 h-5" />
                            Upload & Extract
                        </>
                    )}
                </motion.button>
            )}

            {/* Progress */}
            {uploading && (
                <div className="mt-4 bg-slate-100 rounded-full h-2 overflow-hidden">
                    <div className="bg-blue-600 h-full rounded-full transition-all" style={{ width: `${progress}%` }} />
                </div>
            )}

            {/* Success */}
            {result && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="mt-6 bg-green-50 border border-green-200 rounded-lg p-4 flex items-center gap-3"
                >
                    <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0" />
                    <div>
                        <p className="font-medium text-green-800">{result.message}</p>
                        <p className="text-xs text-green-600 mt-1">Document ID: {result.document_id}</p>
                    </div>
                </motion.div>
            )}

            {/* Error */}
            {error && (
                <div className="mt-6 bg-red-50 border border-red-200 rounded-lg p-4 flex items-center gap-3">
                    <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0" />
                    <p className="text-red-700 text-sm">{error}</p>
                </div>
            )}
        </div>
    );
};

export default UploadForm;
