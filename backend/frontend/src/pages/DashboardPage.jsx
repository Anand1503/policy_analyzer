import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { docsAPI, analysisAPI } from '../services/api';
import UploadForm from '../components/UploadForm';
import AnalysisResults from '../components/AnalysisResults';
import ChatPanel from '../components/ChatPanel';
import {
    ShieldCheck, LogOut, FileText, Brain, Upload,
    Loader2, BarChart3, AlertTriangle, CheckCircle
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const DashboardPage = ({ initialView }) => {
    const { user, logout } = useAuth();
    const [documents, setDocuments] = useState([]);
    const [selectedDoc, setSelectedDoc] = useState(null);
    const [analysisData, setAnalysisData] = useState(null);
    const [view, setView] = useState(initialView || 'overview'); // overview | upload | documents | analysis | chat
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        loadDocuments();
    }, []);

    const loadDocuments = async () => {
        try {
            const res = await docsAPI.list();
            setDocuments(res.data.documents || []);
        } catch (err) {
            console.error('Failed to load documents', err);
        }
    };

    const handleUploadComplete = (docId) => {
        loadDocuments();
        setSelectedDoc(docId);
        setView('documents');
    };

    const handleAnalyze = async (docId) => {
        setLoading(true);
        try {
            await analysisAPI.trigger(docId);
            // Poll for results
            const poll = setInterval(async () => {
                try {
                    const res = await analysisAPI.results(docId);
                    if (res.data) {
                        setAnalysisData(res.data);
                        setSelectedDoc(docId);
                        setView('analysis');
                        clearInterval(poll);
                        setLoading(false);
                    }
                } catch {
                    // Still processing, continue polling
                }
            }, 3000);
            // Safety timeout after 5 minutes
            setTimeout(() => { clearInterval(poll); setLoading(false); }, 300000);
        } catch (err) {
            console.error('Analysis failed', err);
            setLoading(false);
        }
    };

    const handleViewResults = async (docId) => {
        setLoading(true);
        try {
            const res = await analysisAPI.results(docId);
            if (res.data) {
                setAnalysisData(res.data);
                setSelectedDoc(docId);
                setView('analysis');
            }
        } catch (err) {
            console.error('Failed to load results', err);
        } finally {
            setLoading(false);
        }
    };

    const statusBadge = (status) => {
        const colors = {
            uploaded: 'bg-gray-100 text-gray-600',
            processing: 'bg-blue-100 text-blue-700',
            extracted: 'bg-yellow-100 text-yellow-700',
            processed: 'bg-purple-100 text-purple-700',
            classified: 'bg-orange-100 text-orange-700',
            analyzed: 'bg-green-100 text-green-700',
            failed: 'bg-red-100 text-red-700',
        };
        return (
            <span className={`text-xs px-2 py-1 rounded-full font-medium ${colors[status] || colors.uploaded}`}>
                {status}
            </span>
        );
    };

    return (
        <div className="min-h-screen">
            <div className="max-w-7xl mx-auto px-4 py-8">
                {/* Navigation Tabs */}
                <div className="flex gap-2 mb-8">
                    {[
                        { id: 'overview', label: 'Overview', icon: BarChart3 },
                        { id: 'upload', label: 'Upload', icon: Upload },
                        { id: 'documents', label: 'Documents', icon: FileText },
                        { id: 'analysis', label: 'Analysis', icon: BarChart3 },
                        { id: 'chat', label: 'AI Chat', icon: Brain },
                    ].map(tab => (
                        <button
                            key={tab.id}
                            onClick={() => setView(tab.id)}
                            className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-all ${view === tab.id
                                ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/25'
                                : 'bg-white text-slate-600 hover:bg-slate-50 border border-slate-200'
                                }`}
                        >
                            <tab.icon className="w-4 h-4" />
                            {tab.label}
                        </button>
                    ))}
                </div>

                {/* Content */}
                <AnimatePresence mode="wait">
                    {view === 'overview' && (
                        <motion.div key="overview" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                                <div className="bg-white rounded-2xl shadow-soft border border-slate-200/60 p-6 hover:shadow-hover transition-all duration-300 hover:-translate-y-1">
                                    <div className="flex items-center gap-3 mb-3">
                                        <div className="bg-indigo-50 p-2.5 rounded-xl"><FileText className="w-5 h-5 text-indigo-600" /></div>
                                        <h3 className="font-semibold text-slate-700 font-display">Total Documents</h3>
                                    </div>
                                    <p className="text-4xl font-display font-bold text-slate-900">{documents.length}</p>
                                </div>
                                <div className="bg-white rounded-2xl shadow-soft border border-slate-200/60 p-6 hover:shadow-hover transition-all duration-300 hover:-translate-y-1">
                                    <div className="flex items-center gap-3 mb-3">
                                        <div className="bg-emerald-50 p-2.5 rounded-xl"><CheckCircle className="w-5 h-5 text-emerald-600" /></div>
                                        <h3 className="font-semibold text-slate-700 font-display">Analyzed</h3>
                                    </div>
                                    <p className="text-4xl font-display font-bold text-slate-900">{documents.filter(d => d.status === 'analyzed').length}</p>
                                </div>
                                <div className="bg-white rounded-2xl shadow-soft border border-slate-200/60 p-6 hover:shadow-hover transition-all duration-300 hover:-translate-y-1">
                                    <div className="flex items-center gap-3 mb-3">
                                        <div className="bg-rose-50 p-2.5 rounded-xl"><AlertTriangle className="w-5 h-5 text-rose-600" /></div>
                                        <h3 className="font-semibold text-slate-700 font-display">High Risk</h3>
                                    </div>
                                    <p className="text-4xl font-display font-bold text-slate-900">{documents.filter(d => d.risk_level === 'high' || d.risk_level === 'critical').length}</p>
                                </div>
                            </div>
                            <div className="bg-white rounded-2xl shadow-soft border border-slate-200/60 p-6">
                                <h3 className="text-xl font-display font-bold text-slate-900 mb-6">Recent Documents</h3>
                                {documents.length === 0 ? (
                                    <p className="text-slate-400 text-center py-8">No documents yet. Upload one to get started.</p>
                                ) : (
                                    <div className="divide-y divide-slate-100">
                                        {documents.slice(0, 5).map(doc => (
                                            <div key={doc.id} className="py-4 flex items-center justify-between group">
                                                <div className="flex items-center gap-4">
                                                    <div className="p-2.5 bg-slate-50 rounded-xl group-hover:bg-indigo-50 transition-colors">
                                                        <FileText className="w-5 h-5 text-slate-400 group-hover:text-indigo-500 transition-colors" />
                                                    </div>
                                                    <span className="text-sm font-medium text-slate-700 group-hover:text-indigo-700 transition-colors">{doc.original_filename}</span>
                                                </div>
                                                {statusBadge(doc.status)}
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </motion.div>
                    )}

                    {view === 'upload' && (
                        <motion.div key="upload" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                            <UploadForm onComplete={handleUploadComplete} />
                        </motion.div>
                    )}

                    {view === 'documents' && (
                        <motion.div key="docs" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                            <div className="bg-white rounded-2xl shadow-soft border border-slate-200/60 overflow-hidden">
                                <div className="p-6 border-b border-slate-100">
                                    <h2 className="text-xl font-display font-bold text-slate-800 flex items-center gap-2">
                                        <FileText className="w-5 h-5 text-indigo-600" /> Your Documents
                                    </h2>
                                </div>
                                {documents.length === 0 ? (
                                    <div className="p-12 text-center text-slate-400">
                                        No documents yet. Upload one to get started.
                                    </div>
                                ) : (
                                    <div className="divide-y divide-slate-100">
                                        {documents.map(doc => (
                                            <div key={doc.id} className="p-4 flex items-center justify-between hover:bg-slate-50 transition-colors">
                                                <div className="flex items-center gap-3">
                                                    <FileText className="w-8 h-8 text-slate-400" />
                                                    <div>
                                                        <p className="font-medium text-slate-700">{doc.original_filename}</p>
                                                        <p className="text-xs text-slate-400">
                                                            {doc.file_type.toUpperCase()} • {(doc.file_size_bytes / 1024).toFixed(1)} KB
                                                        </p>
                                                    </div>
                                                </div>
                                                <div className="flex items-center gap-3">
                                                    {statusBadge(doc.status)}
                                                    {(doc.status === 'processed' || doc.status === 'extracted' || doc.status === 'classified') && (
                                                        <button
                                                            onClick={() => handleAnalyze(doc.id)}
                                                            disabled={loading}
                                                            className="bg-indigo-600 hover:bg-indigo-700 text-white text-xs px-4 py-2 rounded-xl flex items-center gap-1.5 disabled:opacity-50 transition-all shadow-sm shadow-indigo-600/20"
                                                        >
                                                            {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Brain className="w-3.5 h-3.5" />}
                                                            Analyze
                                                        </button>
                                                    )}
                                                    {doc.status === 'analyzed' && (
                                                        <button
                                                            onClick={() => handleViewResults(doc.id)}
                                                            className="bg-emerald-600 hover:bg-emerald-700 text-white text-xs px-4 py-2 rounded-xl flex items-center gap-1.5 transition-all shadow-sm shadow-emerald-600/20"
                                                        >
                                                            <BarChart3 className="w-3.5 h-3.5" />
                                                            View Results
                                                        </button>
                                                    )}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </motion.div>
                    )}

                    {view === 'analysis' && (
                        <motion.div key="analysis" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                            {analysisData ? (
                                <AnalysisResults data={analysisData} />
                            ) : (
                                <div className="text-center py-20 text-slate-400">
                                    <BarChart3 className="w-12 h-12 mx-auto mb-4 opacity-50" />
                                    <p>No analysis results yet. Select a document and click "Analyze".</p>
                                </div>
                            )}
                        </motion.div>
                    )}

                    {view === 'chat' && (
                        <motion.div key="chat" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                            <ChatPanel documentId={selectedDoc} />
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>

            {/* Loading Overlay */}
            {loading && (
                <div className="fixed inset-0 bg-black/30 backdrop-blur-sm flex items-center justify-center z-50">
                    <div className="bg-white rounded-2xl p-8 shadow-2xl flex flex-col items-center gap-4">
                        <Loader2 className="w-10 h-10 text-blue-600 animate-spin" />
                        <p className="font-medium text-slate-700">Analyzing document...</p>
                        <p className="text-sm text-slate-400">This may take 1-3 minutes</p>
                    </div>
                </div>
            )}
        </div>
    );
};

export default DashboardPage;
