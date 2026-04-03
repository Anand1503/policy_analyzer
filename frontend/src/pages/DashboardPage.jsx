import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { docsAPI, analysisAPI } from '../services/api';
import UploadForm from '../components/UploadForm';
import AnalysisResults from '../components/AnalysisResults';
import ChatPanel from '../components/ChatPanel';
import {
    FileText, Loader2, BarChart3, AlertTriangle,
    CheckCircle, Clock, ArrowRight, MessageSquare, XCircle
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const DashboardPage = ({ initialView }) => {
    const { user } = useAuth();
    const [documents, setDocuments] = useState([]);
    const [selectedDoc, setSelectedDoc] = useState(null);
    const [analysisData, setAnalysisData] = useState(null);
    const [view, setView] = useState(initialView || 'overview');
    const [loading, setLoading] = useState(false);

    useEffect(() => { setView(initialView || 'overview'); }, [initialView]);
    useEffect(() => { loadDocuments(); }, []);

    const loadDocuments = async () => {
        try {
            const res = await docsAPI.list();
            setDocuments(res.data.documents || []);
        } catch (err) {
            console.error('Failed to load documents', err);
        }
    };

    const handleAnalysisReady = (docId, data) => {
        loadDocuments();
        setAnalysisData(data);
        setSelectedDoc(docId);
        setView('results');
    };

    const handleViewResults = async (docId) => {
        setLoading(true);
        try {
            const res = await analysisAPI.results(docId);
            if (res.data) {
                setAnalysisData(res.data);
                setSelectedDoc(docId);
                setView('results');
            }
        } catch (err) {
            console.error('Failed to load results', err);
        } finally {
            setLoading(false);
        }
    };

    const statusBadge = (status) => {
        const styles = {
            uploaded: { bg: 'rgba(119,141,169,0.1)', color: 'var(--color-denim)' },
            processing: { bg: 'rgba(65,90,119,0.1)', color: 'var(--color-dusk)' },
            processed: { bg: 'rgba(65,90,119,0.12)', color: 'var(--color-dusk)' },
            analyzed: { bg: 'rgba(16,185,129,0.1)', color: 'var(--color-success)' },
            failed: { bg: 'rgba(239,68,68,0.1)', color: 'var(--color-danger)' },
        };
        const icons = { uploaded: Clock, processing: Loader2, processed: CheckCircle, analyzed: CheckCircle, failed: XCircle };
        const Icon = icons[status] || Clock;
        const s = styles[status] || styles.uploaded;
        return (
            <span className="inline-flex items-center gap-1 text-xs px-2.5 py-1 rounded-full font-medium"
                style={{ background: s.bg, color: s.color }}>
                <Icon className={`w-3 h-3 ${status === 'processing' ? 'animate-spin' : ''}`} />
                {status}
            </span>
        );
    };

    const totalDocs = documents.length;
    const analyzedDocs = documents.filter(d => d.status === 'analyzed').length;
    const failedDocs = documents.filter(d => d.status === 'failed').length;

    return (
        <div className="min-h-full flex flex-col h-[calc(100vh-48px)]">
            <div className="max-w-6xl mx-auto px-4 py-4 w-full flex-1 flex flex-col">
                <AnimatePresence mode="wait">

                    {/* ─── OVERVIEW ─────────────────────────── */}
                    {view === 'overview' && (
                        <motion.div key="overview" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                            <div className="mb-8">
                                <h2 className="text-3xl font-display font-bold" style={{ color: 'var(--color-ink)' }}>
                                    Welcome, {user?.full_name || user?.username}
                                </h2>
                                <p className="mt-2 text-base" style={{ color: 'var(--color-denim)' }}>Overview of your policy analysis activity.</p>
                            </div>

                            <div className="grid grid-cols-3 gap-4 mb-8">
                                {[
                                    { label: 'Documents', value: totalDocs, icon: FileText, iconColor: 'var(--color-dusk)' },
                                    { label: 'Analyzed', value: analyzedDocs, icon: CheckCircle, iconColor: 'var(--color-success)' },
                                    { label: 'Failed', value: failedDocs, icon: AlertTriangle, iconColor: 'var(--color-danger)' },
                                ].map(stat => (
                                    <div key={stat.label} className="rounded-xl border p-5"
                                        style={{ background: 'var(--color-card)', borderColor: 'var(--color-card-border)' }}>
                                        <div className="flex items-center justify-between mb-3">
                                            <span className="text-sm font-semibold" style={{ color: stat.iconColor !== 'var(--color-dusk)' ? stat.iconColor : 'var(--color-denim)' }}>{stat.label}</span>
                                             <stat.icon className="w-5 h-5" style={{ color: stat.iconColor }} />
                                         </div>
                                         <p className="text-4xl font-display font-bold mt-2" style={{ color: 'var(--color-ink)' }}>{stat.value}</p>
                                    </div>
                                ))}
                            </div>

                            <div className="rounded-xl border" style={{ background: 'var(--color-card)', borderColor: 'var(--color-card-border)' }}>
                                <div className="p-5 flex items-center justify-between" style={{ borderBottom: '1px solid rgba(119,141,169,0.08)' }}>
                            <h3 className="text-lg font-display font-semibold" style={{ color: 'var(--color-ink)' }}>Recent Documents</h3>
                                    {documents.length > 0 && (
                                        <button onClick={() => setView('history')} className="text-xs font-medium flex items-center gap-1" style={{ color: 'var(--color-dusk)' }}>
                                            View all <ArrowRight className="w-3 h-3" />
                                        </button>
                                    )}
                                </div>
                                {documents.length === 0 ? (
                                    <div className="p-12 text-center">
                                        <FileText className="w-10 h-10 mx-auto mb-3" style={{ color: 'var(--color-denim)' }} />
                                        <p className="text-sm" style={{ color: 'var(--color-denim)' }}>No documents yet. Upload one to get started.</p>
                                    </div>
                                ) : (
                                    <div>
                                        {documents.slice(0, 5).map((doc, i) => (
                                            <div key={doc.id} className="px-5 py-3.5 flex items-center justify-between transition-colors hover:bg-black/5 dark:hover:bg-white/5"
                                                style={{ borderTop: i > 0 ? '1px solid rgba(119,141,169,0.06)' : 'none' }}>
                                                <div className="flex items-center gap-3 min-w-0">
                                                    <FileText className="w-4 h-4 flex-shrink-0" style={{ color: 'var(--color-denim)' }} />
                                                    <span className="text-base font-semibold truncate" style={{ color: 'var(--color-ink)' }}>{doc.original_filename}</span>
                                                </div>
                                                <div className="flex items-center gap-2 flex-shrink-0">
                                                    {statusBadge(doc.status)}
                                                    {doc.status === 'analyzed' && (
                                                        <button onClick={() => handleViewResults(doc.id)}
                                                            className="text-xs font-medium flex items-center gap-1 ml-2" style={{ color: 'var(--color-dusk)' }}>
                                                            View <ArrowRight className="w-3 h-3" />
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

                    {/* ─── UPLOAD ───────────────────────────── */}
                    {view === 'upload' && (
                        <motion.div key="upload" className="flex flex-col flex-1 w-full" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                            <UploadForm onAnalysisReady={handleAnalysisReady} />
                        </motion.div>
                    )}

                    {/* ─── HISTORY ──────────────────────────── */}
                    {view === 'history' && (
                        <motion.div key="history" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                            <div className="rounded-xl border" style={{ background: 'var(--color-card)', borderColor: 'var(--color-card-border)' }}>
                                <div className="p-5" style={{ borderBottom: '1px solid rgba(119,141,169,0.08)' }}>
                                    <h3 className="text-base font-display font-semibold" style={{ color: 'var(--color-ink)' }}>All Documents</h3>
                                </div>
                                {documents.length === 0 ? (
                                    <div className="p-12 text-center">
                                        <FileText className="w-10 h-10 mx-auto mb-3" style={{ color: 'var(--color-denim)' }} />
                                        <p className="text-sm" style={{ color: 'var(--color-denim)' }}>No documents uploaded yet.</p>
                                    </div>
                                ) : (
                                    <div>
                                        {documents.map((doc, i) => (
                                            <div key={doc.id} className="px-5 py-4 flex items-center justify-between transition-colors hover:bg-black/5 dark:hover:bg-white/5"
                                                style={{ borderTop: i > 0 ? '1px solid rgba(119,141,169,0.06)' : 'none' }}>
                                                <div className="flex items-center gap-3 min-w-0 flex-1">
                                                    <FileText className="w-4 h-4 flex-shrink-0" style={{ color: 'var(--color-denim)' }} />
                                                    <div className="min-w-0">
                                                        <p className="text-sm font-medium truncate" style={{ color: 'var(--color-prussian)' }}>{doc.original_filename}</p>
                                                        <p className="text-xs mt-0.5" style={{ color: 'var(--color-denim)' }}>
                                                            {doc.file_type?.toUpperCase()} • {(doc.file_size_bytes / 1024).toFixed(0)} KB
                                                        </p>
                                                    </div>
                                                </div>
                                                <div className="flex items-center gap-3 flex-shrink-0">
                                                    {statusBadge(doc.status)}
                                                    {doc.status === 'analyzed' && (
                                                        <button onClick={() => handleViewResults(doc.id)}
                                                            className="text-xs px-3 py-1.5 rounded-lg font-medium flex items-center gap-1 transition-colors"
                                                            style={{ background: 'rgba(65,90,119,0.08)', color: 'var(--color-dusk)' }}>
                                                            <BarChart3 className="w-3 h-3" /> Results
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

                    {/* ─── RESULTS ──────────────────────────── */}
                    {view === 'results' && (
                        <motion.div key="results" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                            <button onClick={() => { setView('overview'); setAnalysisData(null); }}
                                className="text-sm font-medium mb-4 flex items-center gap-1" style={{ color: 'var(--color-denim)' }}>
                                ← Back to Dashboard
                            </button>
                            {analysisData ? (
                                <div className="space-y-6">
                                    <AnalysisResults data={analysisData} />
                                    {selectedDoc && (
                                        <div className="rounded-xl border p-5" style={{ background: 'var(--color-card)', borderColor: 'var(--color-card-border)' }}>
                                            <h3 className="text-base font-display font-semibold mb-4 flex items-center gap-2" style={{ color: 'var(--color-ink)' }}>
                                                <MessageSquare className="w-4 h-4" style={{ color: 'var(--color-dusk)' }} />
                                                Ask about this document
                                            </h3>
                                            <ChatPanel documentId={selectedDoc} />
                                        </div>
                                    )}
                                </div>
                            ) : (
                                <div className="text-center py-20">
                                    <BarChart3 className="w-10 h-10 mx-auto mb-3 opacity-30" style={{ color: 'var(--color-denim)' }} />
                                    <p className="text-sm" style={{ color: 'var(--color-denim)' }}>No results to display.</p>
                                </div>
                            )}
                        </motion.div>
                    )}

                </AnimatePresence>
            </div>

            {loading && (
                <div className="fixed inset-0 flex items-center justify-center z-50" style={{ background: 'rgba(224,225,221,0.7)', backdropFilter: 'blur(4px)' }}>
                    <Loader2 className="w-8 h-8 animate-spin" style={{ color: 'var(--color-dusk)' }} />
                </div>
            )}
        </div>
    );
};

export default DashboardPage;
