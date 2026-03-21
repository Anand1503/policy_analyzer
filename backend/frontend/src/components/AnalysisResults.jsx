import React, { useState } from 'react';
import {
    ShieldAlert, ShieldCheck, AlertTriangle, ChevronDown, ChevronUp,
    BarChart3, ListChecks, Lightbulb, FileText, Eye, EyeOff,
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { RiskDistributionChart, CategoryChart, ComplianceChart, ShapExplanation } from './RiskChart';

const RISK_COLORS = {
    low: { bg: 'bg-green-50', border: 'border-green-200', text: 'text-green-700', badge: 'bg-green-100 text-green-700' },
    medium: { bg: 'bg-yellow-50', border: 'border-yellow-200', text: 'text-yellow-700', badge: 'bg-yellow-100 text-yellow-700' },
    high: { bg: 'bg-red-50', border: 'border-red-200', text: 'text-red-700', badge: 'bg-red-100 text-red-700' },
    critical: { bg: 'bg-red-100', border: 'border-red-300', text: 'text-red-800', badge: 'bg-red-200 text-red-800' },
};

const RiskIcon = ({ level }) => {
    if (level === 'low') return <ShieldCheck className="w-5 h-5 text-green-600" />;
    if (level === 'high' || level === 'critical') return <ShieldAlert className="w-5 h-5 text-red-600" />;
    return <AlertTriangle className="w-5 h-5 text-yellow-600" />;
};

const AnalysisResults = ({ data }) => {
    const [expandedClause, setExpandedClause] = useState(null);
    const [showCharts, setShowCharts] = useState(true);

    if (!data) return null;

    const { summary, overall_risk, overall_score, recommendations, category_breakdown, clauses, compliance, shap_explanation } = data;

    return (
        <div className="space-y-6">
            {/* Overall Score Card */}
            <div className="bg-white rounded-2xl shadow-soft border border-slate-200/60 p-6">
                <div className="flex items-center justify-between mb-6">
                    <h2 className="text-xl font-display font-bold text-slate-900 flex items-center gap-2">
                        <BarChart3 className="w-5 h-5 text-indigo-600" />
                        Analysis Overview
                    </h2>
                    <div className="flex items-center gap-3">
                        <button
                            onClick={() => setShowCharts(!showCharts)}
                            className="text-xs text-slate-500 hover:text-blue-600 flex items-center gap-1 transition-colors"
                        >
                            {showCharts ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                            {showCharts ? 'Hide Charts' : 'Show Charts'}
                        </button>
                        <div className={`px-4 py-2 rounded-full font-bold text-sm ${RISK_COLORS[overall_risk]?.badge || 'bg-gray-100'}`}>
                            {overall_risk?.toUpperCase()} RISK
                        </div>
                    </div>
                </div>

                {/* Risk Score Gauge */}
                <div className="flex items-center gap-6 mb-6">
                    <div className="relative w-24 h-24">
                        <svg className="w-24 h-24 -rotate-90" viewBox="0 0 100 100">
                            <circle cx="50" cy="50" r="40" fill="none" stroke="#e2e8f0" strokeWidth="10" />
                            <circle
                                cx="50" cy="50" r="40" fill="none"
                                stroke={overall_risk === 'low' ? '#22c55e' : overall_risk === 'high' ? '#ef4444' : '#eab308'}
                                strokeWidth="10" strokeLinecap="round"
                                strokeDasharray={`${(overall_score || 0) * 251.2} 251.2`}
                            />
                        </svg>
                        <span className="absolute inset-0 flex items-center justify-center text-xl font-bold text-slate-700">
                            {Math.round((overall_score || 0) * 100)}
                        </span>
                    </div>
                    <div className="flex-1">
                        <p className="text-sm text-slate-500 mb-1">Risk Score</p>
                        <p className="text-slate-700 text-sm">{summary}</p>
                    </div>
                </div>

                {/* Category Breakdown (inline) */}
                {category_breakdown && (
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                        {Object.entries(category_breakdown).map(([cat, count]) => (
                            <div key={cat} className="bg-slate-50 rounded-lg p-3 text-center">
                                <p className="text-lg font-bold text-slate-700">{count}</p>
                                <p className="text-xs text-slate-400 leading-tight">{cat.replace(/_/g, ' ')}</p>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Charts Section */}
            <AnimatePresence>
                {showCharts && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="grid grid-cols-1 lg:grid-cols-2 gap-6"
                    >
                        <RiskDistributionChart clauses={clauses} />
                        <CategoryChart categoryBreakdown={category_breakdown} />
                        {compliance && <ComplianceChart complianceData={compliance} />}
                        {shap_explanation && <ShapExplanation explanation={shap_explanation} />}
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Recommendations */}
            {recommendations?.length > 0 && (
                <div className="bg-white rounded-2xl shadow-soft border border-slate-200/60 p-6">
                    <h3 className="text-lg font-display font-bold text-slate-900 mb-4 flex items-center gap-2">
                        <Lightbulb className="w-5 h-5 text-amber-500" />
                        AI Recommendations
                    </h3>
                    <ul className="space-y-3">
                        {recommendations.map((rec, i) => (
                            <li key={i} className="flex items-start gap-3 bg-amber-50 rounded-lg p-3">
                                <span className="bg-amber-200 text-amber-800 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0 mt-0.5">
                                    {i + 1}
                                </span>
                                <p className="text-sm text-amber-900">{rec}</p>
                            </li>
                        ))}
                    </ul>
                </div>
            )}

            {/* Clause List */}
            <div className="bg-white rounded-2xl shadow-soft border border-slate-200/60 p-6">
                <h3 className="text-lg font-display font-bold text-slate-900 mb-4 flex items-center gap-2">
                    <ListChecks className="w-5 h-5 text-indigo-600" />
                    Extracted Clauses ({clauses?.length || 0})
                </h3>
                <div className="space-y-3">
                    {clauses?.map((clause, i) => {
                        const c = RISK_COLORS[clause.risk_level] || RISK_COLORS.low;
                        const isOpen = expandedClause === i;
                        return (
                            <motion.div
                                key={clause.id || i}
                                className={`${c.bg} ${c.border} border rounded-lg overflow-hidden cursor-pointer`}
                                onClick={() => setExpandedClause(isOpen ? null : i)}
                            >
                                <div className="p-4 flex items-center justify-between">
                                    <div className="flex items-center gap-3 flex-1 min-w-0">
                                        <RiskIcon level={clause.risk_level} />
                                        <div className="min-w-0 flex-1">
                                            <div className="flex items-center gap-2 flex-wrap">
                                                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${c.badge}`}>
                                                    {clause.category}
                                                </span>
                                                <span className="text-xs text-slate-400">
                                                    {clause.confidence ? `${(clause.confidence * 100).toFixed(0)}% conf` : ''}
                                                </span>
                                            </div>
                                            <p className="text-sm text-slate-700 mt-1 truncate">{clause.clause_text}</p>
                                        </div>
                                    </div>
                                    {isOpen ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
                                </div>

                                <AnimatePresence>
                                    {isOpen && (
                                        <motion.div
                                            initial={{ height: 0, opacity: 0 }}
                                            animate={{ height: 'auto', opacity: 1 }}
                                            exit={{ height: 0, opacity: 0 }}
                                            className="border-t border-inherit"
                                        >
                                            <div className="p-4 space-y-3">
                                                <div>
                                                    <p className="text-xs font-semibold text-slate-500 mb-1">Full Clause Text</p>
                                                    <p className="text-sm text-slate-700">{clause.clause_text}</p>
                                                </div>
                                                <div>
                                                    <p className="text-xs font-semibold text-slate-500 mb-1">Explanation</p>
                                                    <p className="text-sm text-slate-600">{clause.explanation}</p>
                                                </div>
                                                {/* SHAP for individual clause */}
                                                {clause.shap_tokens && clause.shap_scores && (
                                                    <div>
                                                        <p className="text-xs font-semibold text-slate-500 mb-2">Token Importance (SHAP)</p>
                                                        <div className="flex flex-wrap gap-0.5">
                                                            {clause.shap_tokens.map((token, j) => {
                                                                const score = clause.shap_scores[j] || 0;
                                                                const maxS = Math.max(...clause.shap_scores.map(Math.abs));
                                                                const norm = maxS > 0 ? score / maxS : 0;
                                                                const op = Math.min(Math.abs(norm) * 0.7 + 0.1, 0.85);
                                                                const bg = score > 0
                                                                    ? `rgba(239,68,68,${op})`
                                                                    : `rgba(34,197,94,${op})`;
                                                                return (
                                                                    <span
                                                                        key={j}
                                                                        title={`${score.toFixed(4)}`}
                                                                        className="px-0.5 rounded text-xs font-mono"
                                                                        style={{ backgroundColor: bg, color: Math.abs(norm) > 0.5 ? 'white' : '#374151' }}
                                                                    >
                                                                        {token}
                                                                    </span>
                                                                );
                                                            })}
                                                        </div>
                                                    </div>
                                                )}
                                                <div className="flex gap-4">
                                                    <div>
                                                        <p className="text-xs text-slate-400">Risk Score</p>
                                                        <p className={`text-sm font-bold ${c.text}`}>{(clause.risk_score * 100).toFixed(0)}%</p>
                                                    </div>
                                                    <div>
                                                        <p className="text-xs text-slate-400">Risk Level</p>
                                                        <p className={`text-sm font-bold ${c.text} uppercase`}>{clause.risk_level}</p>
                                                    </div>
                                                </div>
                                            </div>
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </motion.div>
                        );
                    })}
                </div>
            </div>
        </div>
    );
};

export default AnalysisResults;
