import React, { useState } from 'react';
import {
    ShieldAlert, ShieldCheck, AlertTriangle, ChevronDown, ChevronUp,
    FileText, Lightbulb, ListChecks, TrendingUp,
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { RiskDistributionChart, CategoryChart, ComplianceChart, ShapExplanation } from './RiskChart';

/* ── Helpers ─────────────────────────────────── */

const riskStyle = (level) => {
    const map = {
        low:      { bg: 'rgba(16,185,129,0.06)', border: 'rgba(16,185,129,0.15)', text: 'var(--color-success)', label: 'Low Risk' },
        medium:   { bg: 'rgba(245,158,11,0.06)', border: 'rgba(245,158,11,0.15)', text: 'var(--color-warning)', label: 'Medium Risk' },
        high:     { bg: 'rgba(239,68,68,0.06)',   border: 'rgba(239,68,68,0.15)',  text: 'var(--color-danger)', label: 'High Risk' },
        critical: { bg: 'rgba(153,27,27,0.08)',   border: 'rgba(153,27,27,0.2)',   text: '#991b1b', label: 'Critical' },
    };
    return map[level] || map.low;
};

const RiskIcon = ({ level, size = 18 }) => {
    const props = { width: size, height: size };
    if (level === 'low') return <ShieldCheck {...props} style={{ color: 'var(--color-success)' }} />;
    if (level === 'high' || level === 'critical') return <ShieldAlert {...props} style={{ color: 'var(--color-danger)' }} />;
    return <AlertTriangle {...props} style={{ color: 'var(--color-warning)' }} />;
};

const formatCategory = (cat) =>
    (cat || '').replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

/* ── Main Component ──────────────────────────── */

const AnalysisResults = ({ data }) => {
    const [expandedClause, setExpandedClause] = useState(null);
    const [filter, setFilter] = useState('all'); // all | high | medium | low

    if (!data) return null;

    const { summary, overall_risk, overall_score, recommendations, category_breakdown, clauses, compliance, shap_explanation } = data;
    const rs = riskStyle(overall_risk);
    const scorePercent = Math.round((overall_score || 0) * 100);

    // Filter clauses
    const filteredClauses = filter === 'all'
        ? clauses
        : clauses?.filter(c => c.risk_level === filter || (filter === 'high' && c.risk_level === 'critical'));

    // Risk counts for filter tabs
    const riskCounts = clauses?.reduce((acc, c) => {
        acc[c.risk_level] = (acc[c.risk_level] || 0) + 1;
        return acc;
    }, {}) || {};

    return (
        <div className="space-y-5">

            {/* ── 1. Summary Card ──────────────────────── */}
            <div className="rounded-xl border p-6" style={{ background: 'var(--color-card)', borderColor: 'var(--color-card-border)' }}>
                <div className="flex flex-col md:flex-row md:items-center gap-6">
                    {/* Score ring */}
                    <div className="flex-shrink-0">
                        <div className="relative w-28 h-28">
                            <svg className="w-28 h-28 -rotate-90" viewBox="0 0 100 100">
                                <circle cx="50" cy="50" r="42" fill="none" stroke="rgba(119,141,169,0.08)" strokeWidth="8" />
                                <circle cx="50" cy="50" r="42" fill="none"
                                    stroke={rs.text} strokeWidth="8" strokeLinecap="round"
                                    strokeDasharray={`${(overall_score || 0) * 263.9} 263.9`}
                                    style={{ transition: 'stroke-dasharray 1s ease' }}
                                />
                            </svg>
                            <div className="absolute inset-0 flex flex-col items-center justify-center">
                                <span className="text-2xl font-display font-bold" style={{ color: 'var(--color-ink)' }}>
                                    {scorePercent}
                                </span>
                                <span className="text-[10px] font-medium" style={{ color: 'var(--color-denim)' }}>/ 100</span>
                            </div>
                        </div>
                    </div>

                    {/* Summary text */}
                    <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-2">
                            <h2 className="text-lg font-display font-bold" style={{ color: 'var(--color-ink)' }}>Analysis Report</h2>
                            <span className="text-xs font-semibold px-2.5 py-1 rounded-full"
                                style={{ background: rs.bg, color: rs.text, border: `1px solid ${rs.border}` }}>
                                {rs.label}
                            </span>
                        </div>
                        <p className="text-sm leading-relaxed" style={{ color: 'var(--color-dusk)' }}>
                            {summary || 'Analysis complete. See detailed findings below.'}
                        </p>

                        {/* Quick stats row */}
                        <div className="flex flex-wrap gap-3 mt-4">
                            {[
                                { label: 'Clauses Found', value: clauses?.length || 0 },
                                { label: 'High/Critical', value: (riskCounts.high || 0) + (riskCounts.critical || 0) },
                                { label: 'Categories', value: category_breakdown ? Object.keys(category_breakdown).length : 0 },
                            ].map(s => (
                                <div key={s.label} className="px-3 py-2 rounded-lg" style={{ background: 'rgba(119,141,169,0.05)' }}>
                                    <span className="text-base font-display font-bold" style={{ color: 'var(--color-ink)' }}>{s.value}</span>
                                    <span className="text-[11px] ml-1.5" style={{ color: 'var(--color-denim)' }}>{s.label}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>

            {/* ── 2. Charts Row ────────────────────────── */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <RiskDistributionChart clauses={clauses} />
                <CategoryChart categoryBreakdown={category_breakdown} />
                {compliance && <ComplianceChart complianceData={compliance} />}
                {shap_explanation && <ShapExplanation explanation={shap_explanation} />}
            </div>

            {/* ── 3. Recommendations ───────────────────── */}
            {recommendations?.length > 0 && (
                <div className="rounded-xl border p-5" style={{ background: 'var(--color-card)', borderColor: 'var(--color-card-border)' }}>
                    <h3 className="text-sm font-display font-semibold mb-4 flex items-center gap-2" style={{ color: 'var(--color-ink)' }}>
                        <Lightbulb className="w-4 h-4" style={{ color: 'var(--color-warning)' }} />
                        Recommendations
                    </h3>
                    <div className="space-y-2">
                        {recommendations.map((rec, i) => (
                            <div key={i} className="flex items-start gap-3 p-3 rounded-lg"
                                style={{ background: 'rgba(245,158,11,0.04)', border: '1px solid rgba(245,158,11,0.08)' }}>
                                <span className="flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold"
                                    style={{ background: 'rgba(245,158,11,0.12)', color: 'var(--color-warning)' }}>
                                    {i + 1}
                                </span>
                                <p className="text-sm leading-relaxed" style={{ color: 'var(--color-prussian)' }}>{rec}</p>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* ── 4. Clause Details ────────────────────── */}
            <div className="rounded-xl border" style={{ background: 'var(--color-card)', borderColor: 'var(--color-card-border)' }}>
                <div className="p-5" style={{ borderBottom: '1px solid rgba(119,141,169,0.08)' }}>
                    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                        <h3 className="text-sm font-display font-semibold flex items-center gap-2" style={{ color: 'var(--color-ink)' }}>
                            <ListChecks className="w-4 h-4" style={{ color: 'var(--color-dusk)' }} />
                            Clause Details
                            <span className="text-xs font-normal px-2 py-0.5 rounded-full" style={{ background: 'var(--color-card-border)', color: 'var(--color-denim)' }}>
                                {filteredClauses?.length || 0}
                            </span>
                        </h3>

                        {/* Filter tabs */}
                        <div className="flex gap-1 p-0.5 rounded-lg" style={{ background: 'rgba(119,141,169,0.05)' }}>
                            {[
                                { key: 'all', label: 'All' },
                                { key: 'high', label: 'High Risk', color: 'var(--color-danger)' },
                                { key: 'medium', label: 'Medium', color: 'var(--color-warning)' },
                                { key: 'low', label: 'Low', color: 'var(--color-success)' },
                            ].map(tab => (
                                <button key={tab.key}
                                    onClick={() => setFilter(tab.key)}
                                    className="text-[11px] font-medium px-2.5 py-1 rounded-md transition-all"
                                    style={{
                                        background: filter === tab.key ? 'var(--color-card)' : 'transparent',
                                        color: filter === tab.key ? (tab.color || 'var(--color-ink)') : 'var(--color-denim)',
                                        boxShadow: filter === tab.key ? '0 1px 3px rgba(13,27,42,0.06)' : 'none',
                                    }}>
                                    {tab.label}
                                    {tab.key !== 'all' && riskCounts[tab.key] ? ` (${riskCounts[tab.key]})` : ''}
                                </button>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Clause list */}
                <div>
                    {filteredClauses?.length === 0 ? (
                        <div className="p-8 text-center">
                            <ShieldCheck className="w-8 h-8 mx-auto mb-2" style={{ color: 'var(--color-success)' }} />
                            <p className="text-sm" style={{ color: 'var(--color-denim)' }}>No clauses match this filter.</p>
                        </div>
                    ) : (
                        filteredClauses?.map((clause, i) => {
                            const cs = riskStyle(clause.risk_level);
                            const isOpen = expandedClause === i;

                            return (
                                <div key={clause.id || i}
                                    style={{ borderTop: i > 0 ? '1px solid rgba(119,141,169,0.06)' : 'none' }}>
                                    {/* Collapsed row */}
                                    <div className="px-5 py-4 flex items-start gap-3 cursor-pointer transition-colors hover:bg-gray-50/50"
                                        onClick={() => setExpandedClause(isOpen ? null : i)}>
                                        <RiskIcon level={clause.risk_level} size={16} />

                                        <div className="flex-1 min-w-0">
                                            <div className="flex items-center gap-2 mb-1 flex-wrap">
                                                <span className="text-xs font-medium px-2 py-0.5 rounded-full"
                                                    style={{ background: cs.bg, color: cs.text, border: `1px solid ${cs.border}` }}>
                                                    {cs.label}
                                                </span>
                                                <span className="text-xs font-medium px-2 py-0.5 rounded-full"
                                                    style={{ background: 'rgba(65,90,119,0.06)', color: 'var(--color-dusk)' }}>
                                                    {formatCategory(clause.category)}
                                                </span>
                                            </div>
                                            <p className="text-sm leading-relaxed" style={{ color: 'var(--color-prussian)' }}>
                                                {isOpen ? clause.clause_text : (clause.clause_text?.length > 200 ? clause.clause_text.slice(0, 200) + '…' : clause.clause_text)}
                                            </p>
                                        </div>

                                        <div className="flex-shrink-0 flex items-center gap-2">
                                            <span className="text-xs font-bold" style={{ color: cs.text }}>
                                                {Math.round((clause.risk_score || 0) * 100)}%
                                            </span>
                                            {isOpen
                                                ? <ChevronUp className="w-4 h-4" style={{ color: 'var(--color-denim)' }} />
                                                : <ChevronDown className="w-4 h-4" style={{ color: 'var(--color-denim)' }} />
                                            }
                                        </div>
                                    </div>

                                    {/* Expanded details */}
                                    <AnimatePresence>
                                        {isOpen && (
                                            <motion.div
                                                initial={{ height: 0, opacity: 0 }}
                                                animate={{ height: 'auto', opacity: 1 }}
                                                exit={{ height: 0, opacity: 0 }}
                                                transition={{ duration: 0.2 }}
                                            >
                                                <div className="px-5 pb-4 ml-7 space-y-3">
                                                    {/* Explanation */}
                                                    {clause.explanation && (
                                                        <div className="p-3 rounded-lg" style={{ background: 'rgba(119,141,169,0.04)' }}>
                                                            <p className="text-xs font-semibold mb-1" style={{ color: 'var(--color-dusk)' }}>
                                                                Why this matters
                                                            </p>
                                                            <p className="text-sm leading-relaxed" style={{ color: 'var(--color-prussian)' }}>
                                                                {clause.explanation}
                                                            </p>
                                                        </div>
                                                    )}

                                                    {/* Key phrases (SHAP tokens simplified) */}
                                                    {clause.shap_tokens && clause.shap_scores && (
                                                        <div>
                                                            <p className="text-xs font-semibold mb-1.5" style={{ color: 'var(--color-dusk)' }}>
                                                                Key phrases that affect risk
                                                            </p>
                                                            <div className="flex flex-wrap gap-0.5 p-3 rounded-lg" style={{ background: 'rgba(119,141,169,0.03)' }}>
                                                                {clause.shap_tokens.map((token, j) => {
                                                                    const score = clause.shap_scores[j] || 0;
                                                                    const maxS = Math.max(...clause.shap_scores.map(Math.abs));
                                                                    const norm = maxS > 0 ? score / maxS : 0;
                                                                    const op = Math.min(Math.abs(norm) * 0.6 + 0.1, 0.75);
                                                                    const bg = score > 0 ? `rgba(239,68,68,${op})` : `rgba(16,185,129,${op})`;
                                                                    return (
                                                                        <span key={j}
                                                                            className="inline-block px-1 py-0.5 rounded text-xs"
                                                                            style={{ backgroundColor: bg, color: Math.abs(norm) > 0.5 ? 'var(--color-card)' : 'var(--color-ink)' }}>
                                                                            {token}
                                                                        </span>
                                                                    );
                                                                })}
                                                            </div>
                                                            <p className="text-[10px] mt-1" style={{ color: 'var(--color-denim)' }}>
                                                                <span style={{ color: 'var(--color-danger)' }}>■</span> increases risk &nbsp;
                                                                <span style={{ color: 'var(--color-success)' }}>■</span> reduces risk
                                                            </p>
                                                        </div>
                                                    )}

                                                    {/* Meta row */}
                                                    <div className="flex gap-4 pt-1">
                                                        <div>
                                                            <p className="text-[10px]" style={{ color: 'var(--color-denim)' }}>Risk Score</p>
                                                            <p className="text-sm font-bold" style={{ color: cs.text }}>
                                                                {Math.round((clause.risk_score || 0) * 100)}%
                                                            </p>
                                                        </div>
                                                        {clause.confidence && (
                                                            <div>
                                                                <p className="text-[10px]" style={{ color: 'var(--color-denim)' }}>Confidence</p>
                                                                <p className="text-sm font-bold" style={{ color: 'var(--color-dusk)' }}>
                                                                    {Math.round(clause.confidence * 100)}%
                                                                </p>
                                                            </div>
                                                        )}
                                                    </div>
                                                </div>
                                            </motion.div>
                                        )}
                                    </AnimatePresence>
                                </div>
                            );
                        })
                    )}
                </div>
            </div>
        </div>
    );
};

export default AnalysisResults;
