import React from 'react';
import {
    PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip,
    ResponsiveContainer, RadarChart, Radar, PolarGrid,
    PolarAngleAxis, PolarRadiusAxis, Legend,
} from 'recharts';
import { BarChart3 } from 'lucide-react';

const COLORS = [
    '#3b82f6', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b',
    '#ef4444', '#ec4899', '#6366f1', '#14b8a6', '#f97316',
];

const RISK_COLORS = {
    low: '#22c55e',
    medium: '#eab308',
    high: '#ef4444',
    critical: '#991b1b',
};

/**
 * Risk Distribution Pie Chart
 */
export const RiskDistributionChart = ({ clauses }) => {
    if (!clauses?.length) return null;

    const riskCounts = clauses.reduce((acc, c) => {
        const level = c.risk_level || 'low';
        acc[level] = (acc[level] || 0) + 1;
        return acc;
    }, {});

    const data = Object.entries(riskCounts).map(([name, value]) => ({
        name: name.charAt(0).toUpperCase() + name.slice(1),
        value,
        fill: RISK_COLORS[name] || '#94a3b8',
    }));

    return (
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
            <h3 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-blue-600" />
                Risk Distribution
            </h3>
            <ResponsiveContainer width="100%" height={280}>
                <PieChart>
                    <Pie
                        data={data}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={100}
                        paddingAngle={4}
                        dataKey="value"
                        label={({ name, value }) => `${name} (${value})`}
                    >
                        {data.map((entry, i) => (
                            <Cell key={i} fill={entry.fill} />
                        ))}
                    </Pie>
                    <Tooltip />
                    <Legend />
                </PieChart>
            </ResponsiveContainer>
        </div>
    );
};

/**
 * Clause Category Bar Chart
 */
export const CategoryChart = ({ categoryBreakdown }) => {
    if (!categoryBreakdown) return null;

    const data = Object.entries(categoryBreakdown).map(([name, count]) => ({
        name: name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
        count,
    }));

    return (
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
            <h3 className="text-lg font-bold text-slate-800 mb-4">
                Clause Categories
            </h3>
            <ResponsiveContainer width="100%" height={300}>
                <BarChart data={data} layout="vertical" margin={{ left: 100 }}>
                    <XAxis type="number" />
                    <YAxis
                        type="category"
                        dataKey="name"
                        width={120}
                        tick={{ fontSize: 11 }}
                    />
                    <Tooltip />
                    <Bar dataKey="count" radius={[0, 6, 6, 0]}>
                        {data.map((_, i) => (
                            <Cell key={i} fill={COLORS[i % COLORS.length]} />
                        ))}
                    </Bar>
                </BarChart>
            </ResponsiveContainer>
        </div>
    );
};

/**
 * Compliance Coverage Radar Chart
 */
export const ComplianceChart = ({ complianceData }) => {
    if (!complianceData) return null;

    // Build radar data from compliance coverage
    const data = Object.entries(complianceData).map(([key, val]) => ({
        subject: key.replace(/_/g, ' ').slice(0, 20),
        coverage: typeof val === 'number' ? val * 100 : (val?.coverage || 0) * 100,
        fullMark: 100,
    }));

    if (data.length === 0) return null;

    return (
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
            <h3 className="text-lg font-bold text-slate-800 mb-4">
                Compliance Coverage
            </h3>
            <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={data}>
                    <PolarGrid stroke="#e2e8f0" />
                    <PolarAngleAxis dataKey="subject" tick={{ fontSize: 10 }} />
                    <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10 }} />
                    <Radar
                        name="Coverage %"
                        dataKey="coverage"
                        stroke="#6366f1"
                        fill="#6366f1"
                        fillOpacity={0.3}
                    />
                    <Tooltip />
                    <Legend />
                </RadarChart>
            </ResponsiveContainer>
        </div>
    );
};

/**
 * SHAP Explanation Visualization
 */
export const ShapExplanation = ({ explanation }) => {
    if (!explanation?.tokens || !explanation?.scores) return null;

    const { tokens, scores } = explanation;
    const maxScore = Math.max(...scores.map(Math.abs));

    return (
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
            <h3 className="text-lg font-bold text-slate-800 mb-4">
                SHAP Token Importance
            </h3>
            <div className="flex flex-wrap gap-1 leading-relaxed">
                {tokens.map((token, i) => {
                    const score = scores[i] || 0;
                    const normalized = maxScore > 0 ? score / maxScore : 0;
                    const opacity = Math.min(Math.abs(normalized) * 0.8 + 0.1, 0.9);
                    const bg = score > 0
                        ? `rgba(239, 68, 68, ${opacity})`  // Red for positive (risky)
                        : `rgba(34, 197, 94, ${opacity})`;   // Green for negative (safe)
                    const textColor = Math.abs(normalized) > 0.5 ? 'white' : '#374151';

                    return (
                        <span
                            key={i}
                            title={`Score: ${score.toFixed(4)}`}
                            className="inline-block px-1 py-0.5 rounded text-sm font-mono cursor-help transition-transform hover:scale-110"
                            style={{ backgroundColor: bg, color: textColor }}
                        >
                            {token}
                        </span>
                    );
                })}
            </div>
            <div className="flex items-center gap-4 mt-4 text-xs text-slate-500">
                <div className="flex items-center gap-1">
                    <span className="w-4 h-3 rounded" style={{ backgroundColor: 'rgba(34, 197, 94, 0.5)' }} />
                    Low risk
                </div>
                <div className="flex items-center gap-1">
                    <span className="w-4 h-3 rounded" style={{ backgroundColor: 'rgba(239, 68, 68, 0.5)' }} />
                    High risk
                </div>
            </div>
        </div>
    );
};
