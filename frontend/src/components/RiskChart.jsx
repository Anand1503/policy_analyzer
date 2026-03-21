import React from 'react';
import {
    PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip,
    ResponsiveContainer, Legend,
} from 'recharts';

const RISK_COLORS = {
    low: '#10b981',
    medium: '#f59e0b',
    high: '#ef4444',
    critical: '#991b1b',
};

const CATEGORY_COLORS = [
    '#415a77', '#778da9', '#0d1b2a', '#10b981', '#f59e0b',
    '#ef4444', '#6366f1', '#14b8a6', '#f97316', '#ec4899',
];

const CustomTooltip = ({ active, payload }) => {
    if (active && payload?.[0]) {
        return (
            <div className="rounded-lg px-3 py-2 text-xs font-medium shadow-lg"
                style={{ background: '#1b263b', color: '#e0e1dd' }}>
                {payload[0].name}: <strong>{payload[0].value}</strong>
            </div>
        );
    }
    return null;
};

/**
 * Risk Distribution — Donut chart
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
        fill: RISK_COLORS[name] || '#778da9',
    }));

    return (
        <div className="rounded-xl border p-5" style={{ background: 'white', borderColor: 'rgba(119,141,169,0.12)' }}>
            <h3 className="text-sm font-display font-semibold mb-4" style={{ color: '#0d1b2a' }}>
                Risk Distribution
            </h3>
            <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                    <Pie
                        data={data}
                        cx="50%" cy="50%"
                        innerRadius={55} outerRadius={85}
                        paddingAngle={3}
                        dataKey="value"
                    >
                        {data.map((entry, i) => (
                            <Cell key={i} fill={entry.fill} />
                        ))}
                    </Pie>
                    <Tooltip content={<CustomTooltip />} />
                    <Legend
                        verticalAlign="bottom"
                        formatter={(value) => <span style={{ color: '#778da9', fontSize: '12px' }}>{value}</span>}
                    />
                </PieChart>
            </ResponsiveContainer>
        </div>
    );
};

/**
 * Category Breakdown — Horizontal bar chart
 */
export const CategoryChart = ({ categoryBreakdown }) => {
    if (!categoryBreakdown) return null;

    const data = Object.entries(categoryBreakdown).map(([name, count]) => ({
        name: name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
        count,
    }));

    return (
        <div className="rounded-xl border p-5" style={{ background: 'white', borderColor: 'rgba(119,141,169,0.12)' }}>
            <h3 className="text-sm font-display font-semibold mb-4" style={{ color: '#0d1b2a' }}>
                Clause Categories
            </h3>
            <ResponsiveContainer width="100%" height={Math.max(200, data.length * 36)}>
                <BarChart data={data} layout="vertical" margin={{ left: 20, right: 20 }}>
                    <XAxis type="number" tick={{ fontSize: 11, fill: '#778da9' }} axisLine={false} tickLine={false} />
                    <YAxis type="category" dataKey="name" width={130}
                        tick={{ fontSize: 11, fill: '#415a77' }} axisLine={false} tickLine={false} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="count" radius={[0, 4, 4, 0]} barSize={18}>
                        {data.map((_, i) => (
                            <Cell key={i} fill={CATEGORY_COLORS[i % CATEGORY_COLORS.length]} />
                        ))}
                    </Bar>
                </BarChart>
            </ResponsiveContainer>
        </div>
    );
};

/**
 * Compliance Coverage — Simple visual bars
 */
export const ComplianceChart = ({ complianceData }) => {
    if (!complianceData) return null;

    const data = Object.entries(complianceData).map(([key, val]) => ({
        label: key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
        coverage: typeof val === 'number' ? Math.round(val * 100) : Math.round((val?.coverage || 0) * 100),
    }));

    if (data.length === 0) return null;

    const getBarColor = (pct) => {
        if (pct >= 70) return '#10b981';
        if (pct >= 40) return '#f59e0b';
        return '#ef4444';
    };

    return (
        <div className="rounded-xl border p-5" style={{ background: 'white', borderColor: 'rgba(119,141,169,0.12)' }}>
            <h3 className="text-sm font-display font-semibold mb-4" style={{ color: '#0d1b2a' }}>
                Compliance Coverage
            </h3>
            <div className="space-y-3">
                {data.map(item => (
                    <div key={item.label}>
                        <div className="flex items-center justify-between mb-1">
                            <span className="text-xs font-medium" style={{ color: '#415a77' }}>{item.label}</span>
                            <span className="text-xs font-semibold" style={{ color: getBarColor(item.coverage) }}>{item.coverage}%</span>
                        </div>
                        <div className="h-2 rounded-full overflow-hidden" style={{ background: 'rgba(119,141,169,0.08)' }}>
                            <div className="h-full rounded-full transition-all duration-500"
                                style={{ width: `${item.coverage}%`, background: getBarColor(item.coverage) }} />
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

/**
 * SHAP Explanation — Simplified for general users
 */
export const ShapExplanation = ({ explanation }) => {
    if (!explanation?.tokens || !explanation?.scores) return null;

    const { tokens, scores } = explanation;
    const maxScore = Math.max(...scores.map(Math.abs));

    return (
        <div className="rounded-xl border p-5" style={{ background: 'white', borderColor: 'rgba(119,141,169,0.12)' }}>
            <h3 className="text-sm font-display font-semibold mb-1" style={{ color: '#0d1b2a' }}>
                Key Phrase Impact
            </h3>
            <p className="text-xs mb-4" style={{ color: '#778da9' }}>
                Words highlighted in <span style={{ color: '#ef4444' }}>red</span> increase risk. Words in <span style={{ color: '#10b981' }}>green</span> reduce risk.
            </p>
            <div className="flex flex-wrap gap-0.5 leading-relaxed">
                {tokens.map((token, i) => {
                    const score = scores[i] || 0;
                    const normalized = maxScore > 0 ? score / maxScore : 0;
                    const opacity = Math.min(Math.abs(normalized) * 0.65 + 0.1, 0.8);
                    const bg = score > 0
                        ? `rgba(239,68,68,${opacity})`
                        : `rgba(16,185,129,${opacity})`;
                    const textColor = Math.abs(normalized) > 0.5 ? 'white' : '#0d1b2a';
                    return (
                        <span key={i} className="inline-block px-1 py-0.5 rounded text-xs"
                            style={{ backgroundColor: bg, color: textColor }}>
                            {token}
                        </span>
                    );
                })}
            </div>
        </div>
    );
};
