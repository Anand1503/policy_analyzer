import React, { useState, useEffect, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import { analysisAPI } from '../services/api';
import AnalysisResults from '../components/AnalysisResults';
import { BarChart3, Loader2, RefreshCw, AlertCircle } from 'lucide-react';

// ── Skeleton Loading Components ───────────────────────────────

const SkeletonPulse = ({ width = '100%', height = '16px', borderRadius = '8px', style = {} }) => (
    <div
        style={{
            width,
            height,
            borderRadius,
            background: 'linear-gradient(90deg, var(--color-card-border) 25%, rgba(119,141,169,0.15) 50%, var(--color-card-border) 75%)',
            backgroundSize: '200% 100%',
            animation: 'skeleton-shimmer 1.5s ease-in-out infinite',
            ...style,
        }}
    />
);

const SkeletonCard = ({ children }) => (
    <div style={{
        background: 'var(--color-card)',
        border: '1px solid var(--color-card-border)',
        borderRadius: '16px',
        padding: '24px',
        marginBottom: '16px',
    }}>
        {children}
    </div>
);

const AnalysisSkeletonUI = () => (
    <div style={{ maxWidth: '900px', margin: '0 auto' }}>
        <style>{`
            @keyframes skeleton-shimmer {
                0%   { background-position: -200% 0; }
                100% { background-position:  200% 0; }
            }
        `}</style>

        {/* Summary card skeleton */}
        <SkeletonCard>
            <div style={{ display: 'flex', gap: '24px', alignItems: 'center' }}>
                {/* Score ring skeleton */}
                <div style={{
                    width: '112px', height: '112px', borderRadius: '50%', flexShrink: 0,
                    background: 'linear-gradient(90deg, var(--color-card-border) 25%, rgba(119,141,169,0.15) 50%, var(--color-card-border) 75%)',
                    backgroundSize: '200% 100%',
                    animation: 'skeleton-shimmer 1.5s ease-in-out infinite',
                }} />
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    <SkeletonPulse width="40%" height="22px" />
                    <SkeletonPulse width="100%" height="14px" />
                    <SkeletonPulse width="85%" height="14px" />
                    <SkeletonPulse width="60%" height="14px" />
                    <div style={{ display: 'flex', gap: '12px', marginTop: '8px' }}>
                        <SkeletonPulse width="80px" height="36px" borderRadius="10px" />
                        <SkeletonPulse width="80px" height="36px" borderRadius="10px" />
                        <SkeletonPulse width="80px" height="36px" borderRadius="10px" />
                    </div>
                </div>
            </div>
        </SkeletonCard>

        {/* Charts row skeleton */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '16px' }}>
            {[0, 1].map(i => (
                <SkeletonCard key={i}>
                    <SkeletonPulse width="50%" height="16px" style={{ marginBottom: '20px' }} />
                    <SkeletonPulse width="100%" height="140px" borderRadius="12px" />
                </SkeletonCard>
            ))}
        </div>

        {/* Recommendations skeleton */}
        <SkeletonCard>
            <SkeletonPulse width="35%" height="16px" style={{ marginBottom: '16px' }} />
            {[0, 1, 2].map(i => (
                <div key={i} style={{ display: 'flex', gap: '12px', marginBottom: '10px', alignItems: 'flex-start' }}>
                    <SkeletonPulse width="20px" height="20px" borderRadius="50%" style={{ flexShrink: 0 }} />
                    <SkeletonPulse width={`${70 + (i * 10)}%`} height="14px" />
                </div>
            ))}
        </SkeletonCard>

        {/* Clause list skeleton */}
        <SkeletonCard>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px' }}>
                <SkeletonPulse width="25%" height="16px" />
                <SkeletonPulse width="200px" height="30px" borderRadius="8px" />
            </div>
            {[0, 1, 2, 3, 4].map(i => (
                <div key={i} style={{
                    padding: '16px 0',
                    borderTop: i > 0 ? '1px solid rgba(119,141,169,0.06)' : 'none',
                    display: 'flex', gap: '12px', alignItems: 'flex-start',
                }}>
                    <SkeletonPulse width="18px" height="18px" borderRadius="50%" style={{ flexShrink: 0, marginTop: '2px' }} />
                    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        <div style={{ display: 'flex', gap: '8px' }}>
                            <SkeletonPulse width="70px" height="20px" borderRadius="20px" />
                            <SkeletonPulse width="120px" height="20px" borderRadius="20px" />
                        </div>
                        <SkeletonPulse width="100%" height="13px" />
                        <SkeletonPulse width="80%" height="13px" />
                    </div>
                </div>
            ))}
        </SkeletonCard>

        {/* Status message */}
        <div style={{ textAlign: 'center', padding: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px' }}>
            <Loader2
                style={{ width: '18px', height: '18px', color: 'var(--color-accent)', animation: 'spin 1s linear infinite' }}
            />
            <p style={{ color: 'var(--color-denim)', fontSize: '14px', fontWeight: '500' }}>
                Loading analysis results...
            </p>
        </div>
    </div>
);

// ── Empty State ───────────────────────────────────────────────

const EmptyState = () => (
    <div style={{ textAlign: 'center', padding: '80px 24px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px' }}>
        <div style={{
            width: '64px', height: '64px', borderRadius: '18px',
            background: 'rgba(119,141,169,0.08)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
            <BarChart3 style={{ width: '28px', height: '28px', color: 'var(--color-denim)', opacity: 0.5 }} />
        </div>
        <div>
            <p style={{ fontWeight: '600', fontSize: '16px', color: 'var(--color-ink)', margin: 0 }}>
                No analysis selected
            </p>
            <p style={{ fontSize: '14px', color: 'var(--color-denim)', marginTop: '6px' }}>
                Select a document and click "Analyze" to see results.
            </p>
        </div>
    </div>
);

// ── Error State ───────────────────────────────────────────────

const ErrorState = ({ error, onRetry }) => (
    <div style={{
        borderRadius: '16px', border: '1px solid rgba(239,68,68,0.2)',
        background: 'rgba(239,68,68,0.03)', padding: '32px 24px',
        display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px',
        textAlign: 'center',
    }}>
        <AlertCircle style={{ width: '32px', height: '32px', color: 'var(--color-danger)' }} />
        <div>
            <p style={{ fontWeight: '600', color: 'var(--color-danger)', margin: 0 }}>Failed to load results</p>
            <p style={{ fontSize: '14px', color: 'var(--color-dusk)', marginTop: '6px' }}>{error}</p>
        </div>
        {onRetry && (
            <button
                onClick={onRetry}
                style={{
                    display: 'flex', alignItems: 'center', gap: '8px',
                    padding: '10px 20px', borderRadius: '10px',
                    border: '1.5px solid var(--color-card-border)',
                    background: 'var(--color-card)', color: 'var(--color-ink)',
                    fontSize: '14px', fontWeight: '600', cursor: 'pointer',
                    transition: 'all 0.2s',
                }}
            >
                <RefreshCw style={{ width: '15px', height: '15px' }} />
                Retry
            </button>
        )}
    </div>
);

// ── Main Page Component ───────────────────────────────────────

const AnalysisResult = () => {
    const [searchParams] = useSearchParams();
    const docId = searchParams.get('docId');
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const fetchResults = useCallback(async (id) => {
        setLoading(true);
        setError('');
        try {
            const res = await analysisAPI.results(id);
            setData(res.data);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to fetch analysis results.');
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        if (docId) {
            fetchResults(docId);
        }
    }, [docId, fetchResults]);

    return (
        <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '0 0 48px' }}>
            <style>{`
                @keyframes spin {
                    from { transform: rotate(0deg); }
                    to   { transform: rotate(360deg); }
                }
            `}</style>

            {loading && <AnalysisSkeletonUI />}

            {!loading && error && (
                <ErrorState error={error} onRetry={docId ? () => fetchResults(docId) : null} />
            )}

            {!loading && !error && data && (
                <AnalysisResults data={data} />
            )}

            {!loading && !error && !data && (
                <EmptyState />
            )}
        </div>
    );
};

export default AnalysisResult;
